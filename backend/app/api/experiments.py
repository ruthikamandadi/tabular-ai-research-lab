"""
Experiment creation, tracking, results, and explainability endpoints.
"""
import uuid
import json
import joblib
import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db, ExperimentRecord, DatasetRecord
from app.core.config import ARTIFACTS_DIR, UPLOADS_DIR
from app.models.model_factory import get_available_models
from app.core.datetime_utils import parse_datetime
from app.experiments.runner import run_training_job, get_job
from app.explainability.importance import compute_model_importances, compute_local_explanation

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


class ExperimentCreate(BaseModel):
    dataset_id: str
    target_column: str
    task_type: str
    model_id: str
    model_config: dict = {}
    preprocessing_config: dict = {}
    random_seed: int = 42


@router.get("/models/available")
def get_models():
    return get_available_models()


@router.post("")
def create_experiment(body: ExperimentCreate, db: Session = Depends(get_db)):
    # Validate dataset
    dataset = db.query(DatasetRecord).filter(DatasetRecord.id == body.dataset_id).first()
    if not dataset:
        raise HTTPException(404, "Dataset not found.")

    # Validate task type
    valid_tasks = ["binary_classification", "multiclass_classification", "regression", "time_series"]
    if body.task_type not in valid_tasks:
        raise HTTPException(400, f"task_type must be one of: {valid_tasks}")
    if body.task_type == "time_series":
        date_column = body.preprocessing_config.get("time_series_date_column")
        if not date_column:
            raise HTTPException(400, "Choose a date column for time-series forecasting.")
        if date_column == body.target_column:
            raise HTTPException(400, "The time-series date column and target column must be different.")
        try:
            frame = pd.read_csv(dataset.filepath, usecols=[date_column, body.target_column])
        except ValueError as exc:
            raise HTTPException(400, f"Date or target column is not present in the dataset: {exc}") from exc
        except Exception as exc:
            raise HTTPException(400, f"Could not validate time-series columns: {exc}") from exc
        if not pd.api.types.is_numeric_dtype(frame[body.target_column]):
            raise HTTPException(400, "Time-series target must be numeric.")
        parsed_dates = parse_datetime(frame[date_column])
        if parsed_dates.isna().any():
            raise HTTPException(400, f"Date column '{date_column}' contains invalid or missing dates.")
        if not parsed_dates.is_monotonic_increasing:
            raise HTTPException(400, f"Date column '{date_column}' must be sorted in ascending chronological order.")

    # Validate model for task
    available = get_available_models()
    model_info = next((m for m in available if m["id"] == body.model_id), None)
    if not model_info:
        raise HTTPException(400, f"Unknown model: {body.model_id}")
    if body.task_type not in model_info["tasks"]:
        raise HTTPException(400, f"Model '{body.model_id}' does not support task '{body.task_type}'.")

    experiment_id = str(uuid.uuid4())

    exp = ExperimentRecord(
        id=experiment_id,
        dataset_id=body.dataset_id,
        dataset_name=dataset.name,
        target_column=body.target_column,
        task_type=body.task_type,
        model_name=model_info["name"],
        model_config=body.model_config,
        preprocessing_config=body.preprocessing_config,
        train_size=1.0 - body.preprocessing_config.get("test_size", 0.2),
        test_size=body.preprocessing_config.get("test_size", 0.2),
        random_seed=body.random_seed,
        status="pending",
    )
    db.add(exp)
    db.commit()

    # Start background job
    job_config = {
        "dataset_path": dataset.filepath,
        "target_column": body.target_column,
        "task_type": body.task_type,
        "model_id": body.model_id,
        "model_config": body.model_config,
        "preprocessing_config": body.preprocessing_config,
        "random_seed": body.random_seed,
        "experiment_id": experiment_id,
    }
    run_training_job(experiment_id, job_config)

    return {"experiment_id": experiment_id, "status": "pending", "message": "Training started."}


@router.get("")
def list_experiments(db: Session = Depends(get_db)):
    records = db.query(ExperimentRecord).order_by(ExperimentRecord.created_at.desc()).all()
    return [_serialize_experiment(r) for r in records]


@router.get("/{experiment_id}")
def get_experiment(experiment_id: str, db: Session = Depends(get_db)):
    record = db.query(ExperimentRecord).filter(ExperimentRecord.id == experiment_id).first()
    if not record:
        raise HTTPException(404, "Experiment not found.")
    return _serialize_experiment(record)


@router.get("/{experiment_id}/status")
def get_experiment_status(experiment_id: str, db: Session = Depends(get_db)):
    job = get_job(experiment_id)
    if job:
        return {
            "experiment_id": experiment_id,
            "status": job.get("status"),
            "progress": job.get("progress", 0),
            "elapsed_seconds": round(max(0, time.time() - job.get("started_at", time.time())), 1),
            "message": job.get("message", ""),
            "training_history": job.get("training_history", []),
            "device": job.get("device"),
            "warning": job.get("warning"),
            "best_params": job.get("best_params"),
            "best_cv_score": job.get("best_cv_score"),
            "tuning_trials": job.get("tuning_trials"),
        }
    # Fall back to DB
    record = db.query(ExperimentRecord).filter(ExperimentRecord.id == experiment_id).first()
    if not record:
        raise HTTPException(404, "Experiment not found.")
    return {"experiment_id": experiment_id, "status": record.status, "progress": 100 if record.status == "completed" else 0, "message": ""}


@router.get("/{experiment_id}/results")
def get_experiment_results(experiment_id: str, db: Session = Depends(get_db)):
    record = db.query(ExperimentRecord).filter(ExperimentRecord.id == experiment_id).first()
    if not record:
        raise HTTPException(404, "Experiment not found.")
    if record.status != "completed":
        raise HTTPException(400, f"Experiment status is '{record.status}', not completed.")
    return {
        "metrics": record.metrics,
        "training_history": record.training_history,
        "n_features": record.n_features,
        "n_train_rows": record.n_train_rows,
        "n_test_rows": record.n_test_rows,
        "train_duration_s": record.train_duration_s,
        "inference_duration_s": record.inference_duration_s,
    }


@router.get("/{experiment_id}/explanations")
def get_explanations(experiment_id: str, db: Session = Depends(get_db)):
    record = db.query(ExperimentRecord).filter(ExperimentRecord.id == experiment_id).first()
    if not record or record.status != "completed":
        raise HTTPException(400, "Experiment not completed.")

    artifact_dir = Path(record.artifact_path)
    meta_path = artifact_dir / "meta.json"
    if not meta_path.exists():
        raise HTTPException(500, "Artifact metadata missing.")

    with open(meta_path) as f:
        meta = json.load(f)

    # Load model and preprocessor
    try:
        preprocessor = joblib.load(str(artifact_dir / "preprocessor.pkl"))
        from app.models.model_factory import load_model_wrapper
        model = load_model_wrapper(
            str(artifact_dir / ("model.pt" if meta.get("model_type") == "pytorch_mlp" else "model.pkl")),
            meta.get("model_type", "sklearn"),
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to load model artifacts: {e}")

    # Load dataset
    dataset = db.query(DatasetRecord).filter(DatasetRecord.id == record.dataset_id).first()
    if not dataset:
        raise HTTPException(404, "Original dataset not found.")

    df = pd.read_csv(dataset.filepath)
    input_cols = meta.get("input_feature_cols", [])
    X = df[input_cols].head(1000)  # sample for speed
    X_transformed = preprocessor.transform(X)

    target = df[record.target_column].head(1000)
    if record.task_type not in ("regression", "time_series"):
        from sklearn.preprocessing import LabelEncoder
        le_path = artifact_dir / "label_encoder.pkl"
        if le_path.exists():
            le = joblib.load(str(le_path))
            y = le.transform(target.astype(str))
        else:
            y = target.values
    else:
        y = pd.to_numeric(target, errors="coerce").fillna(0).values

    feature_names = meta.get("feature_names", [f"f_{i}" for i in range(X_transformed.shape[1])])

    if meta.get("model_type") == "tabnet":
        attributions, _ = model.explain(X_transformed)
        mean_attributions = np.abs(np.asarray(attributions)).mean(axis=0)
        ranked_importances = [
            {
                "feature": feature_names[index] if index < len(feature_names) else f"feature_{index}",
                "importance": round(float(value), 6),
                "method": "TabNet explain",
            }
            for index, value in enumerate(mean_attributions)
        ]
        ranked_importances.sort(key=lambda item: item["importance"], reverse=True)
        importance = {
            "method": "TabNet explain",
            "importances": ranked_importances[:30],
            "fallback_used": False,
        }
    else:
        importance = compute_model_importances(
            model, X_transformed, X_transformed, y, feature_names, record.task_type
        )
    return importance


@router.delete("/{experiment_id}")
def delete_experiment(experiment_id: str, db: Session = Depends(get_db)):
    record = db.query(ExperimentRecord).filter(ExperimentRecord.id == experiment_id).first()
    if not record:
        raise HTTPException(404, "Experiment not found.")
    db.delete(record)
    db.commit()
    return {"message": "Experiment deleted."}


def _serialize_experiment(r: ExperimentRecord) -> dict:
    return {
        "id": r.id,
        "dataset_id": r.dataset_id,
        "dataset_name": r.dataset_name,
        "target_column": r.target_column,
        "task_type": r.task_type,
        "model_name": r.model_name,
        "model_config": r.model_config,
        "preprocessing_config": r.preprocessing_config,
        "train_size": r.train_size,
        "test_size": r.test_size,
        "random_seed": r.random_seed,
        "n_features": r.n_features,
        "n_train_rows": r.n_train_rows,
        "n_test_rows": r.n_test_rows,
        "metrics": r.metrics,
        "train_duration_s": r.train_duration_s,
        "inference_duration_s": r.inference_duration_s,
        "status": r.status,
        "error_message": r.error_message,
        "training_history": r.training_history,
        "forecast_data": r.forecast_data,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "updated_at": r.updated_at.isoformat() if r.updated_at else None,
    }
