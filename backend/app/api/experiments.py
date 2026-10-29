"""
Experiment creation, tracking, results, and explainability endpoints.
"""
import uuid
import json
import joblib
import numpy as np
import pandas as pd
import time
import math
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db, ExperimentRecord, DatasetRecord
from app.core.auth import request_user_id, require_authenticated_user
from app.core.datetime_utils import parse_datetime

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/experiments",
    tags=["experiments"],
    dependencies=[Depends(require_authenticated_user)],
)


def _sanitize_for_json(value):
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(x) for x in value]
    if isinstance(value, np.generic):
        return _sanitize_for_json(value.item())
    if isinstance(value, np.ndarray):
        return [_sanitize_for_json(x) for x in value.tolist()]
    if hasattr(value, "__class__") and "device" in value.__class__.__name__.lower():
        return str(value)
    return value


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
    from app.models.model_factory import get_available_models

    return get_available_models()


@router.post("")
def create_experiment(body: ExperimentCreate, request: Request, db: Session = Depends(get_db)):
    # Validate dataset
    owner_id = request_user_id(request)
    dataset = db.query(DatasetRecord).filter(
        DatasetRecord.id == body.dataset_id,
        DatasetRecord.owner_id == owner_id,
    ).first()
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
    from app.models.model_factory import get_available_models

    available = get_available_models()
    model_info = next((m for m in available if m["id"] == body.model_id), None)
    if not model_info:
        raise HTTPException(400, f"Unknown model: {body.model_id}")
    if model_info.get("disabled"):
        raise HTTPException(400, model_info.get("disabled_reason") or f"Model '{body.model_id}' is disabled.")
    if body.task_type not in model_info["tasks"]:
        raise HTTPException(400, f"Model '{body.model_id}' does not support task '{body.task_type}'.")

    experiment_id = str(uuid.uuid4())

    exp = ExperimentRecord(
        id=experiment_id,
        owner_id=owner_id,
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
        "dataset_id": body.dataset_id,
        "target_column": body.target_column,
        "task_type": body.task_type,
        "model_id": body.model_id,
        "model_config": body.model_config,
        "preprocessing_config": body.preprocessing_config,
        "random_seed": body.random_seed,
        "experiment_id": experiment_id,
    }
    from app.experiments.runner import run_training_job

    run_training_job(experiment_id, job_config)

    return {"experiment_id": experiment_id, "status": "pending", "message": "Training started."}


@router.get("")
def list_experiments(request: Request, db: Session = Depends(get_db)):
    records = db.query(ExperimentRecord).filter(
        ExperimentRecord.owner_id == request_user_id(request),
    ).order_by(ExperimentRecord.created_at.desc()).all()
    return [_serialize_experiment(r) for r in records]


@router.get("/{experiment_id}")
def get_experiment(experiment_id: str, request: Request, db: Session = Depends(get_db)):
    record = db.query(ExperimentRecord).filter(
        ExperimentRecord.id == experiment_id,
        ExperimentRecord.owner_id == request_user_id(request),
    ).first()
    if not record:
        raise HTTPException(404, "Experiment not found.")
    return _serialize_experiment(record)


@router.get("/{experiment_id}/status")
def get_experiment_status(experiment_id: str, request: Request, db: Session = Depends(get_db)):
    owner_id = request_user_id(request)
    record = db.query(ExperimentRecord).filter(
        ExperimentRecord.id == experiment_id,
        ExperimentRecord.owner_id == owner_id,
    ).first()
    if not record:
        raise HTTPException(404, "Experiment not found.")

    from app.experiments.runner import get_job

    try:
        job = get_job(experiment_id)
        if job:
            status = job.get("status") or record.status or "pending"
            progress = job.get("progress")
            if progress is None:
                progress = 100 if status == "completed" else 0

            started_at = job.get("started_at")
            if isinstance(started_at, (int, float)):
                elapsed_seconds = round(max(0.0, time.time() - started_at), 1)
            elif record.train_duration_s is not None:
                elapsed_seconds = round(record.train_duration_s, 1)
            else:
                elapsed_seconds = None

            message = job.get("message")
            if not message:
                if status == "completed":
                    message = "Training complete!"
                elif status == "failed":
                    message = record.error_message or "Training failed."
                else:
                    message = "Queued..."

            error = job.get("error") or (record.error_message if status == "failed" else None)
            metrics = job.get("metrics") or record.metrics
            device_val = job.get("device")
            if device_val is None:
                device_val = (record.model_config or {}).get("device")
            device_str = str(device_val) if device_val is not None else None

            response_data = {
                "experiment_id": experiment_id,
                "model_name": record.model_name,
                "status": status,
                "progress": progress,
                "elapsed_seconds": elapsed_seconds,
                "message": message,
                "metrics": metrics,
                "error": error,
                "training_history": job.get("training_history") or record.training_history or [],
                "device": device_str,
                "warning": job.get("warning"),
                "best_params": job.get("best_params") or record.best_params,
                "best_cv_score": job.get("best_cv_score") if job.get("best_cv_score") is not None else record.best_cv_score,
                "tuning_trials": job.get("tuning_trials") or record.tuning_trials,
            }
            return _sanitize_for_json(response_data)

        # Fall back to DB
        status = record.status or "pending"
        progress = 100 if status == "completed" else 0
        elapsed_seconds = round(record.train_duration_s, 1) if record.train_duration_s is not None else None
        message = record.error_message if status == "failed" else ("Training complete!" if status == "completed" else "Queued...")
        device_val = (record.model_config or {}).get("device")

        response_data = {
            "experiment_id": experiment_id,
            "model_name": record.model_name,
            "status": status,
            "progress": progress,
            "elapsed_seconds": elapsed_seconds,
            "message": message,
            "metrics": record.metrics,
            "error": record.error_message if status == "failed" else None,
            "training_history": record.training_history or [],
            "device": str(device_val) if device_val is not None else None,
            "warning": None,
            "best_params": record.best_params,
            "best_cv_score": record.best_cv_score,
            "tuning_trials": record.tuning_trials,
        }
        return _sanitize_for_json(response_data)
    except Exception as exc:
        logger.exception("Error preparing experiment status for %s", experiment_id)
        raise HTTPException(500, f"Could not retrieve status for experiment {experiment_id}: {exc}") from exc


@router.get("/{experiment_id}/results")
def get_experiment_results(experiment_id: str, request: Request, db: Session = Depends(get_db)):
    record = db.query(ExperimentRecord).filter(
        ExperimentRecord.id == experiment_id,
        ExperimentRecord.owner_id == request_user_id(request),
    ).first()
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
        "forecast_data": record.forecast_data,
    }


@router.get("/{experiment_id}/explanations")
def get_explanations(experiment_id: str, request: Request, db: Session = Depends(get_db)):
    owner_id = request_user_id(request)
    record = db.query(ExperimentRecord).filter(
        ExperimentRecord.id == experiment_id,
        ExperimentRecord.owner_id == owner_id,
    ).first()
    if not record:
        raise HTTPException(404, "Experiment not found.")
    if record.status != "completed":
        raise HTTPException(400, "Experiment has not completed.")
    if record.error_message == "model not persisted, retrain to predict":
        raise HTTPException(400, "Model not persisted, retrain to predict.")

    from app.core.config import ARTIFACTS_DIR
    import zipfile
    import io

    artifact_dir = Path(record.artifact_path or (ARTIFACTS_DIR / experiment_id))
    meta_path = artifact_dir / "meta.json"
    if (not meta_path.exists() or not (artifact_dir / "preprocessor.pkl").exists()) and record.artifact_data:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(io.BytesIO(record.artifact_data)) as zf:
                zf.extractall(artifact_dir)
        except Exception as exc:
            raise HTTPException(500, f"Failed to restore model artifacts from database: {exc}")

    if not meta_path.exists():
        raise HTTPException(500, "Artifact metadata missing.")

    with open(meta_path) as f:
        meta = json.load(f)

    model_file = "model.pt" if meta.get("model_type") == "pytorch_mlp" else "model.pkl"
    if not (artifact_dir / model_file).exists():
        raise HTTPException(400, "Model not persisted, retrain to predict.")

    # Load model and preprocessor
    try:
        preprocessor = joblib.load(str(artifact_dir / "preprocessor.pkl"))
        from app.models.model_factory import load_model_wrapper
        model = load_model_wrapper(
            str(artifact_dir / model_file),
            meta.get("model_type", "sklearn"),
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to load model artifacts: {e}")

    # Load dataset
    dataset = db.query(DatasetRecord).filter(
        DatasetRecord.id == record.dataset_id,
        DatasetRecord.owner_id == owner_id,
    ).first()
    if not dataset:
        raise HTTPException(404, "Original dataset not found.")

    dataset_file = Path(dataset.filepath)
    if not dataset_file.exists() and dataset.file_content:
        dataset_file.parent.mkdir(parents=True, exist_ok=True)
        dataset_file.write_bytes(dataset.file_content)

    if dataset_file.exists():
        df = pd.read_csv(str(dataset_file), low_memory=False)
    elif dataset.file_content:
        df = pd.read_csv(io.BytesIO(dataset.file_content), low_memory=False)
    else:
        raise HTTPException(404, "Original dataset file not found.")
    input_cols = meta.get("input_feature_cols", [])
    X = df[input_cols].head(1000)  # sample for speed
    X_transformed = preprocessor.transform(X)

    target = df[record.target_column].head(1000)
    if record.task_type not in ("regression", "time_series"):
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
        from app.explainability.importance import compute_model_importances

        importance = compute_model_importances(
            model, X_transformed, X_transformed, y, feature_names, record.task_type
        )
    return importance


@router.delete("/{experiment_id}")
def delete_experiment(experiment_id: str, request: Request, db: Session = Depends(get_db)):
    record = db.query(ExperimentRecord).filter(
        ExperimentRecord.id == experiment_id,
        ExperimentRecord.owner_id == request_user_id(request),
    ).first()
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
