"""
Prediction endpoints: batch CSV prediction and single-row prediction.
"""
import uuid
import json
import io
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db, ExperimentRecord, DatasetRecord

router = APIRouter(prefix="/api/models", tags=["predictions"])


def _load_model_and_meta(experiment_id: str, db: Session):
    record = db.query(ExperimentRecord).filter(ExperimentRecord.id == experiment_id).first()
    if not record or record.status != "completed":
        raise HTTPException(400, f"Experiment not completed or not found.")

    artifact_dir = Path(record.artifact_path)
    meta_path = artifact_dir / "meta.json"
    if not meta_path.exists():
        raise HTTPException(500, "Artifact metadata missing.")

    with open(meta_path) as f:
        meta = json.load(f)

    preprocessor = joblib.load(str(artifact_dir / "preprocessor.pkl"))
    label_encoder = None
    le_path = artifact_dir / "label_encoder.pkl"
    if le_path.exists():
        label_encoder = joblib.load(str(le_path))

    from app.models.model_factory import load_model_wrapper
    model = load_model_wrapper(
        str(artifact_dir / ("model.pt" if meta.get("model_type") == "pytorch_mlp" else "model.pkl")),
        meta.get("model_type", "sklearn"),
    )

    return model, preprocessor, label_encoder, meta, record


@router.post("/{experiment_id}/predict-csv")
async def predict_csv(
    experiment_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Batch prediction from CSV upload."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content), low_memory=False)
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    model, preprocessor, label_encoder, meta, record = _load_model_and_meta(experiment_id, db)
    input_cols = meta["input_feature_cols"]

    # Remove target column if present
    target_col = record.target_column
    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    missing = [c for c in input_cols if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Missing columns in input CSV: {missing}")

    X = df[input_cols].copy()
    X_transformed = preprocessor.transform(X)

    y_pred = model.predict(X_transformed)
    y_proba = model.predict_proba(X_transformed)

    # Decode labels
    predictions = []
    for i, pred in enumerate(y_pred):
        row: Dict[str, Any] = {}
        if label_encoder is not None:
            row["prediction"] = str(label_encoder.inverse_transform([int(pred)])[0])
            if y_proba is not None:
                class_names = meta.get("class_names", [])
                for j, cls in enumerate(class_names):
                    if j < y_proba.shape[1]:
                        row[f"prob_{cls}"] = round(float(y_proba[i, j]), 4)
        else:
            row["prediction"] = round(float(pred), 6)
        predictions.append(row)

    result_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(predictions)], axis=1)
    output = io.StringIO()
    result_df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=predictions_{experiment_id[:8]}.csv"},
    )


class SingleRowPredictRequest(BaseModel):
    features: Dict[str, Any]


@router.post("/{experiment_id}/predict-single")
def predict_single(
    experiment_id: str,
    body: SingleRowPredictRequest,
    db: Session = Depends(get_db),
):
    """Single-row prediction from JSON feature dict."""
    model, preprocessor, label_encoder, meta, record = _load_model_and_meta(experiment_id, db)
    input_cols = meta["input_feature_cols"]

    row = {}
    missing_keys = []
    for col in input_cols:
        if col in body.features:
            row[col] = body.features[col]
        else:
            row[col] = None
            missing_keys.append(col)

    df = pd.DataFrame([row])
    X_transformed = preprocessor.transform(df[input_cols])

    y_pred = model.predict(X_transformed)
    y_proba = model.predict_proba(X_transformed)

    pred_val = y_pred[0]
    result: Dict[str, Any] = {}

    if label_encoder is not None:
        result["prediction"] = str(label_encoder.inverse_transform([int(pred_val)])[0])
        result["task_type"] = record.task_type
        if y_proba is not None:
            class_names = meta.get("class_names", [])
            result["class_probabilities"] = {
                cls: round(float(y_proba[0, j]), 4)
                for j, cls in enumerate(class_names)
                if j < y_proba.shape[1]
            }
    else:
        result["prediction"] = round(float(pred_val), 6)
        result["task_type"] = "regression"

    result["warnings"] = [f"Missing values imputed for: {missing_keys}"] if missing_keys else []
    result["input_features"] = body.features
    return result


@router.get("/{experiment_id}/schema")
def get_model_schema(experiment_id: str, db: Session = Depends(get_db)):
    """Return expected input feature schema for a trained model."""
    model, preprocessor, label_encoder, meta, record = _load_model_and_meta(experiment_id, db)

    dataset = db.query(DatasetRecord).filter(DatasetRecord.id == record.dataset_id).first()
    if not dataset:
        raise HTTPException(404, "Original dataset not found.")

    input_cols = meta["input_feature_cols"]
    dtypes = dataset.dtypes or {}

    schema = [
        {"column": col, "dtype": dtypes.get(col, "unknown")}
        for col in input_cols
    ]
    return {
        "experiment_id": experiment_id,
        "target_column": record.target_column,
        "task_type": record.task_type,
        "input_schema": schema,
        "class_names": meta.get("class_names"),
    }
