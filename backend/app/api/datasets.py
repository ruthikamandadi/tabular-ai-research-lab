"""
Dataset upload, management, profiling, and preview endpoints.
"""
import uuid
import os
import io
import chardet
import importlib.util
import logging
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db, SessionLocal, DatasetRecord
from app.core.config import UPLOADS_DIR, SAMPLE_DIR, MAX_UPLOAD_SIZE_MB, MAX_ROWS_PREVIEW, MAX_SYNTHETIC_ROWS
from app.services.profiler import profile_dataset, suggest_task_type, get_target_distribution

router = APIRouter(prefix="/api/datasets", tags=["datasets"])
logger = logging.getLogger(__name__)
_SYNTHESIS_JOBS: Dict[str, Dict[str, Any]] = {}
_SYNTHESIS_LOCK = threading.Lock()
_SYNTHESIS_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="synthetic-data")


class SynthesisRequest(BaseModel):
    method: str
    num_rows: int = Field(gt=0, le=MAX_SYNTHETIC_ROWS)
    epochs: int = Field(default=100, ge=1, le=1000)
    target_column: Optional[str] = None


def _safe_filename(filename: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)


def _load_csv_safe(filepath: str) -> pd.DataFrame:
    """Load CSV with encoding detection."""
    with open(filepath, "rb") as f:
        raw = f.read(100_000)
    detected = chardet.detect(raw)
    encoding = detected.get("encoding", "utf-8") or "utf-8"

    for enc in [encoding, "utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(filepath, encoding=enc, low_memory=False, on_bad_lines="warn")
            return df
        except Exception:
            continue
    raise ValueError("Could not read CSV with supported encodings.")


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Validate extension
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported.")

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_SIZE_MB}MB limit.")

    dataset_id = str(uuid.uuid4())
    safe_name = _safe_filename(Path(file.filename).stem) + ".csv"
    dest_path = UPLOADS_DIR / f"{dataset_id}_{safe_name}"

    with open(dest_path, "wb") as f:
        f.write(content)

    try:
        df = _load_csv_safe(str(dest_path))
    except Exception as e:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Failed to parse CSV: {e}")

    if df.empty:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(400, "CSV file is empty.")

    record = DatasetRecord(
        id=dataset_id,
        name=Path(file.filename).stem,
        filename=file.filename,
        filepath=str(dest_path),
        size_bytes=len(content),
        row_count=len(df),
        col_count=len(df.columns),
        columns=df.columns.tolist(),
        dtypes={col: str(df[col].dtype) for col in df.columns},
        status="ready",
    )
    db.add(record)
    db.commit()

    return {
        "id": dataset_id,
        "name": record.name,
        "filename": file.filename,
        "size_bytes": len(content),
        "row_count": len(df),
        "col_count": len(df.columns),
        "columns": df.columns.tolist(),
        "status": "ready",
        "is_synthetic": False,
        "created_at": record.created_at.isoformat(),
    }


@router.get("")
def list_datasets(db: Session = Depends(get_db)):
    records = db.query(DatasetRecord).order_by(DatasetRecord.created_at.desc()).all()
    return [
        {
            "id": r.id,
            "name": r.name,
            "filename": r.filename,
            "size_bytes": r.size_bytes,
            "row_count": r.row_count,
            "col_count": r.col_count,
            "columns": r.columns,
            "status": r.status,
            "is_synthetic": bool(r.is_synthetic),
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in records
    ]


def _optional_method_availability():
    checks = {
        "smote": ("imblearn", "imbalanced-learn"),
        "gaussian_copula": ("sdv", "sdv"),
        "ctgan": ("sdv", "sdv"),
    }
    methods = [{"method": "bootstrap", "available": True, "install": None}]
    for method, (module_name, package_name) in checks.items():
        try:
            available = importlib.util.find_spec(module_name) is not None
        except Exception:
            logger.exception("Could not check optional synthetic method %s", method)
            available = False
        methods.append({
            "method": method,
            "available": available,
            "install": None if available else package_name,
        })
    return methods


@router.get("/synthesis/methods")
def get_synthesis_methods():
    return {
        "methods": _optional_method_availability(),
        "max_rows": MAX_SYNTHETIC_ROWS,
    }


def _numeric_noise(frame: pd.DataFrame, target_column: Optional[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include="number").columns:
        if column == target_column:
            continue
        values = result[column]
        scale = float(values.std()) if len(values) > 1 else 0.0
        if scale > 0:
            result[column] = values + np.random.normal(0, scale * 0.01, len(values))
            if pd.api.types.is_integer_dtype(values.dtype):
                result[column] = result[column].round().astype(values.dtype)
    return result


def _make_bootstrap(real: pd.DataFrame, num_rows: int, target_column: Optional[str]) -> pd.DataFrame:
    return _numeric_noise(real.sample(n=num_rows, replace=True, random_state=42).reset_index(drop=True), target_column)


def _make_smote(real: pd.DataFrame, num_rows: int, target_column: Optional[str]) -> pd.DataFrame:
    if not target_column or target_column not in real.columns:
        raise ValueError("Choose a valid target column for SMOTE.")
    if importlib.util.find_spec("imblearn") is None:
        raise ValueError("SMOTE requires imbalanced-learn. Install it with: pip install imbalanced-learn")
    from imblearn.over_sampling import SMOTE, SMOTENC, SMOTEN
    import numpy as np

    feature_columns = [column for column in real.columns if column != target_column]
    if not feature_columns:
        raise ValueError("SMOTE needs at least one feature column in addition to its target.")
    target_values = pd.Categorical(real[target_column])
    y = target_values.codes
    if (y < 0).any():
        raise ValueError("SMOTE does not support missing target values.")
    _, class_counts = np.unique(y, return_counts=True)
    if len(class_counts) < 2:
        raise ValueError("SMOTE requires at least two target classes.")
    if int(class_counts.min()) < 2:
        raise ValueError("SMOTE requires at least two rows in every target class.")

    categorical_columns = [
        column for column in feature_columns
        if pd.api.types.is_object_dtype(real[column])
        or isinstance(real[column].dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(real[column])
    ]
    encoded = pd.DataFrame(index=real.index)
    category_values = {}
    for column in feature_columns:
        series = real[column]
        if column in categorical_columns:
            mode = series.mode(dropna=True)
            filled = series.fillna(mode.iloc[0] if not mode.empty else "missing").astype(str)
            categories = pd.Index(filled.unique())
            category_values[column] = categories
            encoded[column] = pd.Categorical(filled, categories=categories).codes
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            encoded[column] = numeric.fillna(numeric.median() if numeric.notna().any() else 0)
    min_class_count = int(class_counts.min())
    k_neighbors = min(5, min_class_count - 1)
    if len(categorical_columns) == len(feature_columns):
        sampler = SMOTEN(random_state=42, k_neighbors=k_neighbors)
    elif categorical_columns:
        sampler = SMOTENC(
            categorical_features=[feature_columns.index(column) for column in categorical_columns],
            random_state=42,
            k_neighbors=k_neighbors,
        )
    else:
        sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
    sampled_x, sampled_y = sampler.fit_resample(encoded, y)
    sampled = pd.DataFrame(sampled_x, columns=feature_columns)
    for column in feature_columns:
        if column in category_values:
            categories = category_values[column]
            codes = np.rint(pd.to_numeric(sampled[column], errors="coerce").fillna(0)).astype(int)
            codes = np.clip(codes, 0, len(categories) - 1)
            sampled[column] = categories.take(codes).to_numpy()
        elif pd.api.types.is_integer_dtype(real[column].dtype):
            sampled[column] = np.rint(sampled[column]).astype(real[column].dtype)
    sampled[target_column] = target_values.categories.take(np.asarray(sampled_y, dtype=int))
    return sampled.sample(n=num_rows, replace=num_rows > len(sampled), random_state=42).reset_index(drop=True)


def _make_sdv(real: pd.DataFrame, num_rows: int, method: str, epochs: int) -> pd.DataFrame:
    if importlib.util.find_spec("sdv") is None:
        raise ValueError(f"{method} requires SDV. Install it with: pip install sdv")
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer

    training_data = real
    if method == "ctgan" and len(real) > 50_000:
        training_data = real.sample(n=20_000, random_state=42)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=training_data)
    if method == "ctgan":
        synthesizer = CTGANSynthesizer(metadata, epochs=epochs, verbose=False)
    else:
        synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(training_data)
    return synthesizer.sample(num_rows=num_rows)


def _quality_report(real: pd.DataFrame, synthetic: pd.DataFrame):
    from scipy.stats import ks_2samp
    import numpy as np

    numeric_columns = [
        column for column in real.columns
        if pd.api.types.is_numeric_dtype(real[column]) and column in synthetic.columns
    ]
    numeric_report = {}
    for column in numeric_columns:
        real_values = pd.to_numeric(real[column], errors="coerce").dropna()
        synthetic_values = pd.to_numeric(synthetic[column], errors="coerce").dropna()
        statistic = float(ks_2samp(real_values, synthetic_values).statistic) if len(real_values) and len(synthetic_values) else None
        distribution = None
        if len(real_values) and len(synthetic_values):
            edges = np.histogram_bin_edges(
                np.concatenate([real_values.to_numpy(), synthetic_values.to_numpy()]),
                bins=10,
            )
            real_counts, _ = np.histogram(real_values, bins=edges)
            synthetic_counts, _ = np.histogram(synthetic_values, bins=edges)
            distribution = {
                "labels": [f"{edges[index]:.3g}-{edges[index + 1]:.3g}" for index in range(len(edges) - 1)],
                "real": (real_counts / max(1, real_counts.sum())).round(6).tolist(),
                "synthetic": (synthetic_counts / max(1, synthetic_counts.sum())).round(6).tolist(),
            }
        numeric_report[column] = {
            "ks_statistic": round(statistic, 6) if statistic is not None else None,
            "similarity": round(1 - statistic, 6) if statistic is not None else None,
            "distribution": distribution,
        }

    categorical_columns = [column for column in real.columns if column not in numeric_columns]
    categorical_report = {}
    for column in categorical_columns:
        real_freq = real[column].fillna("<missing>").astype(str).value_counts(normalize=True)
        synthetic_freq = synthetic[column].fillna("<missing>").astype(str).value_counts(normalize=True)
        labels = real_freq.index.union(synthetic_freq.index)
        distance = float((real_freq.reindex(labels, fill_value=0) - synthetic_freq.reindex(labels, fill_value=0)).abs().sum() / 2)
        top_labels = (real_freq.add(synthetic_freq, fill_value=0).sort_values(ascending=False).head(10).index)
        categorical_report[column] = {
            "total_variation_distance": round(distance, 6),
            "similarity": round(1 - distance, 6),
            "distribution": {
                "labels": top_labels.tolist(),
                "real": real_freq.reindex(top_labels, fill_value=0).round(6).tolist(),
                "synthetic": synthetic_freq.reindex(top_labels, fill_value=0).round(6).tolist(),
            },
        }

    if len(numeric_columns) > 1:
        real_corr = real[numeric_columns].corr().fillna(0)
        synthetic_corr = synthetic[numeric_columns].corr().reindex(index=numeric_columns, columns=numeric_columns).fillna(0)
        upper = np.triu_indices(len(numeric_columns), k=1)
        correlation_difference = float(np.abs(real_corr.to_numpy() - synthetic_corr.to_numpy())[upper].mean())
    else:
        correlation_difference = 0.0
    return {
        "numeric_columns": numeric_report,
        "categorical_columns": categorical_report,
        "mean_absolute_correlation_difference": round(correlation_difference, 6),
    }


def _run_synthesis(job_id: str, dataset_path: str, original_name: str, request: SynthesisRequest):
    db = None
    output_path = None
    try:
        with _SYNTHESIS_LOCK:
            _SYNTHESIS_JOBS[job_id].update(status="running", progress=10, message="Loading source dataset...")
        real = _load_csv_safe(dataset_path)
        if request.method == "bootstrap":
            synthetic = _make_bootstrap(real, request.num_rows, request.target_column)
        elif request.method == "smote":
            synthetic = _make_smote(real, request.num_rows, request.target_column)
        else:
            synthetic = _make_sdv(real, request.num_rows, request.method, request.epochs)
        with _SYNTHESIS_LOCK:
            _SYNTHESIS_JOBS[job_id].update(progress=75, message="Calculating quality report...")
        quality_source = real.sample(n=min(5000, len(real)), random_state=42)
        quality_synthetic = synthetic.sample(n=min(5000, len(synthetic)), random_state=42)
        quality = _quality_report(quality_source, quality_synthetic)

        dataset_id = str(uuid.uuid4())
        filename = f"{dataset_id}_{_safe_filename(original_name)}_synthetic.csv"
        output_path = UPLOADS_DIR / filename
        synthetic.to_csv(output_path, index=False)
        db = SessionLocal()
        record = DatasetRecord(
            id=dataset_id,
            name=f"{original_name}_synthetic",
            filename=filename,
            filepath=str(output_path),
            size_bytes=output_path.stat().st_size,
            row_count=len(synthetic),
            col_count=len(synthetic.columns),
            columns=synthetic.columns.tolist(),
            dtypes={column: str(synthetic[column].dtype) for column in synthetic.columns},
            status="ready",
            is_synthetic=True,
            notes=f"Synthetic dataset generated with {request.method}.",
        )
        db.add(record)
        db.commit()
        dataset = {
            "id": record.id,
            "name": record.name,
            "filename": record.filename,
            "size_bytes": record.size_bytes,
            "row_count": record.row_count,
            "col_count": record.col_count,
            "columns": record.columns,
            "status": record.status,
            "is_synthetic": True,
        }
        with _SYNTHESIS_LOCK:
            _SYNTHESIS_JOBS[job_id].update(
                status="completed",
                progress=100,
                message="Synthetic dataset created.",
                dataset=dataset,
                quality_report=quality,
            )
    except Exception as exc:
        logger.exception("Synthetic generation job %s failed", job_id)
        if output_path is not None:
            output_path.unlink(missing_ok=True)
        with _SYNTHESIS_LOCK:
            _SYNTHESIS_JOBS[job_id].update(status="failed", progress=0, message=str(exc), error=str(exc))
    finally:
        if db is not None:
            db.close()


@router.post("/{dataset_id}/synthesize")
def synthesize_dataset(dataset_id: str, body: SynthesisRequest, db: Session = Depends(get_db)):
    allowed_methods = {"bootstrap", "smote", "gaussian_copula", "ctgan"}
    if body.method not in allowed_methods:
        raise HTTPException(400, f"method must be one of: {', '.join(sorted(allowed_methods))}.")
    if body.method in {"gaussian_copula", "ctgan"} and importlib.util.find_spec("sdv") is None:
        raise HTTPException(400, f"{body.method} requires SDV. Install it with: pip install sdv")
    if body.method == "smote" and importlib.util.find_spec("imblearn") is None:
        raise HTTPException(400, "SMOTE requires imbalanced-learn. Install it with: pip install imbalanced-learn")
    dataset = db.query(DatasetRecord).filter(DatasetRecord.id == dataset_id).first()
    if not dataset:
        raise HTTPException(404, "Dataset not found.")
    if body.method == "smote" and not body.target_column:
        raise HTTPException(400, "Choose a target column for SMOTE.")

    job_id = str(uuid.uuid4())
    with _SYNTHESIS_LOCK:
        _SYNTHESIS_JOBS[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "progress": 0,
            "message": "Queued...",
        }
    _SYNTHESIS_EXECUTOR.submit(_run_synthesis, job_id, dataset.filepath, dataset.name, body)
    return {"job_id": job_id, "status": "pending"}


@router.get("/synthesis/{job_id}/status")
def get_synthesis_status(job_id: str):
    with _SYNTHESIS_LOCK:
        job = _SYNTHESIS_JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Synthesis job not found.")
        return dict(job)

@router.get("/{dataset_id}/preview")
def preview_dataset(dataset_id: str, rows: int = Query(100, le=500), db: Session = Depends(get_db)):
    record = db.query(DatasetRecord).filter(DatasetRecord.id == dataset_id).first()
    if not record:
        raise HTTPException(404, "Dataset not found.")

    df = _load_csv_safe(record.filepath)
    preview_df = df.head(rows)
    preview_df = preview_df.where(pd.notnull(preview_df), None)

    return {
        "columns": df.columns.tolist(),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "rows": preview_df.to_dict(orient="records"),
        "total_rows": len(df),
        "preview_rows": len(preview_df),
    }


@router.get("/{dataset_id}/profile")
def profile_dataset_endpoint(dataset_id: str, db: Session = Depends(get_db)):
    record = db.query(DatasetRecord).filter(DatasetRecord.id == dataset_id).first()
    if not record:
        raise HTTPException(404, "Dataset not found.")

    df = _load_csv_safe(record.filepath)
    profile = profile_dataset(df)
    profile["dataset_id"] = dataset_id
    profile["dataset_name"] = record.name
    return profile


@router.post("/{dataset_id}/suggest-task")
def suggest_task(dataset_id: str, body: dict, db: Session = Depends(get_db)):
    target_col = body.get("target_column")
    if not target_col:
        raise HTTPException(400, "target_column is required.")

    record = db.query(DatasetRecord).filter(DatasetRecord.id == dataset_id).first()
    if not record:
        raise HTTPException(404, "Dataset not found.")

    df = _load_csv_safe(record.filepath)
    if target_col not in df.columns:
        raise HTTPException(400, f"Column '{target_col}' not found in dataset.")

    _validate_target(df, target_col)
    result = suggest_task_type(df, target_col)
    return result


def _validate_target(df: pd.DataFrame, target_column: str, task_type: Optional[str] = None):
    if target_column not in df.columns:
        raise HTTPException(400, f"Column '{target_column}' not found in dataset.")
    if len(df) < 20:
        raise HTTPException(400, "Dataset must contain at least 20 rows.")

    target = df[target_column]
    unique_count = target.nunique(dropna=True)
    unique_ratio = unique_count / max(len(target), 1)
    if target_column.lower().endswith("_id") or target_column.lower() == "id" or unique_ratio > 0.95:
        raise HTTPException(400, f"Target column '{target_column}' appears to be an identifier.")

    if task_type in ("binary_classification", "multiclass_classification"):
        if not 2 <= unique_count <= 50:
            raise HTTPException(400, "Classification target must have between 2 and 50 unique values.")
    elif task_type in ("regression", "time_series") and not pd.api.types.is_numeric_dtype(target):
        label = "Time-series" if task_type == "time_series" else "Regression"
        raise HTTPException(400, f"{label} target must contain numeric values.")


@router.get("/{dataset_id}/target-distribution")
def target_distribution(dataset_id: str, target_column: str, task_type: str, db: Session = Depends(get_db)):
    record = db.query(DatasetRecord).filter(DatasetRecord.id == dataset_id).first()
    if not record:
        raise HTTPException(404, "Dataset not found.")
    df = _load_csv_safe(record.filepath)
    _validate_target(df, target_column, task_type)

    dist = get_target_distribution(df, target_column, task_type)
    return dist


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    record = db.query(DatasetRecord).filter(DatasetRecord.id == dataset_id).first()
    if not record:
        raise HTTPException(404, "Dataset not found.")

    try:
        Path(record.filepath).unlink(missing_ok=True)
    except Exception:
        pass

    db.delete(record)
    db.commit()
    return {"message": "Dataset deleted."}


@router.get("/sample/list")
def list_sample_datasets():
    """List available sample datasets."""
    samples = []
    for f in SAMPLE_DIR.glob("*.csv"):
        size = f.stat().st_size
        try:
            df = pd.read_csv(f, nrows=5)
            samples.append({
                "filename": f.name,
                "name": f.stem.replace("_", " ").title(),
                "size_bytes": size,
                "columns": df.columns.tolist(),
            })
        except Exception:
            pass
    return samples


@router.post("/sample/load/{filename}")
async def load_sample_dataset(filename: str, db: Session = Depends(get_db)):
    """Load a sample dataset into the system."""
    safe_fn = _safe_filename(filename)
    sample_path = SAMPLE_DIR / safe_fn

    if not sample_path.exists():
        raise HTTPException(404, f"Sample file '{filename}' not found.")

    content = sample_path.read_bytes()

    dataset_id = str(uuid.uuid4())
    dest_path = UPLOADS_DIR / f"{dataset_id}_{safe_fn}"
    dest_path.write_bytes(content)

    try:
        df = _load_csv_safe(str(dest_path))
    except Exception as e:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Failed to parse sample CSV: {e}")

    record = DatasetRecord(
        id=dataset_id,
        name=Path(filename).stem.replace("_", " ").title(),
        filename=filename,
        filepath=str(dest_path),
        size_bytes=len(content),
        row_count=len(df),
        col_count=len(df.columns),
        columns=df.columns.tolist(),
        dtypes={col: str(df[col].dtype) for col in df.columns},
        status="ready",
        notes="Sample dataset",
    )
    db.add(record)
    db.commit()

    return {
        "id": dataset_id,
        "name": record.name,
        "filename": filename,
        "size_bytes": len(content),
        "row_count": len(df),
        "col_count": len(df.columns),
        "columns": df.columns.tolist(),
        "status": "ready",
    }
