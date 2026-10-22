"""Kaggle dataset search and import routes."""
from contextlib import contextmanager
import importlib.util
import json
import logging
import os
import shutil
import tempfile
import threading
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.datasets import _load_csv_safe, _safe_filename
from app.core.config import (
    KAGGLE_CREDENTIALS_PATH,
    MAX_KAGGLE_DOWNLOAD_MB,
    UPLOADS_DIR,
)
from app.core.database import DatasetRecord, get_db

router = APIRouter(prefix="/api/kaggle", tags=["kaggle"])
logger = logging.getLogger(__name__)
_KAGGLE_ENV_LOCK = threading.Lock()


class KaggleImportRequest(BaseModel):
    ref: str = Field(min_length=3, max_length=200)
    csv_filename: Optional[str] = None


class KaggleCredentials(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    key: str = Field(min_length=1, max_length=256)


def get_kaggle_credentials():
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if username and key:
        return username, key
    try:
        saved = json.loads(KAGGLE_CREDENTIALS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return username, key
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not read saved Kaggle credentials")
        raise HTTPException(500, "Saved Kaggle credentials could not be read.")
    return username or saved.get("username"), key or saved.get("key")


def mask_kaggle_key(key: str) -> str:
    return f"{key[:2]}{'*' * max(4, min(len(key) - 2, 12))}" if len(key) > 2 else "****"


def _require_credentials(credentials=None):
    username, key = credentials or get_kaggle_credentials()
    if not username or not key:
        raise HTTPException(
            400,
            "Kaggle credentials are missing. Save a username and API key in Settings or set KAGGLE_USERNAME and KAGGLE_KEY.",
        )
    if importlib.util.find_spec("kaggle") is None:
        raise HTTPException(503, "Kaggle support is not installed. Install it with: pip install kaggle")
    return username, key


@contextmanager
def _kaggle_api(credentials=None):
    username, key = _require_credentials(credentials)
    with _KAGGLE_ENV_LOCK:
        old_username = os.environ.get("KAGGLE_USERNAME")
        old_key = os.environ.get("KAGGLE_KEY")
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            yield api
        finally:
            if old_username is None:
                os.environ.pop("KAGGLE_USERNAME", None)
            else:
                os.environ["KAGGLE_USERNAME"] = old_username
            if old_key is None:
                os.environ.pop("KAGGLE_KEY", None)
            else:
                os.environ["KAGGLE_KEY"] = old_key


def _raise_kaggle_error(exc, ref=None):
    status_code = _kaggle_status_code(exc)
    message = str(exc).lower()
    if status_code == 403 or "403" in message:
        raise HTTPException(403, "Kaggle denied access (403). Accept the dataset rules on Kaggle, then retry.")
    if status_code == 404 or "404" in message or "not found" in message.lower():
        raise HTTPException(404, f"Kaggle dataset '{ref or ''}' was not found.")
    _log_kaggle_exception("Kaggle API request failed", exc)
    raise HTTPException(502, "Kaggle request failed. Check credentials and network access.")


def _kaggle_status_code(exc):
    status_code = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    try:
        return int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        return None


def _log_kaggle_exception(message, exc):
    # Avoid traceback and exception text because either can contain request credentials.
    logger.exception(
        message,
        exc_info=(RuntimeError, RuntimeError("Kaggle request failed"), None),
    )


def kaggle_connection_error(exc):
    status_code = _kaggle_status_code(exc)
    error_text = str(exc).lower()
    error_type = type(exc).__name__.lower()
    if status_code == 401 or any(token in error_text for token in ("401", "unauthorized", "invalid credentials")):
        return 401, "Kaggle credentials are invalid. Check the username and API key."
    if status_code == 403 or "403" in error_text or "forbidden" in error_text:
        return 403, "Kaggle denied access (403). Check dataset permissions and accepted rules."
    if status_code == 404 or "404" in error_text:
        return 404, "Kaggle connection check could not find the requested resource."
    if any(token in error_type for token in ("connection", "timeout")) or any(
        token in error_text for token in ("connection error", "timed out", "network is unreachable", "name resolution")
    ):
        return 502, "Could not reach Kaggle due to a network error. Check your internet connection and retry."
    return 502, "Kaggle connection test failed. Check credentials, network access, and Kaggle availability."


def _dataset_search_result(item):
    total_bytes = getattr(item, "totalBytes", None) or getattr(item, "total_bytes", None)
    return {
        "title": getattr(item, "title", None) or getattr(item, "ref", ""),
        "ref": getattr(item, "ref", ""),
        "size": total_bytes if total_bytes is not None else getattr(item, "size", None),
        "download_count": getattr(item, "downloadCount", None) or getattr(item, "download_count", 0),
    }


@router.get("/search")
def search_kaggle_datasets(q: str = Query(min_length=1, max_length=200)):
    try:
        with _kaggle_api() as api:
            results = api.dataset_list(search=q, page=1)
        return [_dataset_search_result(result) for result in results]
    except HTTPException:
        raise
    except Exception as exc:
        _raise_kaggle_error(exc)


@router.post("/import")
def import_kaggle_dataset(body: KaggleImportRequest, db: Session = Depends(get_db)):
    ref = body.ref.strip().strip("/")
    if len(ref.split("/")) != 2 or any(part in ("", ".", "..") for part in ref.split("/")):
        raise HTTPException(400, "Dataset ref must use the owner/name format.")

    max_bytes = MAX_KAGGLE_DOWNLOAD_MB * 1024 * 1024
    with tempfile.TemporaryDirectory(prefix="kaggle-import-") as temp:
        temp_path = Path(temp)
        try:
            with _kaggle_api() as api:
                detail = api.dataset_view(ref)
                total_bytes = getattr(detail, "totalBytes", None) or getattr(detail, "total_bytes", None)
                if total_bytes is not None and int(total_bytes) > max_bytes:
                    raise HTTPException(413, f"Kaggle dataset exceeds the configured {MAX_KAGGLE_DOWNLOAD_MB} MB download limit.")
                api.dataset_download_files(ref, path=str(temp_path), force=True, quiet=True, unzip=False)
        except HTTPException:
            raise
        except Exception as exc:
            _raise_kaggle_error(exc, ref)

        zip_files = list(temp_path.glob("*.zip"))
        if not zip_files:
            raise HTTPException(404, "Kaggle dataset download did not contain a ZIP archive.")
        archive_path = zip_files[0]
        if archive_path.stat().st_size > max_bytes:
            raise HTTPException(413, f"Kaggle download exceeds the configured {MAX_KAGGLE_DOWNLOAD_MB} MB limit.")

        try:
            with zipfile.ZipFile(archive_path) as archive:
                csv_members = []
                for member in archive.infolist():
                    path = PurePosixPath(member.filename)
                    if member.is_dir() or path.is_absolute() or ".." in path.parts:
                        continue
                    if path.suffix.lower() == ".csv":
                        if member.file_size > max_bytes:
                            raise HTTPException(413, f"CSV file exceeds the configured {MAX_KAGGLE_DOWNLOAD_MB} MB limit.")
                        csv_members.append(member)
                if not csv_members:
                    raise HTTPException(400, "This Kaggle dataset contains no CSV files.")
                csv_names = [member.filename for member in csv_members]
                if body.csv_filename is None and len(csv_names) > 1:
                    return {
                        "selection_required": True,
                        "ref": ref,
                        "csv_files": csv_names,
                    }
                selected_name = body.csv_filename or csv_names[0]
                selected = next((member for member in csv_members if member.filename == selected_name), None)
                if selected is None:
                    raise HTTPException(400, "Choose one of the CSV files listed for this Kaggle dataset.")
                output_filename = _safe_filename(Path(selected.filename).name)
                dest_path = UPLOADS_DIR / f"{uuid.uuid4()}_{output_filename}"
                with archive.open(selected) as source, dest_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
        except HTTPException:
            raise
        except (OSError, zipfile.BadZipFile) as exc:
            raise HTTPException(400, "Could not read the Kaggle dataset archive.") from exc

    try:
        frame = _load_csv_safe(str(dest_path))
        if frame.empty:
            raise ValueError("CSV contains no data rows.")
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not parse selected Kaggle CSV: {exc}") from exc

    dataset_id = dest_path.name.split("_", 1)[0]
    source_title = getattr(detail, "title", None) or ref
    record = DatasetRecord(
        id=dataset_id,
        name=Path(source_title).stem,
        filename=dest_path.name,
        filepath=str(dest_path),
        size_bytes=dest_path.stat().st_size,
        row_count=len(frame),
        col_count=len(frame.columns),
        columns=frame.columns.tolist(),
        dtypes={column: str(frame[column].dtype) for column in frame.columns},
        status="ready",
        notes=f"Imported from Kaggle dataset {ref}.",
    )
    db.add(record)
    db.commit()
    return {
        "selection_required": False,
        "dataset": {
            "id": dataset_id,
            "name": record.name,
            "filename": record.filename,
            "size_bytes": record.size_bytes,
            "row_count": record.row_count,
            "col_count": record.col_count,
            "columns": record.columns,
            "status": record.status,
            "is_synthetic": False,
        },
    }
