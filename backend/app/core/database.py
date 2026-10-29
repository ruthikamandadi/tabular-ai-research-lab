"""
Database setup using SQLAlchemy with support for SQLite and PostgreSQL (Neon).
"""
import os
import uuid
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional, Dict, Any

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Float,
    Text,
    DateTime,
    JSON,
    Boolean,
    LargeBinary,
    inspect,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from app.core.config import (
    DB_PATH,
    DATABASE_URL_ENV,
    MAX_USER_STORAGE_MB,
)

logger = logging.getLogger(__name__)

# Configure engine based on DATABASE_URL
if DATABASE_URL_ENV:
    raw_url = DATABASE_URL_ENV
    # Normalize postgres:// and postgresql:// to postgresql+psycopg://
    if raw_url.startswith("postgres://"):
        raw_url = "postgresql+psycopg://" + raw_url[len("postgres://"):]
    elif raw_url.startswith("postgresql://") and not raw_url.startswith("postgresql+"):
        raw_url = "postgresql+psycopg://" + raw_url[len("postgresql://"):]

    # Ensure sslmode=require for Neon if not explicitly given
    if "sslmode=" not in raw_url and "neon.tech" in raw_url:
        separator = "&" if "?" in raw_url else "?"
        raw_url = f"{raw_url}{separator}sslmode=require"

    DATABASE_URL = raw_url
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=300,
    )
    IS_POSTGRES = True
else:
    DATABASE_URL = f"sqlite:///{DB_PATH}"
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
    )
    IS_POSTGRES = False

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class DatasetRecord(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id = Column(String, nullable=True, index=True)
    name = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    file_content = Column(LargeBinary, nullable=True)  # Raw CSV bytes for cloud/PostgreSQL persistence
    size_bytes = Column(Integer)
    row_count = Column(Integer)
    col_count = Column(Integer)
    columns = Column(JSON)   # list of column names
    dtypes = Column(JSON)    # dict col -> dtype string
    status = Column(String, default="ready")  # ready | error
    is_synthetic = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    notes = Column(Text, default="")


class ExperimentRecord(Base):
    __tablename__ = "experiments"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id = Column(String, nullable=True, index=True)
    dataset_id = Column(String, nullable=False)
    dataset_name = Column(String)
    target_column = Column(String, nullable=False)
    task_type = Column(String, nullable=False)  # binary_classification | multiclass_classification | regression | time_series
    model_name = Column(String, nullable=False)
    model_config = Column(JSON)
    preprocessing_config = Column(JSON)
    train_size = Column(Float)
    test_size = Column(Float)
    random_seed = Column(Integer)
    n_features = Column(Integer)
    n_train_rows = Column(Integer)
    n_test_rows = Column(Integer)
    metrics = Column(JSON)
    train_duration_s = Column(Float)
    inference_duration_s = Column(Float)
    status = Column(String, default="pending")  # pending | running | completed | failed | failed: server restarted
    error_message = Column(Text)
    artifact_path = Column(String)
    artifact_data = Column(LargeBinary, nullable=True)  # Zipped model + preprocessor + meta for persistence
    report_html = Column(Text, nullable=True)
    report_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    random_seed_used = Column(Integer)
    training_history = Column(JSON)   # for MLP loss curves
    best_params = Column(JSON)
    best_cv_score = Column(Float)
    tuning_trials = Column(JSON)
    forecast_data = Column(JSON)


class UserRecord(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    email = Column(String, nullable=False, unique=True, index=True)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)


class AuthSessionRecord(Base):
    __tablename__ = "auth_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    token_hash = Column(String, nullable=False, unique=True, index=True)
    expires_at = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)


def init_db():
    Base.metadata.create_all(bind=engine)
    try:
        inspector = inspect(engine)
        dialect_name = engine.dialect.name

        # Safe automatic migration for datasets table
        if inspector.has_table("datasets"):
            existing_dataset_cols = {col["name"] for col in inspector.get_columns("datasets")}
            with engine.begin() as conn:
                if "file_content" not in existing_dataset_cols:
                    col_type = "BYTEA" if dialect_name == "postgresql" else "BLOB"
                    conn.execute(text(f"ALTER TABLE datasets ADD COLUMN file_content {col_type}"))
                if "is_synthetic" not in existing_dataset_cols:
                    col_type = "BOOLEAN NOT NULL DEFAULT FALSE" if dialect_name == "postgresql" else "BOOLEAN NOT NULL DEFAULT 0"
                    conn.execute(text(f"ALTER TABLE datasets ADD COLUMN is_synthetic {col_type}"))
                if "owner_id" not in existing_dataset_cols:
                    conn.execute(text("ALTER TABLE datasets ADD COLUMN owner_id VARCHAR"))

        # Safe automatic migration for experiments table
        if inspector.has_table("experiments"):
            existing_exp_cols = {col["name"] for col in inspector.get_columns("experiments")}
            with engine.begin() as conn:
                if "artifact_data" not in existing_exp_cols:
                    col_type = "BYTEA" if dialect_name == "postgresql" else "BLOB"
                    conn.execute(text(f"ALTER TABLE experiments ADD COLUMN artifact_data {col_type}"))
                if "report_html" not in existing_exp_cols:
                    conn.execute(text("ALTER TABLE experiments ADD COLUMN report_html TEXT"))
                if "report_json" not in existing_exp_cols:
                    conn.execute(text("ALTER TABLE experiments ADD COLUMN report_json JSON"))
                for name, sql_type in (
                    ("owner_id", "VARCHAR"),
                    ("best_params", "JSON"),
                    ("best_cv_score", "FLOAT"),
                    ("tuning_trials", "JSON"),
                    ("forecast_data", "JSON"),
                ):
                    if name not in existing_exp_cols:
                        conn.execute(text(f"ALTER TABLE experiments ADD COLUMN {name} {sql_type}"))
    except Exception as exc:
        logger.warning("Database schema migration check warning: %s", exc)


def get_user_storage_bytes(owner_id: str, db: Session) -> int:
    if not owner_id:
        return 0
    datasets = db.query(DatasetRecord).filter(DatasetRecord.owner_id == owner_id).all()
    dataset_bytes = sum(
        (len(d.file_content) if d.file_content is not None else (d.size_bytes or 0))
        for d in datasets
    )
    experiments = db.query(ExperimentRecord).filter(ExperimentRecord.owner_id == owner_id).all()
    exp_bytes = sum(
        (len(e.artifact_data) if e.artifact_data is not None else 0)
        for e in experiments
    )
    return dataset_bytes + exp_bytes


def enforce_user_quota(owner_id: str, db: Session, incoming_bytes: int = 0, keep_dataset_id: Optional[str] = None):
    """Keep at most MAX_USER_STORAGE_MB (default 300MB) per user, deleting oldest records if exceeded."""
    if not owner_id:
        return
    max_bytes = MAX_USER_STORAGE_MB * 1024 * 1024
    current_bytes = get_user_storage_bytes(owner_id, db) + incoming_bytes
    if current_bytes <= max_bytes:
        return

    # Delete oldest experiments first
    experiments = (
        db.query(ExperimentRecord)
        .filter(ExperimentRecord.owner_id == owner_id)
        .order_by(ExperimentRecord.created_at.asc())
        .all()
    )
    for exp in experiments:
        if current_bytes <= max_bytes:
            break
        exp_size = len(exp.artifact_data) if exp.artifact_data else 0
        if exp.artifact_path:
            import shutil
            shutil.rmtree(exp.artifact_path, ignore_errors=True)
        db.delete(exp)
        db.commit()
        current_bytes -= exp_size

    # Delete oldest datasets if still exceeding quota
    if current_bytes > max_bytes:
        datasets = (
            db.query(DatasetRecord)
            .filter(DatasetRecord.owner_id == owner_id)
            .order_by(DatasetRecord.created_at.asc())
            .all()
        )
        for ds in datasets:
            if keep_dataset_id and ds.id == keep_dataset_id:
                continue
            if current_bytes <= max_bytes:
                break
            ds_size = len(ds.file_content) if ds.file_content else (ds.size_bytes or 0)
            if ds.filepath:
                try:
                    Path(ds.filepath).unlink(missing_ok=True)
                except Exception:
                    pass
            db.delete(ds)
            db.commit()
            current_bytes -= ds_size


def get_user_storage_summary(owner_id: str, db: Session) -> Dict[str, Any]:
    used_bytes = get_user_storage_bytes(owner_id, db)
    limit_bytes = MAX_USER_STORAGE_MB * 1024 * 1024
    ds_count = db.query(DatasetRecord).filter(DatasetRecord.owner_id == owner_id).count() if owner_id else 0
    exp_count = db.query(ExperimentRecord).filter(ExperimentRecord.owner_id == owner_id).count() if owner_id else 0

    return {
        "used_bytes": used_bytes,
        "used_mb": round(used_bytes / (1024 * 1024), 2),
        "limit_bytes": limit_bytes,
        "limit_mb": MAX_USER_STORAGE_MB,
        "percent_used": round((used_bytes / limit_bytes) * 100, 1) if limit_bytes else 0,
        "datasets_count": ds_count,
        "experiments_count": exp_count,
        "backend": "PostgreSQL (Neon)" if IS_POSTGRES else "SQLite (Local)",
    }
