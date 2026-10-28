"""
SQLite database setup using SQLAlchemy.
"""
from sqlalchemy import create_engine, Column, String, Integer, Float, Text, DateTime, JSON, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import UTC, datetime
import uuid

from app.core.config import DB_PATH

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
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
    task_type = Column(String, nullable=False)  # binary_classification | multiclass_classification | regression
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
    status = Column(String, default="pending")  # pending | running | completed | failed
    error_message = Column(Text)
    artifact_path = Column(String)
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
    with engine.begin() as connection:
        dataset_columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(datasets)")}
        dataset_migrations = {
            "is_synthetic": "BOOLEAN NOT NULL DEFAULT 0",
            "owner_id": "VARCHAR",
        }
        for name, sql_type in dataset_migrations.items():
            if name not in dataset_columns:
                connection.exec_driver_sql(f"ALTER TABLE datasets ADD COLUMN {name} {sql_type}")
        columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(experiments)")}
        for name, sql_type in (
            ("owner_id", "VARCHAR"),
            ("best_params", "JSON"),
            ("best_cv_score", "FLOAT"),
            ("tuning_trials", "JSON"),
            ("forecast_data", "JSON"),
        ):
            if name not in columns:
                connection.exec_driver_sql(f"ALTER TABLE experiments ADD COLUMN {name} {sql_type}")
