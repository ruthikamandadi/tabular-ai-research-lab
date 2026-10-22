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


def init_db():
    Base.metadata.create_all(bind=engine)
    with engine.begin() as connection:
        dataset_columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(datasets)")}
        if "is_synthetic" not in dataset_columns:
            connection.exec_driver_sql("ALTER TABLE datasets ADD COLUMN is_synthetic BOOLEAN NOT NULL DEFAULT 0")
        columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(experiments)")}
        for name, sql_type in (
            ("best_params", "JSON"),
            ("best_cv_score", "FLOAT"),
            ("tuning_trials", "JSON"),
            ("forecast_data", "JSON"),
        ):
            if name not in columns:
                connection.exec_driver_sql(f"ALTER TABLE experiments ADD COLUMN {name} {sql_type}")
