"""
Application configuration and constants.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = Path(os.getenv("TABULAR_AI_DATA_DIR", BASE_DIR / "data"))
UPLOADS_DIR = DATA_DIR / "uploads"
SAMPLE_DIR = DATA_DIR / "sample"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
REPORTS_DIR = DATA_DIR / "reports"
DB_PATH = DATA_DIR / "tabular_lab.db"
STORAGE_DIR = Path(__file__).resolve().parents[2] / "storage"
KAGGLE_CREDENTIALS_PATH = STORAGE_DIR / "kaggle.json"
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))

# Ensure directories exist
for d in [UPLOADS_DIR, SAMPLE_DIR, ARTIFACTS_DIR, REPORTS_DIR, STORAGE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Limits
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "200"))
MAX_KAGGLE_DOWNLOAD_MB = int(os.getenv("MAX_KAGGLE_DOWNLOAD_MB", str(MAX_UPLOAD_SIZE_MB)))
MAX_SYNTHETIC_ROWS = int(os.getenv("MAX_SYNTHETIC_ROWS", "1000000"))
MAX_ROWS_TRAINING = 100_000
MAX_ROWS_PREVIEW = 100

# CORS
FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]

APP_TITLE = "Tabular AI Research Lab"
APP_VERSION = "1.0.0"
