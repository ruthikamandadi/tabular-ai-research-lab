import os
import sys
import tempfile
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
os.environ["TABULAR_AI_DATA_DIR"] = tempfile.mkdtemp(prefix="tabular-ai-tests-")


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
