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
        res = test_client.post(
            "/api/auth/register",
            json={
                "name": "Primary Researcher",
                "email": "primary@example.com",
                "password": "SecurePassword123!",
                "confirm_password": "SecurePassword123!",
            },
        )
        assert res.status_code in (201, 409), res.text
        if res.status_code == 409:
            login_res = test_client.post(
                "/api/auth/login",
                json={
                    "email": "primary@example.com",
                    "password": "SecurePassword123!",
                },
            )
            assert login_res.status_code == 200, login_res.text
        yield test_client


@pytest.fixture
def unauthenticated_client():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def create_authenticated_client():
    from fastapi.testclient import TestClient
    from app.main import app

    clients = []

    def _create(email="user_b@example.com", password="SecurePassword123!", name="Second User"):
        test_client = TestClient(app)
        clients.append(test_client)
        res = test_client.post(
            "/api/auth/register",
            json={
                "name": name,
                "email": email,
                "password": password,
                "confirm_password": password,
            },
        )
        assert res.status_code in (201, 409), res.text
        if res.status_code == 409:
            login_res = test_client.post(
                "/api/auth/login",
                json={"email": email, "password": password},
            )
            assert login_res.status_code == 200, login_res.text
        return test_client

    yield _create

    for c in clients:
        c.close()
