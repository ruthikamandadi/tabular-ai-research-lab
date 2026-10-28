import io
import time
from contextlib import contextmanager

import pandas as pd
import pytest


def make_csv(rows):
    buffer = io.StringIO()
    pd.DataFrame(rows).to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def upload_dataset(client, filename, rows):
    response = client.post(
        "/api/datasets/upload",
        files={"file": (filename, make_csv(rows), "text/csv")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def classification_rows(count=60):
    return [
        {
            "feature_a": index / count,
            "feature_b": (index % 7) * 0.5,
            "target": int(index % 2 == 0),
        }
        for index in range(count)
    ]


def test_health(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.parametrize(
    ("model_id", "model_config"),
    [
        ("logistic_regression", {"max_iter": 200}),
        ("random_forest", {"n_estimators": 10, "max_depth": 4}),
    ],
)
def test_text_and_date_pipeline_models_save_reload_and_predict(
    model_id, model_config, tmp_path
):
    import joblib
    import numpy as np
    import pandas as pd

    from app.models.model_factory import build_model, load_model_wrapper
    from app.preprocessing.pipeline import (
        build_preprocessing_pipeline,
        detect_column_types,
    )

    row_count = 80
    target = np.arange(row_count) % 2
    frame = pd.DataFrame(
        {
            "description": [
                f"Customer feedback with detailed service experience number {index % 8}"
                for index in range(row_count)
            ],
            "signup_date": pd.date_range("2025-01-01", periods=row_count).astype(str),
            "numeric_feature": np.arange(row_count, dtype=float),
            "target": target,
        }
    )
    prepared = build_preprocessing_pipeline(
        frame,
        "target",
        "binary_classification",
        {"test_size": 0.2, "random_seed": 7},
    )
    detected = detect_column_types(frame)
    assert detected["description"] == "text"
    assert detected["signup_date"] == "date"
    transformer_names = {name for name, _, _ in prepared["preprocessor"].transformers_}
    assert {"text", "date"} <= transformer_names

    model = build_model(model_id, "binary_classification", model_config)
    model.fit(prepared["X_train"], prepared["y_train"])

    model_path = tmp_path / f"{model_id}.pkl"
    preprocessor_path = tmp_path / f"{model_id}-preprocessor.pkl"
    model.save(str(model_path))
    joblib.dump(prepared["preprocessor"], preprocessor_path)

    reloaded_model = load_model_wrapper(str(model_path), "sklearn")
    reloaded_preprocessor = joblib.load(preprocessor_path)
    input_features = frame[prepared["input_feature_cols"]]
    predictions = reloaded_model.predict(reloaded_preprocessor.transform(input_features))

    assert len(predictions) == row_count
    assert set(np.unique(predictions)) <= {0, 1}


def test_settings_capabilities(client):
    response = client.get("/api/settings/capabilities")

    assert response.status_code == 200
    capabilities = response.json()
    assert len(capabilities) == 6
    assert all(
        {"key", "label", "available", "note", "install"} <= capability.keys()
        for capability in capabilities
    )
    shap = next(capability for capability in capabilities if capability["key"] == "shap")
    assert "200" in shap["note"]
    synthetic = next(capability for capability in capabilities if capability["key"] == "synthetic_data")
    assert synthetic["available"] is True
    assert "Bootstrap generation is available" in synthetic["note"]


def test_dataset_upload(client):
    dataset = upload_dataset(client, "small.csv", classification_rows())

    assert dataset["row_count"] == 60
    assert dataset["col_count"] == 3
    assert dataset["status"] == "ready"


def test_bootstrap_synthesis(client):
    dataset = upload_dataset(client, "synthetic-source.csv", classification_rows())
    response = client.post(
        f"/api/datasets/{dataset['id']}/synthesize",
        json={"method": "bootstrap", "num_rows": 30},
    )
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        status = client.get(f"/api/datasets/synthesis/{job_id}/status")
        assert status.status_code == 200, status.text
        if status.json()["status"] in {"completed", "failed"}:
            break
        time.sleep(0.1)

    result = status.json()
    assert result["status"] == "completed", result
    assert result["dataset"]["is_synthetic"] is True
    assert result["dataset"]["row_count"] == 30
    assert "mean_absolute_correlation_difference" in result["quality_report"]


def test_kaggle_credentials_are_masked(client, tmp_path, monkeypatch):
    from app.core import config

    credentials_path = tmp_path / "storage" / "kaggle.json"
    monkeypatch.setattr(config, "KAGGLE_CREDENTIALS_PATH", credentials_path)
    response = client.post(
        "/api/settings/kaggle",
        json={"username": "test-user", "key": "secret-api-key"},
    )

    assert response.status_code == 200, response.text
    assert "secret-api-key" not in response.text
    assert response.json()["masked_key"].startswith("se")
    assert credentials_path.exists()


def test_kaggle_search_reports_missing_credentials(client, tmp_path, monkeypatch):
    from app.api import kaggle

    monkeypatch.setattr(kaggle, "KAGGLE_CREDENTIALS_PATH", tmp_path / "missing.json")
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    response = client.get("/api/kaggle/search", params={"q": "test dataset"})

    assert response.status_code == 400
    assert "credentials are missing" in response.json()["detail"]


def test_kaggle_connection_test_uses_saved_credentials_and_mock_client(client, tmp_path, monkeypatch):
    import builtins
    import os
    from types import SimpleNamespace

    from app.api import kaggle

    credentials_path = tmp_path / "kaggle.json"
    credentials_path.write_text(
        '{"username": "saved-user", "key": "saved-secret-key"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(kaggle, "KAGGLE_CREDENTIALS_PATH", credentials_path)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    calls = {}

    class MockKaggleClient:
        def authenticate(self):
            calls["auth_environment"] = (
                os.environ.get("KAGGLE_USERNAME"),
                os.environ.get("KAGGLE_KEY"),
            )

        def dataset_list(self, page):
            calls["page"] = page
            return []

    original_import = builtins.__import__

    def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "kaggle.api.kaggle_api_extended":
            return SimpleNamespace(KaggleApi=MockKaggleClient)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    response = client.post("/api/settings/kaggle/test")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "message": "Kaggle connection successful. Credentials were verified.",
    }
    assert calls == {
        "auth_environment": ("saved-user", "saved-secret-key"),
        "page": 1,
    }
    assert "KAGGLE_USERNAME" not in os.environ
    assert "KAGGLE_KEY" not in os.environ
    assert "saved-secret-key" not in response.text


def test_kaggle_connection_test_reports_missing_credentials(client, tmp_path, monkeypatch):
    from app.api import kaggle

    monkeypatch.setattr(kaggle, "KAGGLE_CREDENTIALS_PATH", tmp_path / "missing.json")
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    response = client.post("/api/settings/kaggle/test")

    assert response.status_code == 400
    assert response.json()["ok"] is False
    assert "No Kaggle credentials saved" in response.json()["message"]


@pytest.mark.parametrize(
    ("status_code", "failure", "expected_message"),
    [
        (401, "unauthorized", "Kaggle credentials are invalid"),
        (403, "forbidden", "Kaggle denied access"),
        (None, "network", "network error"),
    ],
)
def test_kaggle_connection_test_returns_clear_errors(
    client, tmp_path, monkeypatch, caplog, status_code, failure, expected_message
):
    from app.api import kaggle

    credentials_path = tmp_path / "kaggle.json"
    credentials_path.write_text(
        '{"username": "saved-user", "key": "saved-secret-key"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(kaggle, "KAGGLE_CREDENTIALS_PATH", credentials_path)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    class KaggleApiError(Exception):
        def __init__(self, status, message):
            self.status_code = status
            super().__init__(message)

    class MockKaggleClient:
        def dataset_list(self, page):
            if failure == "network":
                raise ConnectionError("network unavailable; key=saved-secret-key")
            raise KaggleApiError(status_code, f"request failed; key=saved-secret-key")

    @contextmanager
    def mock_kaggle_api(credentials):
        yield MockKaggleClient()

    monkeypatch.setattr(kaggle, "_kaggle_api", mock_kaggle_api)
    response = client.post("/api/settings/kaggle/test")

    assert response.status_code == (502 if status_code is None else status_code)
    assert response.json()["ok"] is False
    assert expected_message.lower() in response.json()["message"].lower()
    assert "saved-secret-key" not in response.text
    assert "saved-secret-key" not in caplog.text


def test_target_validation_errors(client):
    rows = [
        {"row_id": index, "feature": index % 3, "target": index % 2}
        for index in range(40)
    ]
    dataset = upload_dataset(client, "identifiers.csv", rows)

    missing = client.post(
        f"/api/datasets/{dataset['id']}/suggest-task",
        json={"target_column": "missing"},
    )
    assert missing.status_code == 400
    assert "not found" in missing.json()["detail"]

    identifier = client.post(
        f"/api/datasets/{dataset['id']}/suggest-task",
        json={"target_column": "row_id"},
    )
    assert identifier.status_code == 400
    assert "identifier" in identifier.json()["detail"]


@pytest.fixture(scope="module")
def trained_experiment(client):
    dataset = upload_dataset(client, "training.csv", classification_rows())
    response = client.post(
        "/api/experiments",
        json={
            "dataset_id": dataset["id"],
            "target_column": "target",
            "task_type": "binary_classification",
            "model_id": "logistic_regression",
            "model_config": {"max_iter": 100},
            "preprocessing_config": {"test_size": 0.2, "numeric_scaler": "standard"},
            "random_seed": 17,
        },
    )
    assert response.status_code == 200, response.text
    experiment_id = response.json()["experiment_id"]

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        status = client.get(f"/api/experiments/{experiment_id}/status")
        assert status.status_code == 200, status.text
        if status.json()["status"] in {"completed", "failed"}:
            break
        time.sleep(0.1)

    assert status.json()["status"] == "completed", status.json()
    return experiment_id


def test_train_small_model(client, trained_experiment):
    response = client.get(f"/api/experiments/{trained_experiment}")

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["metrics"]["accuracy"] >= 0


def test_batch_prediction(client, trained_experiment):
    test_csv = make_csv([
        {"feature_a": 0.2, "feature_b": 1.0},
        {"feature_a": 0.8, "feature_b": 2.0},
    ])
    response = client.post(
        f"/api/models/{trained_experiment}/predict-csv",
        files={"file": ("prediction.csv", test_csv, "text/csv")},
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/csv")
    assert "prediction" in response.text.splitlines()[0]


def test_complete_churn_workflow(client):
    rows = []
    for index in range(120):
        rows.append({
            "tenure_months": (index * 7) % 72 + 1,
            "monthly_charges": round(25 + (index * 13) % 950 / 10, 2),
            "support_calls": index % 6,
            "contract_type": ("month-to-month", "one-year", "two-year")[index % 3],
            "churn": "Yes" if (index + index // 5) % 2 else "No",
        })

    upload = client.post(
        "/api/datasets/upload",
        files={"file": ("churn_workflow.csv", make_csv(rows), "text/csv")},
    )
    assert upload.status_code == 200, upload.text
    dataset = upload.json()
    dataset_id = dataset["id"]

    profile = client.get(f"/api/datasets/{dataset_id}/profile")
    assert profile.status_code == 200, profile.text
    assert profile.json()["n_rows"] == 120

    target = client.post(
        f"/api/datasets/{dataset_id}/suggest-task",
        json={"target_column": "churn"},
    )
    assert target.status_code == 200, target.text
    assert target.json()["suggestion"] == "binary_classification"

    confirmed = client.get(
        f"/api/datasets/{dataset_id}/target-distribution",
        params={"target_column": "churn", "task_type": "binary_classification"},
    )
    assert confirmed.status_code == 200, confirmed.text

    from app.api.datasets import _load_csv_safe
    from app.core.config import UPLOADS_DIR
    from app.preprocessing.pipeline import build_preprocessing_pipeline

    preprocessing = build_preprocessing_pipeline(
        _load_csv_safe(str(next(UPLOADS_DIR.glob(f"{dataset_id}_*")))),
        "churn",
        "binary_classification",
        {"test_size": 0.2, "numeric_scaler": "standard"},
    )
    assert preprocessing["n_train"] + preprocessing["n_test"] == 120
    assert preprocessing["n_features"] > 0

    experiment_ids = []
    for model_id, model_config in (
        ("logistic_regression", {"max_iter": 200}),
        ("random_forest", {"n_estimators": 20, "max_depth": 5}),
    ):
        created = client.post(
            "/api/experiments",
            json={
                "dataset_id": dataset_id,
                "target_column": "churn",
                "task_type": "binary_classification",
                "model_id": model_id,
                "model_config": model_config,
                "preprocessing_config": {"test_size": 0.2, "numeric_scaler": "standard"},
                "random_seed": 7,
            },
        )
        assert created.status_code == 200, created.text
        experiment_id = created.json()["experiment_id"]
        experiment_ids.append(experiment_id)

        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            status = client.get(f"/api/experiments/{experiment_id}/status")
            assert status.status_code == 200, status.text
            if status.json()["status"] in {"completed", "failed"}:
                break
            time.sleep(0.1)
        assert status.json()["status"] == "completed", status.json()

        metrics = client.get(f"/api/experiments/{experiment_id}/results")
        assert metrics.status_code == 200, metrics.text
        assert metrics.json()["metrics"]["accuracy"] >= 0

        explanations = client.get(f"/api/experiments/{experiment_id}/explanations")
        assert explanations.status_code == 200, explanations.text
        assert explanations.json()["importances"]

        single_prediction = client.post(
            f"/api/models/{experiment_id}/predict-single",
            json={"features": {
                "tenure_months": 12,
                "monthly_charges": 75.0,
                "support_calls": 2,
                "contract_type": "one-year",
            }},
        )
        assert single_prediction.status_code == 200, single_prediction.text
        assert single_prediction.json()["prediction"] in {"Yes", "No"}

        batch_prediction = client.post(
            f"/api/models/{experiment_id}/predict-csv",
            files={
                "file": (
                    "batch.csv",
                    make_csv([
                        {"tenure_months": 12, "monthly_charges": 75.0, "support_calls": 2, "contract_type": "one-year"},
                        {"tenure_months": 30, "monthly_charges": 45.5, "support_calls": 0, "contract_type": "two-year"},
                    ]),
                    "text/csv",
                )
            },
        )
        assert batch_prediction.status_code == 200, batch_prediction.text
        assert batch_prediction.headers["content-type"].startswith("text/csv")
        assert "prediction" in batch_prediction.text.splitlines()[0]

        html_report = client.get(f"/api/reports/{experiment_id}/html")
        assert html_report.status_code == 200, html_report.text
        assert "<html" in html_report.text.lower()

        json_report = client.get(f"/api/reports/{experiment_id}/json")
        assert json_report.status_code == 200, json_report.text
        assert json_report.json()["id"] == experiment_id

    experiments = client.get("/api/experiments")
    assert experiments.status_code == 200, experiments.text
    assert set(experiment_ids).issubset({experiment["id"] for experiment in experiments.json()})

    comparison = client.get(
        "/api/reports/compare/csv",
        params={"ids": ",".join(experiment_ids)},
    )
    assert comparison.status_code == 200, comparison.text
    assert comparison.headers["content-type"].startswith("text/csv")
    assert all(experiment_id in comparison.text for experiment_id in experiment_ids)

    synthesis = client.post(
        f"/api/datasets/{dataset_id}/synthesize",
        json={"method": "bootstrap", "num_rows": 60},
    )
    assert synthesis.status_code == 200, synthesis.text
    job_id = synthesis.json()["job_id"]
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        synthesis_status = client.get(f"/api/datasets/synthesis/{job_id}/status")
        assert synthesis_status.status_code == 200, synthesis_status.text
        if synthesis_status.json()["status"] in {"completed", "failed"}:
            break
        time.sleep(0.1)
    assert synthesis_status.json()["status"] == "completed", synthesis_status.json()
    assert synthesis_status.json()["dataset"]["is_synthetic"] is True
    assert synthesis_status.json()["quality_report"]["numeric_columns"]

    deleted = client.delete(f"/api/datasets/{dataset_id}")
    assert deleted.status_code == 200, deleted.text
    missing_after_delete = client.get(f"/api/datasets/{dataset_id}/profile")
    assert missing_after_delete.status_code == 404


# =====================================================================
# Authentication & Multi-user Security Tests
# =====================================================================

def test_unauthenticated_access_is_blocked(unauthenticated_client):
    endpoints = [
        ("GET", "/api/datasets"),
        ("GET", "/api/experiments"),
        ("GET", "/api/auth/me"),
        ("POST", "/api/auth/logout"),
    ]
    for method, path in endpoints:
        if method == "GET":
            response = unauthenticated_client.get(path)
        else:
            response = unauthenticated_client.post(path)
        assert response.status_code == 401, f"{path} should return 401"
        assert "Authentication required" in response.json()["detail"]


def test_registration_validation(unauthenticated_client):
    # Passwords do not match
    res = unauthenticated_client.post(
        "/api/auth/register",
        json={
            "name": "Validation User",
            "email": "val1@example.com",
            "password": "Password12345!",
            "confirm_password": "DifferentPassword12345!",
        },
    )
    assert res.status_code == 422
    assert "Passwords do not match" in res.text

    # Password too short (< 12)
    res = unauthenticated_client.post(
        "/api/auth/register",
        json={
            "name": "Validation User",
            "email": "val2@example.com",
            "password": "Short1!",
            "confirm_password": "Short1!",
        },
    )
    assert res.status_code == 422

    # Invalid email
    res = unauthenticated_client.post(
        "/api/auth/register",
        json={
            "name": "Validation User",
            "email": "not-an-email",
            "password": "Password12345!",
            "confirm_password": "Password12345!",
        },
    )
    assert res.status_code == 422


def test_auth_lifecycle(unauthenticated_client):
    email = "lifecycle@example.com"
    password = "SecureLifecyclePassword123!"

    # 1. Register
    reg_res = unauthenticated_client.post(
        "/api/auth/register",
        json={
            "name": "Lifecycle Tester",
            "email": email,
            "password": password,
            "confirm_password": password,
        },
    )
    assert reg_res.status_code == 201
    assert reg_res.json()["user"]["email"] == email
    assert "tabular_session" in unauthenticated_client.cookies

    # 2. Get current user
    me_res = unauthenticated_client.get("/api/auth/me")
    assert me_res.status_code == 200
    assert me_res.json()["user"]["email"] == email

    # 3. Duplicate registration is rejected
    dup_res = unauthenticated_client.post(
        "/api/auth/register",
        json={
            "name": "Lifecycle Tester 2",
            "email": email,
            "password": password,
            "confirm_password": password,
        },
    )
    assert dup_res.status_code == 409

    # 4. Logout
    logout_res = unauthenticated_client.post("/api/auth/logout")
    assert logout_res.status_code == 200

    # 5. Me after logout returns 401
    me_after = unauthenticated_client.get("/api/auth/me")
    assert me_after.status_code == 401

    # 6. Login with wrong password fails
    bad_login = unauthenticated_client.post(
        "/api/auth/login",
        json={"email": email, "password": "WrongPassword123!"},
    )
    assert bad_login.status_code == 401

    # 7. Login with correct password succeeds
    login_res = unauthenticated_client.post(
        "/api/auth/login",
        json={"email": email, "password": password},
    )
    assert login_res.status_code == 200
    assert login_res.json()["user"]["email"] == email

    # Session persisted
    me_again = unauthenticated_client.get("/api/auth/me")
    assert me_again.status_code == 200
    assert me_again.json()["user"]["email"] == email


def test_multi_user_isolation(client, create_authenticated_client):
    user_b_client = create_authenticated_client(
        email="user_isolation_b@example.com",
        password="SecureIsolationPassword123!",
        name="User B",
    )

    # User A uploads a dataset
    ds_a = upload_dataset(client, "user_a_secret.csv", classification_rows())
    ds_a_id = ds_a["id"]

    # User A sees it
    user_a_datasets = client.get("/api/datasets").json()
    assert any(d["id"] == ds_a_id for d in user_a_datasets)

    # User B lists datasets: must NOT see User A's dataset
    user_b_datasets = user_b_client.get("/api/datasets").json()
    assert not any(d["id"] == ds_a_id for d in user_b_datasets)

    # User B cannot preview User A's dataset (404)
    preview_b = user_b_client.get(f"/api/datasets/{ds_a_id}/preview")
    assert preview_b.status_code == 404

    # User B cannot profile User A's dataset (404)
    profile_b = user_b_client.get(f"/api/datasets/{ds_a_id}/profile")
    assert profile_b.status_code == 404

    # User B cannot delete User A's dataset (404)
    delete_b = user_b_client.delete(f"/api/datasets/{ds_a_id}")
    assert delete_b.status_code == 404

    # User B cannot trigger experiment on User A's dataset (404)
    exp_b = user_b_client.post(
        "/api/experiments",
        json={
            "dataset_id": ds_a_id,
            "target_column": "target",
            "task_type": "binary_classification",
            "model_id": "logistic_regression",
        },
    )
    assert exp_b.status_code == 404

    # User A creates an experiment
    exp_a_res = client.post(
        "/api/experiments",
        json={
            "dataset_id": ds_a_id,
            "target_column": "target",
            "task_type": "binary_classification",
            "model_id": "logistic_regression",
            "model_config": {"max_iter": 50},
            "preprocessing_config": {"test_size": 0.2},
            "random_seed": 42,
        },
    )
    assert exp_a_res.status_code == 200, exp_a_res.text
    exp_a_id = exp_a_res.json()["experiment_id"]

    # User B cannot view User A's experiment status or details
    assert user_b_client.get(f"/api/experiments/{exp_a_id}").status_code == 404
    assert user_b_client.get(f"/api/experiments/{exp_a_id}/status").status_code == 404
    assert user_b_client.get(f"/api/reports/{exp_a_id}/json").status_code == 404
    assert user_b_client.get(f"/api/reports/{exp_a_id}/html").status_code == 404
    assert user_b_client.get(f"/api/models/{exp_a_id}/schema").status_code == 404

    # User B cannot download or rename User A's dataset
    assert user_b_client.get(f"/api/datasets/{ds_a_id}/download").status_code == 404
    assert user_b_client.patch(f"/api/datasets/{ds_a_id}", json={"name": "stolen.csv"}).status_code == 404

    # User A can download and rename their own dataset
    dl_res = client.get(f"/api/datasets/{ds_a_id}/download")
    assert dl_res.status_code == 200
    assert "text/csv" in dl_res.headers.get("content-type", "")

    rename_res = client.patch(f"/api/datasets/{ds_a_id}", json={"name": "user_a_renamed.csv"})
    assert rename_res.status_code == 200
    assert rename_res.json()["name"] == "user_a_renamed.csv"

    # Search filter finds renamed dataset
    search_res = client.get("/api/datasets?q=renamed")
    assert search_res.status_code == 200
    assert any(d["id"] == ds_a_id for d in search_res.json())


def test_experiment_training_status_serialization_and_db_fallback(client):
    import torch
    import numpy as np
    from app.experiments.runner import _jobs, _jobs_lock

    ds = upload_dataset(client, "status_test.csv", classification_rows(count=60))
    res = client.post(
        "/api/experiments",
        json={
            "dataset_id": ds["id"],
            "target_column": "target",
            "task_type": "binary_classification",
            "model_id": "logistic_regression",
            "model_config": {"max_iter": 50},
            "preprocessing_config": {"test_size": 0.2},
            "random_seed": 42,
        },
    )
    assert res.status_code == 200
    exp_id = res.json()["experiment_id"]

    # Wait for the background worker to finish so it doesn't race with our test injection
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        poll = client.get(f"/api/experiments/{exp_id}/status")
        if poll.status_code == 200 and poll.json()["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)

    # 1. Test in-memory job containing torch.device and numpy values
    with _jobs_lock:
        _jobs[exp_id] = {
            "experiment_id": exp_id,
            "status": "running",
            "progress": 45,
            "started_at": time.time() - 2.5,
            "message": "Epoch 5/10",
            "device": torch.device("cpu"),
            "metrics": {"loss": np.float32(0.354)},
            "training_history": [{"epoch": 1, "train_loss": np.float64(0.5)}],
            "error": None,
            "warning": None,
        }

    status_res = client.get(f"/api/experiments/{exp_id}/status")
    assert status_res.status_code == 200, status_res.text
    data = status_res.json()
    assert data["experiment_id"] == exp_id
    assert data["status"] == "running"
    assert data["progress"] == 45
    assert data["device"] == "cpu"
    assert isinstance(data["elapsed_seconds"], (int, float))
    assert data["elapsed_seconds"] >= 2.0
    assert abs(data["metrics"]["loss"] - 0.354) < 1e-3

    # 2. Test DB fallback when job is evicted from in-memory runner or server restarted
    with _jobs_lock:
        _jobs.pop(exp_id, None)

    fallback_res = client.get(f"/api/experiments/{exp_id}/status")
    assert fallback_res.status_code == 200, fallback_res.text
    fb_data = fallback_res.json()
    assert fb_data["experiment_id"] == exp_id
    assert fb_data["status"] == "completed"
    assert fb_data["progress"] == 100

    # 3. Test non-existent experiment returns 404
    missing_res = client.get("/api/experiments/non-existent-id/status")
    assert missing_res.status_code == 404


def test_pytorch_mlp_training_status_poll(client):
    ds = upload_dataset(client, "pytorch_status.csv", classification_rows(count=80))
    res = client.post(
        "/api/experiments",
        json={
            "dataset_id": ds["id"],
            "target_column": "target",
            "task_type": "binary_classification",
            "model_id": "pytorch_mlp",
            "model_config": {"epochs": 3, "batch_size": 16},
            "preprocessing_config": {"test_size": 0.2},
            "random_seed": 42,
        },
    )
    assert res.status_code == 200
    exp_id = res.json()["experiment_id"]

    deadline = time.monotonic() + 30
    terminal = False
    while time.monotonic() < deadline:
        poll_res = client.get(f"/api/experiments/{exp_id}/status")
        assert poll_res.status_code == 200, f"Poll returned HTTP {poll_res.status_code}: {poll_res.text}"
        poll_data = poll_res.json()
        assert poll_data["device"] in ("cpu", "cuda", None)
        if poll_data["status"] in ("completed", "failed"):
            terminal = True
            assert poll_data["status"] == "completed", f"MLP training failed: {poll_data.get('error')}"
            assert poll_data["progress"] == 100
            assert poll_data["metrics"] is not None
            assert "accuracy" in poll_data["metrics"]
            break
        time.sleep(0.2)

    assert terminal, "PyTorch MLP training timed out"


def make_demo_churn_csv():
    rows = []
    for index in range(120):
        tenure = (index * 11) % 72 + 1
        monthly_charges = 35 + ((index * 7) % 20) * 5
        contract = (
            "month-to-month"
            if index % 5 < 3
            else ("one-year" if index % 5 == 3 else "two-year")
        )
        support_calls = (index * 7) % 8
        internet = ["fiber", "dsl", "no"][index % 3]
        churn = (
            "Yes"
            if (contract == "month-to-month" and support_calls >= 3)
            or (tenure < 10 and monthly_charges > 80)
            else "No"
        )
        rows.append(
            {
                "tenure_months": tenure,
                "monthly_charges": float(f"{monthly_charges:.2f}"),
                "total_charges": float(f"{(tenure * monthly_charges):.2f}"),
                "contract_type": contract,
                "internet_service": internet,
                "payment_method": ["electronic", "credit_card", "bank_transfer"][
                    index % 3
                ],
                "num_support_calls": support_calls,
                "tech_support": "yes" if index % 2 == 0 else "no",
                "senior_citizen": 1 if index % 4 == 0 else 0,
                "churn": churn,
            }
        )
    buffer = io.StringIO()
    pd.DataFrame(rows).to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def test_e2e_demo_churn_full_workflow(unauthenticated_client, monkeypatch, tmp_path):
    import builtins
    import os
    from types import SimpleNamespace
    from app.api import kaggle

    # 1. Register
    reg_res = unauthenticated_client.post(
        "/api/auth/register",
        json={
            "name": "E2E Churn Tester",
            "email": "churn_e2e_user@example.com",
            "password": "SecurePassword123!",
            "confirm_password": "SecurePassword123!",
        },
    )
    assert reg_res.status_code == 201, reg_res.text

    # 2. Login
    login_res = unauthenticated_client.post(
        "/api/auth/login",
        json={
            "email": "churn_e2e_user@example.com",
            "password": "SecurePassword123!",
        },
    )
    assert login_res.status_code == 200, login_res.text

    # 3. Upload 120-row demo churn dataset
    churn_bytes = make_demo_churn_csv()
    upload_res = unauthenticated_client.post(
        "/api/datasets/upload",
        files={"file": ("demo_churn_120.csv", churn_bytes, "text/csv")},
    )
    assert upload_res.status_code == 200, upload_res.text
    dataset = upload_res.json()
    dataset_id = dataset["id"]
    assert dataset["row_count"] == 120

    # 4. Data profile
    profile_res = unauthenticated_client.get(f"/api/datasets/{dataset_id}/profile")
    assert profile_res.status_code == 200, profile_res.text
    profile = profile_res.json()
    assert profile["n_rows"] == 120
    assert "columns" in profile

    # 5. Confirm target (churn, classification)
    suggest_res = unauthenticated_client.post(
        f"/api/datasets/{dataset_id}/suggest-task",
        json={"target_column": "churn"},
    )
    assert suggest_res.status_code == 200, suggest_res.text
    assert suggest_res.json()["suggestion"] == "binary_classification"

    dist_res = unauthenticated_client.get(
        f"/api/datasets/{dataset_id}/target-distribution?target_column=churn&task_type=binary_classification"
    )
    assert dist_res.status_code == 200, dist_res.text
    dist = dist_res.json()
    assert dist["type"] == "categorical"
    assert "counts" in dist

    # 6. Preprocessing config
    prep_cfg = {
        "test_size": 0.2,
        "random_seed": 42,
        "numeric_imputer": "median",
        "numeric_scaler": "standard",
        "categorical_imputer": "most_frequent",
        "categorical_encoding": "onehot",
        "drop_identifiers": True,
    }

    # 7. Train Logistic Regression and Random Forest
    lr_res = unauthenticated_client.post(
        "/api/experiments",
        json={
            "dataset_id": dataset_id,
            "target_column": "churn",
            "task_type": "binary_classification",
            "model_id": "logistic_regression",
            "model_config": {"max_iter": 300},
            "preprocessing_config": prep_cfg,
            "random_seed": 42,
        },
    )
    assert lr_res.status_code == 200, lr_res.text
    lr_exp_id = lr_res.json()["experiment_id"]

    rf_res = unauthenticated_client.post(
        "/api/experiments",
        json={
            "dataset_id": dataset_id,
            "target_column": "churn",
            "task_type": "binary_classification",
            "model_id": "random_forest",
            "model_config": {"n_estimators": 40, "max_depth": 6},
            "preprocessing_config": prep_cfg,
            "random_seed": 42,
        },
    )
    assert rf_res.status_code == 200, rf_res.text
    rf_exp_id = rf_res.json()["experiment_id"]

    # Poll status for both models
    for exp_id in (lr_exp_id, rf_exp_id):
        deadline = time.monotonic() + 30
        finished = False
        while time.monotonic() < deadline:
            st = unauthenticated_client.get(f"/api/experiments/{exp_id}/status")
            assert st.status_code == 200, st.text
            status_data = st.json()
            if status_data["status"] in ("completed", "failed"):
                assert status_data["status"] == "completed", status_data
                finished = True
                break
            time.sleep(0.1)
        assert finished, f"Experiment {exp_id} did not finish in time"

    # 8. List experiments
    list_res = unauthenticated_client.get("/api/experiments")
    assert list_res.status_code == 200, list_res.text
    exp_ids = {e["id"] for e in list_res.json()}
    assert {lr_exp_id, rf_exp_id} <= exp_ids

    # 9. Metrics (accuracy, F1, ROC-AUC)
    lr_exp = unauthenticated_client.get(f"/api/experiments/{lr_exp_id}").json()
    assert lr_exp["metrics"] is not None
    assert "accuracy" in lr_exp["metrics"]
    assert "f1" in lr_exp["metrics"]
    assert "roc_auc" in lr_exp["metrics"]

    rf_exp = unauthenticated_client.get(f"/api/experiments/{rf_exp_id}").json()
    assert rf_exp["metrics"] is not None
    assert "accuracy" in rf_exp["metrics"]
    assert "f1" in rf_exp["metrics"]
    assert "roc_auc" in rf_exp["metrics"]

    # 10. Feature importance and SHAP
    shap_res = unauthenticated_client.get(f"/api/experiments/{lr_exp_id}/explanations")
    assert shap_res.status_code == 200, shap_res.text
    shap_data = shap_res.json()
    assert "importances" in shap_data
    assert len(shap_data["importances"]) > 0
    assert "method" in shap_data

    # 11. Single prediction
    single_res = unauthenticated_client.post(
        f"/api/models/{lr_exp_id}/predict-single",
        json={
            "features": {
                "tenure_months": 12,
                "monthly_charges": 65.0,
                "total_charges": 780.0,
                "contract_type": "month-to-month",
                "internet_service": "fiber",
                "payment_method": "electronic",
                "num_support_calls": 4,
                "tech_support": "no",
                "senior_citizen": 0,
            }
        },
    )
    assert single_res.status_code == 200, single_res.text
    single_data = single_res.json()
    assert "prediction" in single_data

    # 12. Batch CSV prediction (file picker, FormData field "file", blob download)
    batch_rows = [
        {
            "tenure_months": 24,
            "monthly_charges": 45.0,
            "total_charges": 1080.0,
            "contract_type": "one-year",
            "internet_service": "dsl",
            "payment_method": "credit_card",
            "num_support_calls": 1,
            "tech_support": "yes",
            "senior_citizen": 0,
        },
        {
            "tenure_months": 2,
            "monthly_charges": 95.0,
            "total_charges": 190.0,
            "contract_type": "month-to-month",
            "internet_service": "fiber",
            "payment_method": "electronic",
            "num_support_calls": 5,
            "tech_support": "no",
            "senior_citizen": 0,
        },
    ]
    batch_bytes = make_csv(batch_rows)
    batch_res = unauthenticated_client.post(
        f"/api/models/{lr_exp_id}/predict-csv",
        files={"file": ("batch_input.csv", batch_bytes, "text/csv")},
    )
    assert batch_res.status_code == 200, batch_res.text
    assert "text/csv" in batch_res.headers.get("content-type", "")
    assert "prediction" in batch_res.text

    # 13. HTML and JSON report
    html_res = unauthenticated_client.get(f"/api/reports/{lr_exp_id}/html")
    assert html_res.status_code == 200, html_res.text
    assert "text/html" in html_res.headers.get("content-type", "")
    assert "<html" in html_res.text.lower()

    json_res = unauthenticated_client.get(f"/api/reports/{lr_exp_id}/json")
    assert json_res.status_code == 200, json_res.text
    report_json = json_res.json()
    assert report_json["id"] == lr_exp_id
    assert "metrics" in report_json

    # 14. Compare two experiments
    comp_res = unauthenticated_client.get(
        f"/api/reports/compare/csv?ids={lr_exp_id},{rf_exp_id}"
    )
    assert comp_res.status_code == 200, comp_res.text
    assert "text/csv" in comp_res.headers.get("content-type", "")
    assert lr_exp_id in comp_res.text
    assert rf_exp_id in comp_res.text

    # 15. Bootstrap synthetic data
    synth_res = unauthenticated_client.post(
        f"/api/datasets/{dataset_id}/synthesize",
        json={"method": "bootstrap", "num_rows": 40},
    )
    assert synth_res.status_code == 200, synth_res.text
    job_id = synth_res.json()["job_id"]

    deadline = time.monotonic() + 15
    synth_done = False
    while time.monotonic() < deadline:
        s_poll = unauthenticated_client.get(f"/api/datasets/synthesis/{job_id}/status")
        assert s_poll.status_code == 200, s_poll.text
        s_data = s_poll.json()
        if s_data["status"] in ("completed", "failed"):
            assert s_data["status"] == "completed", s_data
            synth_done = True
            break
        time.sleep(0.1)
    assert synth_done

    # 16. Kaggle credentials save and test (mocked)
    from app.core import config

    creds_path = tmp_path / "storage" / "kaggle.json"
    monkeypatch.setattr(config, "KAGGLE_CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(kaggle, "KAGGLE_CREDENTIALS_PATH", creds_path)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    save_kaggle_res = unauthenticated_client.post(
        "/api/settings/kaggle",
        json={"username": "churn_researcher", "key": "kaggle_secret_key_123"},
    )
    assert save_kaggle_res.status_code == 200, save_kaggle_res.text
    assert save_kaggle_res.json()["masked_key"].startswith("ka")

    class MockKaggleClient:
        def authenticate(self):
            pass

        def dataset_list(self, page):
            return []

    original_import = builtins.__import__

    def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "kaggle.api.kaggle_api_extended":
            return SimpleNamespace(KaggleApi=MockKaggleClient)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    test_kaggle_res = unauthenticated_client.post("/api/settings/kaggle/test")
    assert test_kaggle_res.status_code == 200, test_kaggle_res.text
    assert test_kaggle_res.json()["ok"] is True

    # 17. Delete dataset
    del_res = unauthenticated_client.delete(f"/api/datasets/{dataset_id}")
    assert del_res.status_code == 200, del_res.text
    assert del_res.json()["ok"] is True

    # 18. Logout
    logout_res = unauthenticated_client.post("/api/auth/logout")
    assert logout_res.status_code == 200, logout_res.text
    assert logout_res.json()["message"] in ("Signed out.", "Logged out")




def test_time_series_split_is_chronological():
    import numpy as np
    import pandas as pd
    from app.preprocessing.pipeline import build_preprocessing_pipeline

    n = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d")
    df = pd.DataFrame({
        "date": dates,
        "sales": np.arange(n, dtype=float) * 1.5 + 10.0,
        "feature": np.sin(np.arange(n)),
    })
    prep = build_preprocessing_pipeline(
        df,
        "sales",
        "time_series",
        {"time_series_date_column": "date", "test_size": 0.2, "time_series_lags": [1, 7], "time_series_rolling_windows": [3]},
    )
    assert prep["is_time_series"] is True
    assert prep["test_dates"] is not None
    test_dates = pd.to_datetime(prep["test_dates"])
    assert test_dates.is_monotonic_increasing
    assert len(test_dates) == prep["n_test"]
    assert prep["n_train"] > prep["n_test"]


def test_time_series_lags_use_only_past_values():
    import numpy as np
    import pandas as pd
    from app.preprocessing.pipeline import build_preprocessing_pipeline

    n = 40
    dates = pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d")
    sales = np.arange(n, dtype=float) * 10
    df = pd.DataFrame({
        "date": dates,
        "sales": sales,
    })
    lags = [1, 7, 14]
    windows = [3, 7]
    prep = build_preprocessing_pipeline(
        df,
        "sales",
        "time_series",
        {
            "time_series_date_column": "date",
            "time_series_lags": lags,
            "time_series_rolling_windows": windows,
            "numeric_scaler": "none",
        },
    )
    expected_rows = n - 14
    assert prep["n_train"] + prep["n_test"] == expected_rows
    feat_names = prep["feature_names"]
    assert any("month" in name for name in feat_names)
    assert any("weekday" in name for name in feat_names)
    assert any("is_weekend" in name for name in feat_names)
    for lag in lags:
        assert any(f"sales_lag_{lag}" in name for name in feat_names)


def test_time_series_experiment_trains_and_returns_mae_and_rmse(client):
    import numpy as np
    import pandas as pd

    n = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d")
    rows = [
        {
            "date": dates[i],
            "sales": float(50.0 + i * 2.0 + (i % 7) * 3.0),
            "promotions": int(i % 5 == 0),
        }
        for i in range(n)
    ]
    uploaded = upload_dataset(client, "test_ts_sales.csv", rows)
    dataset_id = uploaded["id"]

    created = client.post(
        "/api/experiments",
        json={
            "dataset_id": dataset_id,
            "target_column": "sales",
            "task_type": "time_series",
            "model_id": "linear_regression",
            "model_config": {"fit_intercept": True},
            "preprocessing_config": {
                "time_series_date_column": "date",
                "time_series_lags": [1, 7],
                "time_series_rolling_windows": [3],
                "test_size": 0.2,
            },
            "random_seed": 42,
        },
    )
    assert created.status_code == 200, created.text
    experiment_id = created.json()["experiment_id"]

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status_res = client.get(f"/api/experiments/{experiment_id}/status")
        assert status_res.status_code == 200, status_res.text
        status_data = status_res.json()
        if status_data["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)

    assert status_data["status"] == "completed", status_data
    results_res = client.get(f"/api/experiments/{experiment_id}/results")
    assert results_res.status_code == 200, results_res.text
    metrics = results_res.json()["metrics"]
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "mape" in metrics
    assert "r2" in metrics
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0

    forecast_data = results_res.json().get("forecast_data")
    assert forecast_data is not None and len(forecast_data) > 0
    assert "date" in forecast_data[0]
    assert "actual" in forecast_data[0]
    assert "forecast" in forecast_data[0]


def test_image_path_skipped_gracefully_when_torchvision_missing(monkeypatch):
    import numpy as np
    import pandas as pd
    import app.preprocessing.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "_image_dependencies_available", lambda: False)

    n = 30
    df = pd.DataFrame({
        "image_url": [f"https://example.com/images/sample_{i}.png" for i in range(n)],
        "target": np.arange(n, dtype=float) * 2.0,
        "feature_x": (np.arange(n) % 4).astype(float),
    })

    prep = pipeline_mod.build_preprocessing_pipeline(
        df,
        "target",
        "regression",
        {"numeric_scaler": "standard", "column_types": {"feature_x": "numeric"}},
    )
    assert prep is not None
    assert prep["n_train"] > 0
    assert "image_url" in prep["skipped_image_columns"]
    assert any("image_url" in w for w in prep["warnings"])


def test_image_embeddings_extracted_and_cached(tmp_path, monkeypatch):
    from PIL import Image
    import numpy as np
    import pandas as pd
    from app.preprocessing.pipeline import build_preprocessing_pipeline
    from app.core.config import STORAGE_DIR
    import torchvision.models

    orig_resnet18 = torchvision.models.resnet18
    monkeypatch.setattr(torchvision.models, "resnet18", lambda weights=None: orig_resnet18(weights=None))

    img_dir = tmp_path / "test_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_file = img_dir / "sample_img.png"
    img = Image.new("RGB", (32, 32), color=(200, 50, 50))
    img.save(img_file)

    n = 25
    df = pd.DataFrame({
        "img_path": [str(img_file)] * n,
        "target": np.arange(n, dtype=float),
    })

    prep = build_preprocessing_pipeline(
        df,
        "target",
        "regression",
        {"column_types": {"img_path": "image_path"}, "handle_constant": False},
    )
    assert prep is not None
    assert prep["n_features"] == 512
    cache_dir = STORAGE_DIR / "image_cache"
    assert cache_dir.exists()
    cached_files = list(cache_dir.glob("*.npy"))
    assert len(cached_files) > 0


def test_capabilities_reports_real_availability(client):
    res = client.get("/api/settings/capabilities")
    assert res.status_code == 200, res.text
    caps = res.json()
    item = next((c for c in caps if c["key"] == "text_images_timeseries"), None)
    assert item is not None
    assert item["available"] is True
    assert "Time-series forecasting is available." in item["note"]
    assert "ResNet-18 image embeddings are available." in item["note"]
    assert "is not accepted by the experiment API" not in item["note"]
