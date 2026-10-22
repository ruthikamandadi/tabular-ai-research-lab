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
