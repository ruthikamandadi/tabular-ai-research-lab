"""
Training runner - executes model training as background jobs.
"""
import uuid
import time
import threading
import traceback
import math
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from app.core.config import ARTIFACTS_DIR, UPLOADS_DIR
from app.core.database import SessionLocal, ExperimentRecord
from app.preprocessing.pipeline import build_preprocessing_pipeline
from app.models.model_factory import build_model, get_torch
from app.evaluation.metrics import compute_metrics
from app.explainability.importance import compute_model_importances

import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold, TimeSeriesSplit
from sklearn.metrics import get_scorer

# In-memory job store (progress tracking)
_jobs: Dict[str, Dict] = {}
_jobs_lock = threading.Lock()


def get_job(job_id: str) -> Optional[Dict]:
    with _jobs_lock:
        return _jobs.get(job_id)


def _update_job(job_id: str, **kwargs):
    sanitized = {k: _json_value(v) for k, v in kwargs.items()}
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(sanitized)


def _search_space(model_id: str) -> Dict[str, list]:
    spaces = {
        "linear_regression": {
            "fit_intercept": [True, False],
        },
        "logistic_regression": {
            "C": [0.1, 1.0, 10.0],
            "max_iter": [500, 1000],
        },
        "random_forest": {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        },
        "gradient_boosting": {
            "n_estimators": [100, 200],
            "learning_rate": [0.03, 0.1],
            "max_depth": [2, 3, 5],
        },
        "xgboost": {
            "n_estimators": [100, 200],
            "max_depth": [3, 6],
            "learning_rate": [0.03, 0.1],
            "subsample": [0.7, 1.0],
        },
        "knn": {
            "n_neighbors": [3, 5, 7, 11],
            "weights": ["uniform", "distance"],
        },
    }
    return spaces.get(model_id, {})


def _json_value(value):
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if hasattr(value, "__class__") and "device" in value.__class__.__name__.lower():
        return str(value)
    return value


def _run_tuning(model, model_id, task_type, model_config, X_train, y_train, job_config, job_id):
    method = job_config.get("tuning_method", "none")
    if method == "none":
        return model, {}, None, []

    search_space = _search_space(model_id)
    if not search_space:
        _update_job(job_id, message=f"Tuning is not configured for {model_id}; training with selected parameters.")
        return model, {}, None, []

    folds = max(2, min(int(job_config.get("cv_folds", 3)), len(y_train)))
    if task_type in ("binary_classification", "multiclass_classification"):
        _, class_counts = np.unique(y_train, return_counts=True)
        folds = min(folds, int(class_counts.min()))
        if folds < 2:
            raise ValueError("Cross-validation needs at least two training examples in every class.")
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=job_config.get("random_seed", 42))
    elif task_type == "time_series":
        folds = min(folds, len(y_train) - 1)
        if folds < 2:
            raise ValueError("Time-series cross-validation needs at least three training rows.")
        splitter = TimeSeriesSplit(n_splits=folds)
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=job_config.get("random_seed", 42))

    requested_method = method
    if method == "optuna":
        try:
            import optuna
        except ImportError:
            method = "random"
            _update_job(job_id, message="Optuna is not installed; using randomized search instead.")

    scoring = job_config.get("scoring") or (
        "neg_root_mean_squared_error" if task_type in ("regression", "time_series") else "f1_weighted"
    )
    # Validate scoring before spending time on the search.
    get_scorer(scoring)
    n_iter = max(1, int(job_config.get("n_iter", 20)))
    if len(job_config.get("dataset_path", "")) and job_config.get("dataset_rows", 0) > 50_000 and len(y_train) > 20_000:
        if task_type == "time_series":
            sample_indices = np.arange(len(y_train) - 20_000, len(y_train))
        elif task_type == "regression":
            sample_indices = np.random.RandomState(job_config.get("random_seed", 42)).choice(
                len(y_train), 20_000, replace=False
            )
        else:
            from sklearn.model_selection import train_test_split
            sample_indices, _ = train_test_split(
                np.arange(len(y_train)),
                train_size=20_000,
                stratify=y_train,
                random_state=job_config.get("random_seed", 42),
            )
        X_search, y_search = X_train[sample_indices], y_train[sample_indices]
    else:
        X_search, y_search = X_train, y_train

    trials = []
    if method == "optuna":
        study = optuna.create_study(direction="maximize")

        def objective(trial):
            params = {
                key: trial.suggest_categorical(key, values)
                for key, values in search_space.items()
            }
            candidate = build_model(model_id, task_type, {**model_config, **params}).estimator
            from sklearn.model_selection import cross_val_score
            score = float(np.mean(cross_val_score(
                candidate, X_search, y_search, cv=splitter, scoring=scoring, n_jobs=-1,
            )))
            trial.set_user_attr("params", _json_value(params))
            _update_job(job_id, progress=min(75, 30 + int(45 * (trial.number + 1) / n_iter)),
                        message=f"Hyperparameter tuning trial {trial.number + 1}/{n_iter}")
            return score

        study.optimize(objective, n_trials=n_iter)
        best_params = _json_value(study.best_trial.user_attrs["params"])
        best_score = float(study.best_value)
        for trial in study.trials:
            trials.append({
                "params": _json_value(trial.user_attrs.get("params", {})),
                "mean_score": float(trial.value) if trial.value is not None else None,
                "std_score": None,
            })
    else:
        search_cls = GridSearchCV if method == "grid" else RandomizedSearchCV
        search_kwargs = {
            "estimator": model.estimator,
            "param_grid" if method == "grid" else "param_distributions": search_space,
            "scoring": scoring,
            "cv": splitter,
            "n_jobs": -1,
            "refit": True,
            "error_score": "raise",
        }
        if method != "grid":
            search_kwargs["n_iter"] = n_iter
            search_kwargs["random_state"] = job_config.get("random_seed", 42)
        search = search_cls(**search_kwargs)
        _update_job(job_id, progress=30, message=f"Running {method} hyperparameter search...")
        search.fit(X_search, y_search)
        best_params = _json_value(search.best_params_)
        best_score = float(search.best_score_)
        cv_results = search.cv_results_
        for index, params in enumerate(cv_results["params"]):
            trials.append({
                "params": _json_value(params),
                "mean_score": float(cv_results["mean_test_score"][index]),
                "std_score": float(cv_results["std_test_score"][index]),
            })
        model.estimator = search.best_estimator_

    trials.sort(key=lambda trial: trial["mean_score"] if trial["mean_score"] is not None else float("-inf"), reverse=True)
    model._best_params = best_params
    model._best_cv_score = best_score
    model._tuning_trials = trials
    model._tuning_method = requested_method
    return model, best_params, best_score, trials


def run_training_job(experiment_id: str, job_config: Dict[str, Any]):
    """Execute training in a background thread."""
    job_id = experiment_id

    def _worker():
        db = SessionLocal()
        try:
            _update_job(job_id, status="running", progress=5, message="Loading dataset...")

            # 1. Load dataset
            dataset_path = job_config["dataset_path"]
            df = pd.read_csv(dataset_path, low_memory=False)
            job_config["dataset_rows"] = len(df)
            deep_model = job_config["model_id"] in {"pytorch_mlp", "tabnet", "ft_transformer"}
            torch_module = get_torch() if deep_model else None
            use_cuda = bool(torch_module and torch_module.cuda.is_available())
            if len(df) > 100_000 and (not deep_model or not use_cuda):
                if job_config["task_type"] == "time_series":
                    df = df.tail(100_000)
                else:
                    df = df.sample(100_000, random_state=job_config.get("random_seed", 42))
                if deep_model:
                    device = "cuda" if use_cuda else "cpu"
                    epochs = int(job_config.get("model_config", {}).get("max_epochs", 50))
                    estimate = max(1, int(np.ceil(100_000 * epochs / 500_000)))
                    _update_job(
                        job_id,
                        message=f"CPU deep-model training sampled data to 100,000 rows; estimated time about {estimate} min.",
                        warning=f"Dataset exceeded 100,000 rows; CPU training is sampled to 100,000 rows. Estimated training time: about {estimate} min.",
                        device=device,
                    )
                else:
                    _update_job(job_id, message="Dataset sampled to 100,000 rows for training.")

            _update_job(job_id, progress=15, message="Preprocessing data...")

            # 2. Preprocess
            prep_config = job_config["preprocessing_config"]
            prep_config["random_seed"] = job_config.get("random_seed", 42)
            prep_result = build_preprocessing_pipeline(
                df,
                job_config["target_column"],
                job_config["task_type"],
                prep_config,
            )

            X_train = prep_result["X_train"]
            X_test = prep_result["X_test"]
            y_train = prep_result["y_train"]
            y_test = prep_result["y_test"]
            feature_names = prep_result["feature_names"]
            class_names = prep_result["class_names"]
            preprocessor = prep_result["preprocessor"]
            label_encoder = prep_result["label_encoder"]

            _update_job(job_id, progress=30, message="Training model...")

            # 3. Build and train model
            model_config = job_config["model_config"].copy()
            model_config["random_seed"] = job_config.get("random_seed", 42)
            model = build_model(job_config["model_id"], job_config["task_type"], model_config)

            model, best_params, best_cv_score, tuning_trials = _run_tuning(
                model, job_config["model_id"], job_config["task_type"], model_config,
                X_train, y_train, job_config, job_id,
            )

            # Use 20% of train as validation for MLP early stopping
            val_size = 0.2
            n_val = max(1, int(len(X_train) * val_size))
            X_val = X_train[-n_val:]
            y_val = y_train[-n_val:]
            X_tr = X_train[:-n_val]
            y_tr = y_train[:-n_val]

            def progress_cb(epoch, total, loss, validation_score=None):
                pct = 30 + int(50 * epoch / total)
                row = {
                    "epoch": epoch,
                    "train_loss": float(loss),
                    "val_loss": float(validation_score) if validation_score is not None and job_config["task_type"] in ("regression", "time_series") else None,
                    "val_score": float(validation_score) if validation_score is not None else None,
                }
                current = get_job(job_id) or {}
                history = [*current.get("training_history", []), row]
                validation_text = f", validation={validation_score:.4f}" if validation_score is not None else ""
                _update_job(
                    job_id, progress=pct,
                    message=f"Training epoch {epoch}/{total}, loss={loss:.4f}{validation_text}",
                    training_history=history,
                )

            if job_config["model_id"] in {"pytorch_mlp", "tabnet", "ft_transformer"}:
                dev = getattr(model, "device", "cpu")
                _update_job(job_id, device=str(dev) if dev is not None else "cpu")
                fit_result = model.fit(X_tr, y_tr, X_val, y_val, progress_callback=progress_cb)
            else:
                fit_result = model.fit(X_tr, y_tr)

            train_duration = fit_result["train_duration_s"]
            training_history = fit_result.get("training_history", [])

            _update_job(job_id, progress=80, message="Evaluating model...")

            # 4. Evaluate
            t_infer = time.time()
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            infer_duration = time.time() - t_infer

            metrics = compute_metrics(y_test, y_pred, y_proba, job_config["task_type"], class_names)
            forecast_data = None
            if prep_result.get("is_time_series"):
                forecast_data = [
                    {"date": date, "actual": float(actual), "forecast": float(forecast)}
                    for date, actual, forecast in zip(prep_result["test_dates"], y_test, y_pred)
                ]

            _update_job(job_id, progress=88, message="Computing feature importance...")

            # 5. Feature importance
            try:
                importance_result = compute_model_importances(
                    model, X_train, X_test, y_test, feature_names, job_config["task_type"]
                )
            except Exception:
                importance_result = {"method": "unavailable", "importances": [], "fallback_used": True}

            _update_job(job_id, progress=93, message="Saving model artifact...")

            # 6. Save artifacts
            artifact_dir = ARTIFACTS_DIR / experiment_id
            artifact_dir.mkdir(parents=True, exist_ok=True)

            model_path = str(artifact_dir / "model.pkl")
            if job_config["model_id"] == "pytorch_mlp":
                model.save(str(artifact_dir / "model.pt"))
                model_path = str(artifact_dir / "model.pt")
            else:
                model.save(model_path)

            joblib.dump(preprocessor, str(artifact_dir / "preprocessor.pkl"))
            if label_encoder:
                joblib.dump(label_encoder, str(artifact_dir / "label_encoder.pkl"))

            meta = {
                "experiment_id": experiment_id,
                "model_id": job_config["model_id"],
                "task_type": job_config["task_type"],
                "target_column": job_config["target_column"],
                "feature_names": feature_names,
                "input_feature_cols": prep_result["input_feature_cols"],
                "class_names": class_names,
                "model_type": job_config["model_id"] if job_config["model_id"] in {"pytorch_mlp", "tabnet", "ft_transformer"} else "sklearn",
                "device": str(getattr(model, "device", "cpu")),
            }
            with open(str(artifact_dir / "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            # 7. Persist to DB
            exp = db.query(ExperimentRecord).filter(ExperimentRecord.id == experiment_id).first()
            if exp:
                exp.metrics = metrics
                exp.train_duration_s = round(train_duration, 3)
                exp.inference_duration_s = round(infer_duration, 4)
                exp.n_features = int(X_train.shape[1])
                exp.n_train_rows = int(len(X_train))
                exp.n_test_rows = int(len(X_test))
                exp.status = "completed"
                exp.artifact_path = str(artifact_dir)
                exp.training_history = training_history
                exp.best_params = best_params or None
                exp.best_cv_score = best_cv_score
                exp.tuning_trials = tuning_trials or None
                exp.forecast_data = forecast_data
                db.commit()

            _update_job(
                job_id,
                status="completed",
                progress=100,
                message="Training complete!",
                metrics=metrics,
                importance=importance_result,
                training_history=training_history,
                device=str(getattr(model, "device", "cpu")),
                n_features=int(X_train.shape[1]),
                n_train=int(len(X_train)),
                n_test=int(len(X_test)),
                best_params=best_params,
                best_cv_score=best_cv_score,
                tuning_trials=tuning_trials,
                forecast_data=forecast_data,
            )

        except Exception as e:
            err_msg = traceback.format_exc()
            _update_job(job_id, status="failed", progress=0, message=str(e), error=err_msg)
            exp = db.query(ExperimentRecord).filter(ExperimentRecord.id == experiment_id).first()
            if exp:
                exp.status = "failed"
                exp.error_message = str(e)
                db.commit()
        finally:
            db.close()

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "message": "Queued...",
            "experiment_id": experiment_id,
            "started_at": time.time(),
        }

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return job_id
