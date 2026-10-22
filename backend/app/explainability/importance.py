"""
Explainability: SHAP + Permutation Importance fallback.
"""
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings("ignore")


def compute_permutation_importance(
    model_predict_fn: Callable,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    task_type: str,
    n_repeats: int = 5,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """
    Model-agnostic permutation importance on held-out test data.
    Works for any model with a predict function.
    """
    from sklearn.metrics import accuracy_score, mean_squared_error

    if task_type in ("regression", "time_series"):
        def scorer(y_true, y_pred):
            return -float(np.sqrt(mean_squared_error(y_true, y_pred)))
        higher_is_better = False
    else:
        def scorer(y_true, y_pred):
            return float(accuracy_score(y_true, y_pred))
        higher_is_better = True

    base_score = scorer(y_test, model_predict_fn(X_test))
    importances = []

    rng = np.random.RandomState(random_state)
    for i in range(X_test.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            perm_score = scorer(y_test, model_predict_fn(X_perm))
            scores.append(base_score - perm_score)
        mean_imp = float(np.mean(scores))
        std_imp = float(np.std(scores))
        importances.append({
            "feature": feature_names[i] if i < len(feature_names) else f"feature_{i}",
            "importance": round(mean_imp, 6),
            "std": round(std_imp, 6),
        })

    importances.sort(key=lambda x: abs(x["importance"]), reverse=True)
    return importances


def compute_shap_importance(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    task_type: str,
    max_samples: int = 200,
) -> Optional[List[Dict[str, Any]]]:
    """
    SHAP-based global importance. Falls back gracefully.
    """
    try:
        import shap

        # Subsample for speed
        if len(X_train) > max_samples:
            idx = np.random.choice(len(X_train), max_samples, replace=False)
            X_bg = X_train[idx]
        else:
            X_bg = X_train

        if len(X_test) > max_samples:
            idx = np.random.choice(len(X_test), max_samples, replace=False)
            X_exp = X_test[idx]
        else:
            X_exp = X_test

        # Choose appropriate explainer
        sklearn_estimator = getattr(model, "estimator", None)
        if sklearn_estimator is not None:
            try:
                explainer = shap.TreeExplainer(sklearn_estimator)
                shap_values = explainer.shap_values(X_exp)
            except Exception:
                explainer = shap.KernelExplainer(
                    sklearn_estimator.predict if task_type in ("regression", "time_series")
                    else sklearn_estimator.predict_proba,
                    shap.sample(X_bg, min(50, len(X_bg))),
                )
                shap_values = explainer.shap_values(X_exp[:50])
        else:
            # PyTorch MLP - use permutation fallback
            return None

        if isinstance(shap_values, list):
            # Multiclass: average absolute across classes
            sv = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            sv = np.abs(shap_values)

        if sv.ndim > 1:
            sv = sv.mean(axis=0)

        result = []
        for i, fname in enumerate(feature_names):
            if i < len(sv):
                result.append({"feature": fname, "importance": round(float(sv[i]), 6), "method": "SHAP"})

        result.sort(key=lambda x: x["importance"], reverse=True)
        return result

    except Exception as e:
        return None  # Will trigger fallback


def compute_model_importances(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    task_type: str,
) -> Dict[str, Any]:
    """
    Compute global feature importances using best available method.
    Returns importances dict with method used and top features.
    """
    result: Dict[str, Any] = {"method": None, "importances": [], "fallback_used": False}

    # 1. Try model built-in importances (Random Forest, GBM, XGBoost)
    built_in = None
    sklearn_estimator = getattr(model, "estimator", None)
    if sklearn_estimator is not None and hasattr(sklearn_estimator, "feature_importances_"):
        imp = sklearn_estimator.feature_importances_
        built_in = [
            {"feature": feature_names[i] if i < len(feature_names) else f"feature_{i}",
             "importance": round(float(imp[i]), 6), "method": "model_built_in"}
            for i in range(len(imp))
        ]
        built_in.sort(key=lambda x: x["importance"], reverse=True)

    # 2. Try SHAP
    shap_result = compute_shap_importance(model, X_train, X_test, feature_names, task_type)

    if shap_result:
        result["method"] = "SHAP"
        result["importances"] = shap_result
    elif built_in:
        result["method"] = "model_built_in"
        result["importances"] = built_in
    else:
        # 3. Fallback: permutation importance
        result["fallback_used"] = True
        result["method"] = "permutation"
        perm = compute_permutation_importance(
            model.predict, X_test, y_test, feature_names, task_type
        )
        result["importances"] = perm

    # Keep top 30
    result["importances"] = result["importances"][:30]
    return result


def compute_local_explanation(
    model,
    X_instance: np.ndarray,
    X_train: np.ndarray,
    feature_names: List[str],
    task_type: str,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Local explanation for a single instance using SHAP or gradient-based fallback.
    """
    try:
        import shap
        sklearn_estimator = getattr(model, "estimator", None)
        if sklearn_estimator is not None:
            try:
                explainer = shap.TreeExplainer(sklearn_estimator)
                sv = explainer.shap_values(X_instance.reshape(1, -1))
            except Exception:
                bg = shap.sample(X_train, min(50, len(X_train)))
                explainer = shap.KernelExplainer(
                    sklearn_estimator.predict if task_type in ("regression", "time_series")
                    else sklearn_estimator.predict_proba,
                    bg,
                )
                sv = explainer.shap_values(X_instance.reshape(1, -1))

            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]  # binary: class 1
            sv = sv.flatten()

            contributions = [
                {"feature": feature_names[i] if i < len(feature_names) else f"f_{i}",
                 "value": round(float(X_instance[i]), 4),
                 "contribution": round(float(sv[i]), 6)}
                for i in range(min(len(sv), len(feature_names)))
            ]
            contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
            return {"method": "SHAP", "contributions": contributions[:20]}
    except Exception:
        pass

    # Fallback: linear approximation using coefficients if available
    return {"method": "unavailable", "contributions": []}
