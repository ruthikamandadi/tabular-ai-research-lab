"""
Evaluation metrics computation for classification and regression.
"""
import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, log_loss,
    mean_absolute_error, mean_squared_error, r2_score,
)
import warnings
warnings.filterwarnings("ignore")


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    task_type: str,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    if task_type == "binary_classification":
        metrics["accuracy"] = round(float(accuracy_score(y_true, y_pred)), 4)
        metrics["precision"] = round(float(precision_score(y_true, y_pred, zero_division=0)), 4)
        metrics["recall"] = round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
        metrics["f1"] = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)

        if y_proba is not None:
            try:
                metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_proba[:, 1])), 4)
            except Exception:
                metrics["roc_auc"] = None
            try:
                metrics["log_loss"] = round(float(log_loss(y_true, y_proba)), 4)
            except Exception:
                metrics["log_loss"] = None
        else:
            metrics["roc_auc"] = None
            metrics["log_loss"] = None

        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        metrics["class_names"] = class_names or ["0", "1"]

    elif task_type == "multiclass_classification":
        metrics["accuracy"] = round(float(accuracy_score(y_true, y_pred)), 4)
        metrics["f1_macro"] = round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)
        metrics["f1_weighted"] = round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4)
        metrics["precision_macro"] = round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4)
        metrics["recall_macro"] = round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4)

        if y_proba is not None:
            try:
                n_classes = y_proba.shape[1]
                classes_present = np.unique(y_true)
                if len(classes_present) == n_classes:
                    metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")), 4)
                else:
                    metrics["roc_auc"] = None
            except Exception:
                metrics["roc_auc"] = None
            try:
                metrics["log_loss"] = round(float(log_loss(y_true, y_proba)), 4)
            except Exception:
                metrics["log_loss"] = None
        else:
            metrics["roc_auc"] = None
            metrics["log_loss"] = None

        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        n_classes = int(y_true.max()) + 1
        metrics["class_names"] = class_names or [str(i) for i in range(n_classes)]

    return metrics


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 6),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 6),
        "r2": round(float(r2_score(y_true, y_pred)), 6),
    }


def compute_time_series_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    nonzero = actual != 0
    mape = float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else None
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 6),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 6),
        "mape": round(mape, 6) if mape is not None else None,
        "r2": round(float(r2_score(actual, predicted)), 6),
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    task_type: str,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if task_type == "time_series":
        return compute_time_series_metrics(y_true, y_pred)
    if task_type == "regression":
        return compute_regression_metrics(y_true, y_pred)
    else:
        return compute_classification_metrics(y_true, y_pred, y_proba, task_type, class_names)
