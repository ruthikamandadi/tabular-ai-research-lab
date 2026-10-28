"""Runtime capability reporting for the Settings page."""
import importlib.util
import inspect
import json
import logging
import threading

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.api import datasets, experiments, kaggle

router = APIRouter(prefix="/api/settings", tags=["settings"])
logger = logging.getLogger(__name__)
_CAPABILITIES_CACHE = {}
_SPEC_CACHE = {}
_CACHE_LOCK = threading.Lock()


class KaggleCredentialsRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    key: str = Field(min_length=1, max_length=256)


@router.post("/kaggle")
def save_kaggle_credentials(body: KaggleCredentialsRequest):
    from app.core.config import KAGGLE_CREDENTIALS_PATH

    KAGGLE_CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        KAGGLE_CREDENTIALS_PATH.write_text(
            json.dumps({"username": body.username, "key": body.key}, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.exception("Could not save Kaggle credentials")
        raise HTTPException(500, "Could not save Kaggle credentials to backend/storage/kaggle.json.") from exc
    return {
        "username": body.username,
        "masked_key": kaggle.mask_kaggle_key(body.key),
        "message": "Kaggle credentials saved.",
    }


@router.post("/kaggle/test")
def test_kaggle_connection():
    try:
        credentials = kaggle.get_kaggle_credentials()
        if not credentials or not all(credentials):
            return JSONResponse(
                status_code=400,
                content={"ok": False, "message": "No Kaggle credentials saved. Save credentials in Settings or configure KAGGLE_USERNAME and KAGGLE_KEY."},
            )
        with kaggle._kaggle_api(credentials) as api:
            api.dataset_list(page=1)
    except HTTPException as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"ok": False, "message": str(exc.detail)},
        )
    except Exception as exc:
        kaggle._log_kaggle_exception("Kaggle connection test failed", exc)
        status_code, message = kaggle.kaggle_connection_error(exc)
        return JSONResponse(
            status_code=status_code,
            content={"ok": False, "message": message},
        )
    except SystemExit as exc:
        kaggle._log_kaggle_exception("Kaggle authentication exited unexpectedly", exc)
        return JSONResponse(
            status_code=401,
            content={"ok": False, "message": "Kaggle authentication failed. Check the username and API key."},
        )
    return {"ok": True, "message": "Kaggle connection successful. Credentials were verified."}


def _module_available(module_name: str) -> bool:
    if module_name in _SPEC_CACHE:
        return _SPEC_CACHE[module_name]
    try:
        available = importlib.util.find_spec(module_name) is not None
    except Exception:
        logger.exception("Could not check optional package %s", module_name)
        available = False
    _SPEC_CACHE[module_name] = available
    return available


def _has_route(router_to_check, path: str) -> bool:
    for route in router_to_check.routes:
        if getattr(route, "path", None) == path:
            return True
        nested_router = getattr(route, "router", None)
        if nested_router is not None and _has_route(nested_router, path):
            return True
    return False


def _experiment_api_accepts_time_series() -> bool:
    try:
        source = inspect.getsource(experiments.create_experiment)
    except (OSError, TypeError):
        logger.exception("Could not inspect experiment task validation")
        return False
    return '"time_series"' in source


def _check_text_images_timeseries():
    text_ready = _module_available("app.preprocessing.pipeline")
    image_ready = all(_module_available(package) for package in ("PIL", "torch", "torchvision"))
    time_series_ready = _experiment_api_accepts_time_series()
    fully_available = text_ready and image_ready and time_series_ready
    note_parts = []
    if text_ready:
        note_parts.append("Text/date preprocessing is implemented.")
    if not image_ready:
        note_parts.append("Image embeddings need Pillow, torch, and torchvision.")
    if not time_series_ready:
        note_parts.append("Time-series forecasting is not accepted by the experiment API.")
    if _module_available("sentence_transformers"):
        note_parts.append("Sentence-transformers is installed for optional text embeddings.")
    return {
        "key": "text_images_timeseries",
        "label": "Text, images, and time-series",
        "available": fully_available,
        "note": " ".join(note_parts),
        "install": "Pillow torch torchvision sentence-transformers" if not image_ready else None,
    }


def _check_tuning():
    available = (
        _module_available("app.experiments.runner")
        and _module_available("sklearn")
    )
    note = "Grid and randomized search are available."
    if not _module_available("optuna"):
        note += " Optuna is optional; without it, Optuna selection falls back to randomized search."
    return {
        "key": "hyperparameter_tuning",
        "label": "Hyperparameter tuning",
        "available": available,
        "note": note if available else "Search functions or scikit-learn are unavailable.",
        "install": None if available else "scikit-learn",
    }


def _check_shap():
    importance_available = _module_available("app.explainability.importance")
    available = importance_available and _module_available("shap")
    sample_limit = 200
    note = (
        f"Current SHAP limit: at most {sample_limit} background and explanation rows; "
        "Kernel SHAP evaluates at most 50 rows."
        if importance_available
        else "SHAP explanation function is unavailable."
    )
    return {
        "key": "shap",
        "label": "SHAP explanations",
        "available": available,
        "note": note,
        "install": None if available else "shap",
    }


def _check_kaggle():
    package_ready = _module_available("kaggle")
    route_ready = (
        _has_route(kaggle.router, "/api/kaggle/search")
        and _has_route(kaggle.router, "/api/kaggle/import")
        and callable(getattr(kaggle, "search_kaggle_datasets", None))
        and callable(getattr(kaggle, "import_kaggle_dataset", None))
    )
    available = package_ready and route_ready
    note = "Kaggle import route is available." if route_ready else "The Kaggle import route is not implemented."
    if not package_ready:
        note += " The Kaggle package is not installed."
    return {
        "key": "kaggle",
        "label": "Kaggle API integration",
        "available": available,
        "note": note,
        "install": None if package_ready else "kaggle",
    }


def _check_synthetic_data():
    route_ready = _has_route(datasets.router, "/api/datasets/{dataset_id}/synthesize")
    imblearn_ready = _module_available("imblearn")
    sdv_ready = _module_available("sdv")
    available = route_ready and callable(getattr(datasets, "_make_bootstrap", None))
    note = "Bootstrap generation is available."
    note += " SMOTE is available." if imblearn_ready else " SMOTE requires imbalanced-learn."
    note += " CTGAN and Gaussian Copula are available." if sdv_ready else " CTGAN and Gaussian Copula require SDV."
    return {
        "key": "synthetic_data",
        "label": "Synthetic data (CTGAN and SMOTE)",
        "available": available,
        "note": note,
        "install": None if available else None,
    }


def _check_deep_models():
    packages_ready = all(
        _module_available(package)
        for package in ("torch", "pytorch_tabnet", "rtdl_revisiting_models")
    )
    registry_ready = _module_available("app.models.model_factory")
    available = packages_ready and registry_ready
    return {
        "key": "deep_models",
        "label": "TabNet and FT-Transformer",
        "available": available,
        "note": (
            "Both optional models are registered and their packages are discoverable."
            if available
            else "Both optional models require discoverable PyTorch, pytorch-tabnet, and rtdl-revisiting-models packages."
        ),
        "install": None if available else "torch pytorch-tabnet rtdl-revisiting-models",
    }


_CAPABILITY_CHECKS = (
    _check_text_images_timeseries,
    _check_tuning,
    _check_shap,
    _check_kaggle,
    _check_synthetic_data,
    _check_deep_models,
)


def _calculate_capabilities():
    capabilities = []
    for check in _CAPABILITY_CHECKS:
        key = check.__name__.removeprefix("_check_")
        try:
            capability = check()
            _CAPABILITIES_CACHE[key] = capability
        except Exception:
            logger.exception("Capability check failed: %s", key)
            capability = {
                "key": key,
                "label": key.replace("_", " ").title(),
                "available": False,
                "note": "Capability check failed; see backend logs for details.",
                "install": None,
            }
            _CAPABILITIES_CACHE[key] = capability
        capabilities.append(capability)
    return capabilities


@router.get("/capabilities")
def get_capabilities(refresh: bool = False):
    with _CACHE_LOCK:
        if refresh or not _CAPABILITIES_CACHE:
            _CAPABILITIES_CACHE.clear()
            _SPEC_CACHE.clear()
            return _calculate_capabilities()
        return list(_CAPABILITIES_CACHE.values())
