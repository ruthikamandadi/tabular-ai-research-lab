"""
Preprocessing pipeline builder.
CRITICAL: Fit preprocessing ONLY on training data, apply to test data.
"""
import io
import logging
import hashlib
import importlib.util
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
from urllib.parse import urlparse
from pathlib import Path
from app.core.datetime_utils import parse_datetime
from app.core.config import STORAGE_DIR

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


COLUMN_TYPES = ("numeric", "categorical", "text", "date", "image_path", "identifier")


def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    detected = {}
    sample = df.head(min(len(df), 5000))
    for column in df.columns:
        series = sample[column].dropna()
        name = str(column).lower()
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            detected[column] = "date"
            continue
        if not len(series):
            detected[column] = "categorical"
            continue
        if pd.api.types.is_numeric_dtype(series):
            detected[column] = "identifier" if "id" in name or series.nunique() / len(series) > 0.95 else "numeric"
            continue

        values = series.astype(str)
        image_ratio = values.map(_looks_like_image_path).mean()
        if image_ratio >= 0.7:
            detected[column] = "image_path"
            continue

        date_ratio = parse_datetime(values).notna().mean()
        if date_ratio >= 0.85 or any(token in name for token in ("date", "time", "timestamp")):
            detected[column] = "date"
            continue

        avg_length = values.str.len().mean()
        cardinality = values.nunique() / len(values)
        if avg_length > 30 or (len(values) >= 20 and values.nunique() > max(50, len(values) * 0.2)):
            detected[column] = "text"
        elif "id" in name or cardinality > 0.95:
            detected[column] = "identifier"
        else:
            detected[column] = "categorical"
    return detected


def _looks_like_image_path(value: str) -> bool:
    parsed = urlparse(value)
    path = parsed.path if parsed.scheme else value
    return Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def _flatten_text(values):
    return pd.Series(np.asarray(values).reshape(-1)).fillna("").astype(str).to_numpy()


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = pd.DataFrame(X)
        output = []
        for column_index in range(frame.shape[1]):
            parsed = parse_datetime(frame.iloc[:, column_index])
            output.extend([
                parsed.dt.year,
                parsed.dt.month,
                parsed.dt.day,
                parsed.dt.weekday,
                parsed.dt.hour,
                (parsed.dt.weekday >= 5).astype(float),
            ])
        return np.column_stack([values.fillna(0).to_numpy() for values in output]).astype(float)

    def get_feature_names_out(self, input_features=None):
        names = []
        for column in ([] if input_features is None else input_features):
            names.extend([f"{column}_{part}" for part in ("year", "month", "day", "weekday", "hour", "is_weekend")])
        return np.asarray(names, dtype=object)


class SentenceEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None

    def fit(self, X, y=None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ValueError("Sentence embeddings require the optional sentence-transformers package.") from exc
        self.model = SentenceTransformer(self.model_name)
        return self

    def transform(self, X):
        return self.model.encode(_flatten_text(X), show_progress_bar=False)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"text_embedding_{index}" for index in range(384)], dtype=object)


class ImageEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, batch_size: int = 16, max_images: int = 2000):
        self.batch_size = batch_size
        self.max_images = max_images
        self.device = "cpu"
        self.model = None
        self.transformer = None
        self._image_class = None
        self.sampled = False
        self.sampling_message = ""

    def fit(self, X, y=None):
        return self

    def _init_model(self):
        if self.model is not None:
            return
        try:
            import torch
            from torchvision.models import ResNet18_Weights, resnet18
            from torchvision.transforms import Compose, Normalize, Resize, ToTensor
            from PIL import Image
        except ImportError as exc:
            raise ValueError(
                "Image embeddings require 'Pillow' and 'torchvision'. Run: pip install Pillow torchvision"
            ) from exc

        self.device = "cpu"
        try:
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception:
            self.model = resnet18(weights=None)
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(self.device)
        self.transformer = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._image_class = Image

    def transform(self, X):
        self._init_model()
        import torch
        import urllib.request

        cache_dir = STORAGE_DIR / "image_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        items = _flatten_text(X)
        total_items = len(items)
        if total_items == 0:
            return np.zeros((0, 512), dtype=float)

        if total_items > self.max_images:
            self.sampled = True
            self.sampling_message = (
                f"Image input contains {total_items} items, capped at {self.max_images} per run. "
                "Sampling was applied."
            )
            logger.warning(self.sampling_message)
            sample_indices = set(range(self.max_images))
        else:
            sample_indices = set(range(total_items))

        embeddings = np.zeros((total_items, 512), dtype=float)
        indices_to_compute = []
        hashes_to_compute = []
        raw_items_to_compute = []

        # Check disk cache first
        for idx in range(total_items):
            if idx not in sample_indices:
                continue
            item = items[idx]
            h = hashlib.sha256(item.encode("utf-8")).hexdigest()
            cache_file = cache_dir / f"{h}.npy"
            if cache_file.exists():
                try:
                    embeddings[idx] = np.load(cache_file)
                    continue
                except Exception:
                    pass
            indices_to_compute.append(idx)
            hashes_to_compute.append(h)
            raw_items_to_compute.append(item)

        # Batch compute missing embeddings (batch size 16, CPU)
        for b_start in range(0, len(indices_to_compute), self.batch_size):
            b_end = min(b_start + self.batch_size, len(indices_to_compute))
            b_indices = indices_to_compute[b_start:b_end]
            b_hashes = hashes_to_compute[b_start:b_end]
            b_items = raw_items_to_compute[b_start:b_end]

            tensors = []
            for item in b_items:
                try:
                    parsed = urlparse(item)
                    if parsed.scheme in ("http", "https"):
                        with urllib.request.urlopen(item, timeout=10) as response:
                            content = response.read(10 * 1024 * 1024 + 1)
                        if len(content) > 10 * 1024 * 1024:
                            raise ValueError("Image URL exceeds 10 MB limit.")
                        img = self._image_class.open(io.BytesIO(content)).convert("RGB")
                    else:
                        img = self._image_class.open(item).convert("RGB")
                except Exception:
                    # Fallback to gray neutral image
                    img = self._image_class.new("RGB", (224, 224), color=(128, 128, 128))
                t = self.transformer(img)
                tensors.append(t)

            batch_tensor = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                out = self.model(batch_tensor).cpu().numpy()
                if out.ndim > 2:
                    out = out.reshape(out.shape[0], -1)

            for i, idx in enumerate(b_indices):
                vec = out[i]
                embeddings[idx] = vec
                cache_file = cache_dir / f"{b_hashes[i]}.npy"
                try:
                    np.save(cache_file, vec)
                except Exception:
                    pass

        # If sampled, fill unsampled rows with the mean embedding of sampled ones
        if self.sampled and len(sample_indices) > 0:
            sampled_mask = np.array([i in sample_indices for i in range(total_items)])
            mean_emb = embeddings[sampled_mask].mean(axis=0)
            embeddings[~sampled_mask] = mean_emb

        return embeddings

    def get_feature_names_out(self, input_features=None):
        prefix = input_features[0] if input_features and len(input_features) > 0 else "image"
        return np.asarray([f"{prefix}_emb_{index}" for index in range(512)], dtype=object)


def build_preprocessing_pipeline(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build and return a fitted preprocessing pipeline and split data.
    
    config keys:
      - numeric_imputer: "median" | "mean"
      - numeric_scaler: "standard" | "robust" | "none"
      - categorical_imputer: "most_frequent" | "constant"
      - categorical_encoding: "onehot" | "ordinal"
      - exclude_columns: list of column names to drop
      - identifier_columns: list of likely ID cols (will warn, optionally drop)
      - drop_identifiers: bool
      - test_size: float (0.1 - 0.4)
      - random_seed: int
      - handle_constant: bool (drop constant cols)
    """
    random_seed = config.get("random_seed", 42)
    test_size = config.get("test_size", 0.2)
    exclude_cols = set(config.get("exclude_columns", []))
    drop_identifiers = config.get("drop_identifiers", True)
    identifier_cols = set(config.get("identifier_columns", []))
    handle_constant = config.get("handle_constant", True)
    overrides = config.get("column_types", {})
    detected_types = detect_column_types(df)
    detected_types.update(overrides)
    invalid_types = set(detected_types.values()) - set(COLUMN_TYPES)
    if invalid_types:
        raise ValueError(f"Unsupported column type override(s): {sorted(invalid_types)}")

    warnings_list = []
    calendar_cols = []
    date_col = config.get("time_series_date_column")
    is_time_series = task_type in ("time_series", "time_series_forecasting")
    if is_time_series:
        if not date_col or date_col not in df.columns:
            raise ValueError("Choose a valid date column for time-series forecasting.")
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            raise ValueError("Time-series forecasting target must be numeric.")
        ordered_dates = parse_datetime(df[date_col])
        if ordered_dates.isna().any():
            raise ValueError(f"Date column '{date_col}' contains invalid or missing dates.")
        if not ordered_dates.is_monotonic_increasing:
            raise ValueError(f"Date column '{date_col}' must be sorted in ascending chronological order.")
        target_series = pd.to_numeric(df[target_col], errors="coerce")
        if target_series.isna().any():
            raise ValueError("Time-series target contains missing or non-numeric values.")
        lags = config.get("time_series_lags", [1, 7, 14])
        if not lags or any(not isinstance(lag, int) or lag < 1 for lag in lags):
            raise ValueError("Time-series lags must be positive integers.")
        for lag in lags:
            df[f"{target_col}_lag_{lag}"] = target_series.shift(lag)
        for window in config.get("time_series_rolling_windows", [3, 7]):
            if not isinstance(window, int) or window < 2:
                raise ValueError("Rolling windows must be integers greater than 1.")
            shifted = target_series.shift(1)
            df[f"{target_col}_rolling_mean_{window}"] = shifted.rolling(window).mean()
            df[f"{target_col}_rolling_std_{window}"] = shifted.rolling(window).std()

        # Calendar features
        calendar_cols = [f"{date_col}_month", f"{date_col}_weekday", f"{date_col}_is_weekend"]
        df[f"{date_col}_month"] = ordered_dates.dt.month.astype(float)
        df[f"{date_col}_weekday"] = ordered_dates.dt.weekday.astype(float)
        df[f"{date_col}_is_weekend"] = (ordered_dates.dt.weekday >= 5).astype(float)
        if "month" not in df.columns:
            df["month"] = df[f"{date_col}_month"]
            calendar_cols.append("month")
        if "weekday" not in df.columns:
            df["weekday"] = df[f"{date_col}_weekday"]
            calendar_cols.append("weekday")
        if "is_weekend" not in df.columns:
            df["is_weekend"] = df[f"{date_col}_is_weekend"]
            calendar_cols.append("is_weekend")

        derived_cols = [c for c in df.columns if c.startswith(f"{target_col}_lag_") or c.startswith(f"{target_col}_rolling_")]
        df = df.dropna(subset=derived_cols).reset_index(drop=True)

    if drop_identifiers:
        identifier_cols |= {col for col, kind in detected_types.items() if kind == "identifier"}
        exclude_cols |= identifier_cols

    skipped_image_cols = []
    if not _image_dependencies_available():
        missing_pkgs = []
        if importlib.util.find_spec("PIL") is None:
            missing_pkgs.append("Pillow")
        if importlib.util.find_spec("torch") is None:
            missing_pkgs.append("torch")
        if importlib.util.find_spec("torchvision") is None:
            missing_pkgs.append("torchvision")
        if not missing_pkgs:
            missing_pkgs = ["Pillow", "torchvision"]

        for col, kind in detected_types.items():
            if kind == "image_path":
                skipped_image_cols.append(col)
                exclude_cols.add(col)
                msg = (
                    f"Skipped image column '{col}': image embeddings require {' and '.join(missing_pkgs)}. "
                    f"Install with: pip install {' '.join(missing_pkgs)}"
                )
                warnings_list.append(msg)
                logger.warning(msg)
    else:
        exclude_cols |= {col for col, kind in detected_types.items() if kind == "image_path" and not _image_dependencies_available()}

    # Drop excluded columns
    feature_cols = [c for c in df.columns if c != target_col and c not in exclude_cols]
    if is_time_series and date_col in feature_cols:
        feature_cols.remove(date_col)

    # Drop constant columns
    dropped_constants = []
    if handle_constant:
        for c in list(feature_cols):
            if df[c].nunique(dropna=False) <= 1:
                feature_cols.remove(c)
                dropped_constants.append(c)

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Encode target for classification
    label_encoder = None
    class_names = None
    if task_type in ("binary_classification", "multiclass_classification"):
        label_encoder = LabelEncoder()
        y_encoded = pd.Series(label_encoder.fit_transform(y.astype(str)), index=y.index)
        class_names = [str(c) for c in label_encoder.classes_]
    else:
        y_encoded = pd.to_numeric(y, errors="coerce")
        if y_encoded.isna().any():
            raise ValueError(f"Target column '{target_col}' contains non-numeric values for regression.")

    # Stratified split for classification
    try:
        if task_type in ("binary_classification", "multiclass_classification"):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=random_seed, stratify=y_encoded
            )
        elif is_time_series:
            split_at = int(len(X) * (1 - test_size))
            if split_at < 2 or split_at >= len(X):
                raise ValueError("Chronological split leaves too few rows for training or testing.")
            X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
            y_train, y_test = y_encoded.iloc[:split_at], y_encoded.iloc[split_at:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=random_seed
            )
    except ValueError as e:
        raise ValueError(f"Data split failed: {e}. Try a larger dataset or different test size.")

    # Identify column types in training data
    numeric_features = [c for c in feature_cols if detected_types.get(c, "numeric" if pd.api.types.is_numeric_dtype(X_train[c]) else "categorical") == "numeric"]
    categorical_features = [c for c in feature_cols if detected_types.get(c) == "categorical"]
    text_features = [c for c in feature_cols if detected_types.get(c) == "text"]
    date_features = [c for c in feature_cols if detected_types.get(c) == "date"]
    image_features = [c for c in feature_cols if detected_types.get(c) == "image_path" and _image_dependencies_available()]
    if is_time_series:
        derived_lags = [c for c in feature_cols if c.startswith(f"{target_col}_lag_") or c.startswith(f"{target_col}_rolling_") or c in calendar_cols]
        numeric_features = list(dict.fromkeys(numeric_features + derived_lags))
        date_features = [c for c in date_features if c != date_col]
        categorical_features = [c for c in categorical_features if c != date_col]

    # Numeric pipeline
    numeric_imputer_strategy = config.get("numeric_imputer", "median")
    scaler_type = config.get("numeric_scaler", "standard")

    numeric_steps = [("imputer", SimpleImputer(strategy=numeric_imputer_strategy))]
    if scaler_type == "standard":
        numeric_steps.append(("scaler", StandardScaler()))
    elif scaler_type == "robust":
        numeric_steps.append(("scaler", RobustScaler()))
    # "none" → no scaler step

    numeric_pipeline = Pipeline(steps=numeric_steps)

    # Categorical pipeline
    cat_imputer_strategy = config.get("categorical_imputer", "most_frequent")
    cat_encoding = config.get("categorical_encoding", "onehot")

    if cat_encoding == "onehot":
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy=cat_imputer_strategy, fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50)),
        ])
    else:
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy=cat_imputer_strategy, fill_value="missing")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", cat_pipeline, categorical_features))
    if text_features:
        text_steps = [("flatten", FunctionTransformer(_flatten_text, validate=False))]
        text_encoding = config.get("text_encoding", "tfidf")
        if text_encoding == "tfidf":
            text_steps.append(("vectorizer", TfidfVectorizer(max_features=int(config.get("text_max_features", 500)))))
        elif text_encoding == "embedding":
            text_steps.append(("vectorizer", SentenceEmbeddingTransformer()))
        else:
            raise ValueError("Text encoding must be 'tfidf' or 'embedding'.")
        transformers.append(("text", Pipeline(steps=text_steps), text_features))
    if date_features:
        transformers.append(("date", DateFeatureExtractor(), date_features))
    if image_features:
        transformers.append(("image", ImageEmbeddingTransformer(batch_size=16, max_images=2000), image_features))

    if not transformers:
        if skipped_image_cols:
            raise ValueError(
                f"No features available: image column(s) {skipped_image_cols} were skipped because "
                "image embeddings require 'Pillow' and 'torchvision'. Run: pip install Pillow torchvision"
            )
        raise ValueError("No features available after preprocessing configuration.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)

    # FIT only on training data
    try:
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
    except Exception as exc:
        raise ValueError(f"Could not preprocess columns: {exc}") from exc

    for name, trans, _ in transformers:
        if name == "image" and getattr(trans, "sampled", False) and getattr(trans, "sampling_message", ""):
            warnings_list.append(trans.sampling_message)

    # Get feature names after transformation
    try:
        feature_names_out = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names_out = [f"feature_{i}" for i in range(X_train_transformed.shape[1])]

    return {
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "class_names": class_names,
        "X_train": X_train_transformed,
        "X_test": X_test_transformed,
        "y_train": y_train.values,
        "y_test": y_test.values,
        "feature_names": feature_names_out,
        "input_feature_cols": feature_cols,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "n_features": X_train_transformed.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "dropped_constants": dropped_constants,
        "excluded_cols": list(exclude_cols),
        "config": config,
        "test_dates": df.loc[X_test.index, date_col].astype(str).tolist() if is_time_series else None,
        "is_time_series": is_time_series,
        "warnings": warnings_list,
        "skipped_image_columns": skipped_image_cols,
    }


def _image_dependencies_available():
    for mod in ("PIL", "torch", "torchvision"):
        if importlib.util.find_spec(mod) is None:
            return False
    try:
        importlib.import_module("PIL")
        importlib.import_module("torch")
        importlib.import_module("torchvision")
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def apply_pipeline_to_new_data(
    df: pd.DataFrame,
    preprocessor,
    input_feature_cols: List[str],
    label_encoder=None,
) -> Tuple[np.ndarray, List[str]]:
    """Apply a fitted preprocessor to new inference data."""
    missing_cols = [c for c in input_feature_cols if c not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    X = df[input_feature_cols].copy()
    X_transformed = preprocessor.transform(X)
    return X_transformed, missing_cols
