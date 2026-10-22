"""
Preprocessing pipeline builder.
CRITICAL: Fit preprocessing ONLY on training data, apply to test data.
"""
import io
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
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
    def __init__(self):
        self.model = None
        self.transformer = None
        self.device = "cpu"

    def fit(self, X, y=None):
        try:
            import torch
            from torchvision.models import ResNet18_Weights, resnet18
            from torchvision.transforms import Compose, Normalize, Resize, ToTensor
            from PIL import Image
        except ImportError as exc:
            raise ValueError("Image embeddings need the optional Pillow, torch, and torchvision packages.") from exc
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(self.device)
        self.transformer = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._image_class = Image
        return self

    def transform(self, X):
        import torch
        import urllib.request

        vectors = []
        for value in _flatten_text(X):
            parsed = urlparse(value)
            if parsed.scheme in ("http", "https"):
                with urllib.request.urlopen(value, timeout=10) as response:
                    content = response.read(10 * 1024 * 1024 + 1)
                if len(content) > 10 * 1024 * 1024:
                    raise ValueError("Image URL exceeds the 10 MB per-image limit.")
                image = self._image_class.open(io.BytesIO(content)).convert("RGB")
            else:
                image = self._image_class.open(value).convert("RGB")
            tensor = self.transformer(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                vectors.append(self.model(tensor).cpu().numpy()[0])
        return np.asarray(vectors)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"image_embedding_{index}" for index in range(512)], dtype=object)


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
        for window in config.get("time_series_rolling_windows", [7]):
            if not isinstance(window, int) or window < 2:
                raise ValueError("Rolling windows must be integers greater than 1.")
            shifted = target_series.shift(1)
            df[f"{target_col}_rolling_mean_{window}"] = shifted.rolling(window).mean()
            df[f"{target_col}_rolling_std_{window}"] = shifted.rolling(window).std()
        df = df.dropna(subset=[c for c in df.columns if c.startswith(f"{target_col}_lag_") or c.startswith(f"{target_col}_rolling_")]).reset_index(drop=True)

    if drop_identifiers:
        identifier_cols |= {col for col, kind in detected_types.items() if kind == "identifier"}
        exclude_cols |= identifier_cols
    exclude_cols |= {col for col, kind in detected_types.items() if kind == "image_path" and not _image_dependencies_available()}

    # Drop excluded columns
    feature_cols = [c for c in df.columns if c != target_col and c not in exclude_cols]
    if is_time_series and date_col not in feature_cols:
        feature_cols.append(date_col)

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
        derived_lags = [c for c in feature_cols if c.startswith(f"{target_col}_lag_") or c.startswith(f"{target_col}_rolling_")]
        numeric_features = list(dict.fromkeys(numeric_features + derived_lags))
        date_features = list(dict.fromkeys(date_features + [date_col]))
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
        transformers.append(("image", ImageEmbeddingTransformer(), image_features))

    if not transformers:
        raise ValueError("No features available after preprocessing configuration.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)

    # FIT only on training data
    try:
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
    except Exception as exc:
        raise ValueError(f"Could not preprocess columns: {exc}") from exc

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
    }


def _image_dependencies_available():
    try:
        import PIL
        import torchvision
        import torch
        return True
    except ImportError:
        return False


def apply_pipeline_to_new_data(
    df: pd.DataFrame,
    preprocessor,
    input_feature_cols: List[str],
    label_encoder=None,
) -> Tuple[np.ndarray, List[str]]:
    """Apply a fitted preprocessor to new inference data."""
    missing_cols = [c for c in input_feature_cols if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in input_feature_cols]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    X = df[input_feature_cols].copy()
    X_transformed = preprocessor.transform(X)
    return X_transformed, missing_cols
