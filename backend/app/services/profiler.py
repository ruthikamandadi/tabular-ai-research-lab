"""
Data profiling engine - computes statistics from uploaded datasets.
All statistics are computed from actual data, never hard-coded.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
import warnings
from app.core.datetime_utils import parse_datetime

warnings.filterwarnings("ignore")


def detect_column_type(series: pd.Series) -> str:
    """Classify a column into: numeric, integer, categorical, boolean, datetime, text, identifier."""
    if series.dtype == bool or (series.dropna().isin([0, 1, True, False]).all() and series.nunique() <= 2):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_float_dtype(series):
        return "numeric"
    if pd.api.types.is_integer_dtype(series):
        n_unique = series.nunique()
        if n_unique <= 20:
            return "categorical_int"
        return "integer"
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        n_unique = series.nunique()
        n_total = len(series.dropna())
        if n_total == 0:
            return "categorical"
        ratio = n_unique / n_total
        # Try datetime parsing
        try:
            sample = series.dropna().head(50)
            parse_datetime(sample)
            return "datetime"
        except Exception:
            pass
        if ratio > 0.9 and n_unique > 50:
            return "text_or_id"
        if n_unique <= 50:
            return "categorical"
        return "text"
    return "other"


def compute_outlier_summary(series: pd.Series) -> Dict[str, Any]:
    """IQR-based outlier detection."""
    clean = series.dropna()
    if len(clean) < 4:
        return {"method": "IQR", "outlier_count": 0, "outlier_pct": 0.0}
    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = clean[(clean < lower) | (clean > upper)]
    return {
        "method": "IQR",
        "lower_fence": round(lower, 4),
        "upper_fence": round(upper, 4),
        "outlier_count": int(len(outliers)),
        "outlier_pct": round(100 * len(outliers) / len(clean), 2),
    }


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Full dataset profile computation."""
    n_rows, n_cols = df.shape
    duplicate_rows = int(df.duplicated().sum())

    columns_profile = []
    for col in df.columns:
        series = df[col]
        col_type = detect_column_type(series)
        missing = int(series.isna().sum())
        missing_pct = round(100 * missing / n_rows, 2) if n_rows > 0 else 0.0
        n_unique = int(series.nunique(dropna=True))

        col_info: Dict[str, Any] = {
            "name": col,
            "dtype": str(series.dtype),
            "inferred_type": col_type,
            "missing_count": missing,
            "missing_pct": missing_pct,
            "unique_count": n_unique,
            "is_constant": n_unique <= 1,
            "is_likely_id": col_type in ("text_or_id",) or (
                col_type in ("integer", "categorical_int") and n_unique == n_rows
            ),
        }

        if col_type in ("numeric", "integer", "categorical_int", "boolean"):
            clean = pd.to_numeric(series, errors="coerce").dropna()
            if len(clean) > 0:
                col_info["stats"] = {
                    "mean": round(float(clean.mean()), 6),
                    "median": round(float(clean.median()), 6),
                    "std": round(float(clean.std()), 6),
                    "min": round(float(clean.min()), 6),
                    "max": round(float(clean.max()), 6),
                    "q25": round(float(clean.quantile(0.25)), 6),
                    "q75": round(float(clean.quantile(0.75)), 6),
                }
                if col_type == "numeric":
                    col_info["outliers"] = compute_outlier_summary(clean)

        if col_type in ("categorical", "categorical_int", "boolean"):
            freq = series.value_counts(dropna=False).head(20)
            col_info["top_values"] = [
                {"value": str(k), "count": int(v)}
                for k, v in freq.items()
            ]

        # Histogram bins for numeric
        if col_type in ("numeric", "integer"):
            clean = pd.to_numeric(series, errors="coerce").dropna()
            if len(clean) >= 2:
                try:
                    counts, edges = np.histogram(clean, bins=min(30, max(5, n_unique)))
                    col_info["histogram"] = {
                        "counts": counts.tolist(),
                        "edges": [round(float(e), 6) for e in edges.tolist()],
                    }
                except Exception:
                    pass

        columns_profile.append(col_info)

    # Numeric correlation matrix
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    correlation = {}
    if len(numeric_cols) >= 2:
        corr_df = df[numeric_cols].corr()
        correlation = {
            "columns": numeric_cols,
            "matrix": [[round(float(v), 4) if not np.isnan(v) else 0 for v in row]
                       for row in corr_df.values],
        }

    # Missing value summary
    missing_summary = [
        {"column": col, "missing": int(df[col].isna().sum()),
         "pct": round(100 * df[col].isna().sum() / n_rows, 2)}
        for col in df.columns if df[col].isna().sum() > 0
    ]
    missing_summary.sort(key=lambda x: x["missing"], reverse=True)

    # Warnings
    warnings_list = []
    for c in columns_profile:
        if c["is_constant"]:
            warnings_list.append({"type": "constant_column", "column": c["name"],
                                   "message": f"Column '{c['name']}' is constant and provides no predictive signal."})
        if c["is_likely_id"]:
            warnings_list.append({"type": "likely_identifier", "column": c["name"],
                                   "message": f"Column '{c['name']}' looks like an identifier. Consider excluding it."})
        if c["missing_pct"] > 50:
            warnings_list.append({"type": "high_missing", "column": c["name"],
                                   "message": f"Column '{c['name']}' has {c['missing_pct']}% missing values."})
        if c["inferred_type"] == "text" and c["unique_count"] > 100:
            warnings_list.append({"type": "high_cardinality_text", "column": c["name"],
                                   "message": f"Column '{c['name']}' is free text with {c['unique_count']} unique values. Not suitable for automatic tabular encoding."})

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "duplicate_rows": duplicate_rows,
        "columns": columns_profile,
        "correlation": correlation,
        "missing_summary": missing_summary,
        "warnings": warnings_list,
        "numeric_columns": numeric_cols,
    }


def suggest_task_type(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """Heuristic task type suggestion based on target column properties."""
    series = df[target_col].dropna()
    n_unique = series.nunique()
    dtype = str(series.dtype)

    if pd.api.types.is_float_dtype(series) and n_unique > 20:
        suggestion = "regression"
        rationale = f"Target is float with {n_unique} unique values → likely regression."
    elif n_unique == 2:
        suggestion = "binary_classification"
        rationale = "Target has exactly 2 unique values → binary classification."
    elif pd.api.types.is_integer_dtype(series) or pd.api.types.is_object_dtype(series):
        if n_unique <= 20:
            suggestion = "multiclass_classification"
            rationale = f"Target has {n_unique} unique values → multiclass classification."
        elif pd.api.types.is_integer_dtype(series) and n_unique > 20:
            suggestion = "regression"
            rationale = f"Target is integer with {n_unique} unique values → likely regression."
        else:
            suggestion = "multiclass_classification"
            rationale = f"Target has {n_unique} unique values → multiclass classification (high cardinality—verify)."
    else:
        suggestion = "regression"
        rationale = f"Defaulting to regression for dtype '{dtype}' with {n_unique} unique values."

    warnings_list = []
    if n_unique > 50 and suggestion in ("binary_classification", "multiclass_classification"):
        warnings_list.append(f"High cardinality target ({n_unique} classes) may cause poor classification performance.")
    if suggestion == "multiclass_classification":
        class_counts = series.value_counts()
        min_count = int(class_counts.min())
        if min_count < 5:
            warnings_list.append(f"Some classes have very few samples (min: {min_count}). Stratified split may fail.")

    return {
        "suggestion": suggestion,
        "rationale": rationale,
        "n_unique": n_unique,
        "dtype": dtype,
        "warnings": warnings_list,
    }


def get_target_distribution(df: pd.DataFrame, target_col: str, task_type: str) -> Dict[str, Any]:
    """Compute target value distribution."""
    series = df[target_col].dropna()
    if task_type in ("binary_classification", "multiclass_classification"):
        freq = series.value_counts().head(30)
        return {
            "type": "categorical",
            "counts": [{"label": str(k), "count": int(v)} for k, v in freq.items()],
            "n_classes": int(series.nunique()),
        }
    else:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        counts, edges = np.histogram(clean, bins=30)
        return {
            "type": "continuous",
            "histogram": {"counts": counts.tolist(), "edges": [round(float(e), 4) for e in edges.tolist()]},
            "mean": round(float(clean.mean()), 4),
            "std": round(float(clean.std()), 4),
        }
