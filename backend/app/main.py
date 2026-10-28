"""
FastAPI main application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import APP_TITLE, APP_VERSION, FRONTEND_ORIGINS
from app.core.database import init_db
from app.api import datasets, experiments, kaggle, predictions, reports, settings
from app.core import auth

app = FastAPI(title=APP_TITLE, version=APP_VERSION, docs_url="/api/docs", redoc_url="/api/redoc")

# CORS - restrict to frontend origins in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth.router)
app.include_router(datasets.router)
app.include_router(experiments.router)
app.include_router(predictions.router)
app.include_router(reports.router)
app.include_router(settings.router)
app.include_router(kaggle.router)


@app.get("/")
def root():
    return {"status": "ok"}


@app.on_event("startup")
def startup_event():
    init_db()
    _create_sample_datasets()


def _create_sample_datasets():
    """Generate sample CSV datasets if they don't exist."""
    import pandas as pd
    import numpy as np
    from app.core.config import SAMPLE_DIR

    np.random.seed(42)

    # Sample 1: Titanic-like passenger survival (binary classification)
    if not (SAMPLE_DIR / "titanic_like.csv").exists():
        n = 800
        df = pd.DataFrame({
            "passenger_class": np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55]),
            "sex": np.random.choice(["male", "female"], n, p=[0.65, 0.35]),
            "age": np.clip(np.random.normal(30, 14, n), 1, 80).round(1),
            "siblings_spouses": np.random.choice([0, 1, 2, 3, 4], n, p=[0.68, 0.23, 0.06, 0.02, 0.01]),
            "parents_children": np.random.choice([0, 1, 2, 3], n, p=[0.76, 0.13, 0.09, 0.02]),
            "fare": np.clip(np.random.exponential(32, n), 5, 512).round(2),
            "embarked": np.random.choice(["S", "C", "Q"], n, p=[0.72, 0.19, 0.09]),
        })
        # Survival probability based on features
        survival_prob = (
            0.5
            + (df["sex"] == "female").astype(float) * 0.35
            - (df["passenger_class"] == 3).astype(float) * 0.25
            + (df["passenger_class"] == 1).astype(float) * 0.15
            - (df["age"] > 60).astype(float) * 0.1
            + np.random.normal(0, 0.1, n)
        ).clip(0.05, 0.95)
        df["survived"] = (np.random.rand(n) < survival_prob).astype(int)
        df.to_csv(SAMPLE_DIR / "titanic_like.csv", index=False)

    # Sample 2: House price prediction (regression)
    if not (SAMPLE_DIR / "house_prices.csv").exists():
        n = 600
        df = pd.DataFrame({
            "square_feet": np.random.randint(600, 4000, n),
            "bedrooms": np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.2, 0.45, 0.25, 0.05]),
            "bathrooms": np.random.choice([1, 1.5, 2, 2.5, 3], n),
            "year_built": np.random.randint(1950, 2023, n),
            "garage_spaces": np.random.choice([0, 1, 2, 3], n, p=[0.1, 0.3, 0.5, 0.1]),
            "neighborhood": np.random.choice(["downtown", "suburban", "rural", "waterfront"], n, p=[0.25, 0.45, 0.2, 0.1]),
            "condition": np.random.choice(["poor", "fair", "good", "excellent"], n, p=[0.05, 0.2, 0.55, 0.2]),
            "has_pool": np.random.choice([0, 1], n, p=[0.75, 0.25]),
        })
        price = (
            df["square_feet"] * 150
            + df["bedrooms"] * 8000
            + df["bathrooms"] * 5000
            + (2024 - df["year_built"]) * -500
            + df["garage_spaces"] * 6000
            + (df["neighborhood"] == "waterfront").astype(int) * 80000
            + (df["neighborhood"] == "downtown").astype(int) * 30000
            + (df["condition"] == "excellent").astype(int) * 20000
            + (df["condition"] == "poor").astype(int) * -15000
            + df["has_pool"] * 15000
            + np.random.normal(0, 20000, n)
        ).clip(50000, 1500000).round(-3)
        df["price_usd"] = price.astype(int)
        df.to_csv(SAMPLE_DIR / "house_prices.csv", index=False)

    # Sample 3: Iris-like flower classification (multiclass)
    if not (SAMPLE_DIR / "iris_like.csv").exists():
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        df = iris.frame
        df["target"] = iris.target_names[iris.target]
        df.to_csv(SAMPLE_DIR / "iris_like.csv", index=False)

    # Sample 4: Customer churn (binary classification with more features)
    if not (SAMPLE_DIR / "customer_churn.csv").exists():
        n = 1000
        tenure = np.random.randint(1, 72, n)
        monthly_charges = np.random.uniform(20, 120, n).round(2)
        total_charges = (tenure * monthly_charges + np.random.normal(0, 50, n)).clip(0).round(2)
        df = pd.DataFrame({
            "tenure_months": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": np.random.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.24, 0.21]),
            "internet_service": np.random.choice(["DSL", "fiber", "no"], n, p=[0.35, 0.44, 0.21]),
            "payment_method": np.random.choice(["electronic", "mailed_check", "bank_transfer", "credit_card"], n),
            "num_support_calls": np.random.poisson(1.5, n),
            "tech_support": np.random.choice(["yes", "no"], n, p=[0.35, 0.65]),
            "online_security": np.random.choice(["yes", "no"], n, p=[0.4, 0.6]),
            "senior_citizen": np.random.choice([0, 1], n, p=[0.84, 0.16]),
            "dependents": np.random.choice(["yes", "no"], n, p=[0.3, 0.7]),
        })
        churn_prob = (
            0.27
            + (df["contract_type"] == "month-to-month").astype(float) * 0.20
            + (df["internet_service"] == "fiber").astype(float) * 0.10
            - (df["tenure_months"] / 72) * 0.2
            + df["num_support_calls"] * 0.03
            + np.random.normal(0, 0.08, n)
        ).clip(0.02, 0.95)
        df["churn"] = (np.random.rand(n) < churn_prob).map({True: "Yes", False: "No"})
        df.to_csv(SAMPLE_DIR / "customer_churn.csv", index=False)


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "service": APP_TITLE,
    }
