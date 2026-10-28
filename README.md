# Tabular AI Research Lab

An end-to-end, local-first research platform for exploring tabular datasets, configuring leakage-safe preprocessing, training classical and deep-learning models, evaluating time-series forecasts, and generating explainable AI reports.

---

## Overview

**Tabular AI Research Lab** provides a unified environment for researchers and engineers working with structured and multimodal tabular data. The platform combines exploratory data analysis, automated task inference, reproducible feature engineering, model training (decision trees, gradient boosting, linear models, TabNet, and FT-Transformers), SHAP explainability, and interactive reporting into a single full-stack application.

---

## Key Features

- **Authentication & User Isolation**: Session-based authentication with bcrypt password hashing (minimum 12 characters), persistent cookie sessions, and per-user dataset/experiment isolation.
- **Dataset Registry & Exploration**:
  - Drag-and-drop CSV upload with automatic delimiter, encoding, and schema detection.
  - In-depth profiling: missing values, summary statistics, histograms, and Pearson correlation matrices.
  - Kaggle API integration: direct in-app search, download, and ingestion of public datasets.
  - Synthetic data generation: statistical bootstrapping and optional CTGAN/CopulaGAN synthesis via SDV.
- **Leakage-Safe Preprocessing Pipeline**:
  - Imputation (mean, median, most frequent, constant), scalers (standard, robust, none), and encodings (one-hot, ordinal).
  - Text feature engineering via TF-IDF or optional sentence embeddings (`all-MiniLM-L6-v2`).
  - Image embedding extraction via pre-trained ResNet-18 (512-dim) with on-disk SHA-256 caching and a 2,000-image safety cap.
  - Target auto-detection (binary classification, multiclass, regression, time-series).
  - Strict isolation: transformers and scalers are fitted exclusively on the training split.
- **Time-Series Forecasting**:
  - Full support for `time_series` task type with chronological train/test splitting (no shuffling).
  - Automated feature extraction: configurable lag features (default: 1, 7, 14), rolling mean and standard deviation windows (default: 3, 7 shifted by 1 to prevent future leakage), and calendar signals (`month`, `weekday`, `is_weekend`).
  - `TimeSeriesSplit` cross-validation and evaluation metrics: MAE, RMSE, MAPE, and R².
  - Interactive forecast vs. actual visualization for test partitions.
  - Built-in `daily_sales.csv` sample dataset with a one-click automated demo.
- **Model Lab & Hyperparameter Tuning**:
  - Classical models: Logistic Regression, Linear Regression, Random Forest, Gradient Boosting, XGBoost, and KNN.
  - Deep tabular models: Multi-Layer Perceptron (PyTorch MLP), TabNet, and FT-Transformer with validation-based early stopping.
  - Hyperparameter optimization: Grid search, randomized search, and optional Bayesian optimization via Optuna.
  - Real-time training telemetry: live progress tracking, loss curves, and non-blocking background workers.
- **Explainability & Model Diagnostics**:
  - TreeSHAP and KernelSHAP summary plots for global and local feature contributions.
  - Permutation importance fallback for complex pipelines.
- **Inference & Export**:
  - Interactive single-row prediction with form controls.
  - Batch CSV inference with downloadable scored CSV files.
  - HTML and JSON research report generation with comparative metric evaluations.

---

## Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | React 19, Vite, React Router 7, Tailwind CSS, Recharts, Lucide Icons, Axios |
| **Backend API** | Python 3.11+, FastAPI, Uvicorn, Pydantic v2, SQLAlchemy, SQLite |
| **Machine Learning** | scikit-learn, NumPy, pandas, PyTorch, Torchvision, Pillow, XGBoost, SHAP |
| **Optional Extras** | `pytorch-tabnet`, `rtdl-revisiting-models`, `sentence-transformers`, `sdv`, `optuna` |
| **Storage** | SQLite (`data/tabular_lab.db`), local filesystem artifact store (`backend/storage/`) |

---

## Architecture

```text
┌────────────────────────────────────────────────────────┐
│               React + Vite Frontend                    │
│               http://localhost:5173                    │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP / API Requests
                            ▼
┌────────────────────────────────────────────────────────┐
│                 Vite Reverse Proxy                     │
│               (routes /api to :8000)                   │
└───────────────────────────┬────────────────────────────┘
                            │ Internal HTTP Forwarding
                            ▼
┌────────────────────────────────────────────────────────┐
│               FastAPI + Uvicorn Backend                │
│                 http://127.0.0.1:8000                  │
└─────────────┬───────────────────────────┬──────────────┘
              │                           │
              ▼                           ▼
┌───────────────────────────┐ ┌──────────────────────────┐
│  ML & Preprocessing Core  │ │  Background Task Runner  │
│  - scikit-learn pipelines │ │  - ThreadPoolExecutor    │
│  - ResNet-18 embeddings   │ │  - Async model training  │
│  - PyTorch / TabNet / FT  │ │  - Non-blocking status   │
│  - SHAP explainability    │ │  - Synthetic data jobs   │
└─────────────┬─────────────┘ └───────────┬──────────────┘
              │                           │
              └─────────────┬─────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│                  Persistence Layer                     │
│  - SQLite Database (Users, Datasets, Experiments)      │
│  - Local Storage (CSV uploads, joblib/pt models)       │
│  - Image Embedding Cache (backend/storage/image_cache) │
└────────────────────────────────────────────────────────┘
```

---

## Windows PowerShell Setup

Follow these steps to set up and run the full stack on Windows PowerShell:

### 1. Clone the Repository

```powershell
git clone https://github.com/ruthikamandadi/tabular-ai-research-lab.git
Set-Location tabular-ai-research-lab
```

### 2. Configure Environment Variables

Copy the provided `.env.example` file to `.env`:

```powershell
Copy-Item .env.example .env
```

### 3. Backend Setup

Create a virtual environment and install the core dependencies:

```powershell
Set-Location backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

*(Optional)* Install deep-learning extensions (TabNet, FT-Transformer, Pillow, Torchvision):

```powershell
python -m pip install -r requirements-optional.txt
Set-Location ..
```

### 4. Frontend & Root Setup

Install dependencies for both the frontend and root orchestrator:

```powershell
npm --prefix frontend install
npm install
```

### 5. Start Both Servers Together

From the repository root, run the unified startup script:

```powershell
npm run dev:all
```

The orchestrator starts the FastAPI backend, waits for `http://127.0.0.1:8000/api/health` to respond, and then launches the Vite development server.

---

## Application URLs

Once running, access the services at:

- **Frontend Application**: [http://localhost:5173](http://localhost:5173)
- **Backend Health Check**: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)
- **Interactive OpenAPI Documentation**: [http://127.0.0.1:8000/api/docs](http://127.0.0.1:8000/api/docs)
- **Alternative Redoc Documentation**: [http://127.0.0.1:8000/api/redoc](http://127.0.0.1:8000/api/redoc)

---

## User Registration & Authentication

1. Open [http://localhost:5173/register](http://localhost:5173/register) in your browser.
2. Fill in:
   - **Full Name**: e.g. `Research Engineer`
   - **Email Address**: e.g. `researcher@example.com`
   - **Password**: Must contain at least **12 characters** (e.g., `SecurePassword123!`).
   - **Confirm Password**: Matching password.
3. Upon clicking **Create Account**, you are immediately signed in with an HTTP-only session cookie.
4. All uploaded datasets, experiments, and saved models belong exclusively to your user session.

---

## Kaggle API Credentials Setup

To search and download Kaggle datasets directly from the **Datasets** page:

1. Log into your account at [kaggle.com](https://www.kaggle.com).
2. Click your profile avatar (top-right) and navigate to **Settings** (`https://www.kaggle.com/settings`).
3. Scroll down to the **API** section and click **Create New Token**.
4. A file named `kaggle.json` will be downloaded to your machine:
   ```json
   {"username": "your_username", "key": "your_api_key"}
   ```
5. Choose one of two ways to configure credentials:
   - **Via the UI (Recommended)**: Open **Settings** in the application, paste your username and API key into the **Kaggle API Credentials** card, and click **Save Credentials** followed by **Test Connection**.
   - **Via File/Environment**: Copy `kaggle.json` directly into `backend/storage/kaggle.json`, or set `KAGGLE_USERNAME` and `KAGGLE_KEY` in your `.env` file.

---

## Running Tests and Building

### Backend Tests

Execute the full pytest suite (including time-series, image caching, and authentication tests):

```powershell
backend\.venv\Scripts\python.exe -m pytest -q
```

### Frontend Production Build

Validate TypeScript/JSX compilation and asset bundling:

```powershell
npm --prefix frontend run build
```

---

## Design Decisions

- **Leakage-Safe Preprocessing**:
  Preprocessing transformations (imputation, scaling, one-hot encoding, and feature engineering) are fitted strictly on `X_train` and applied out-of-sample to `X_test`. For time-series tasks, train/test splitting is strictly chronological without shuffling, and rolling features are shifted by 1 step so that predictions at time step $t$ only consume observations up to $t-1$.
- **Non-Blocking Background Execution**:
  Model fitting, cross-validation, and synthetic dataset generation execute inside background threads. The API yields execution immediately upon task submission, allowing the client to poll `/api/experiments/{id}/status` without HTTP request timeouts.
- **SHAP Subsampling**:
  Computing exact Shapley values across thousands of rows can lead to exponential complexity and CPU exhaustion. The platform subsamples background reference data to 50–100 points via k-means clustering and evaluates explanations on a capped sample of 100 test instances, falling back to model-agnostic permutation importance when TreeSHAP is unavailable.
- **Large Dataset Defenses**:
  To ensure smooth client performance, data exploration previews are capped at 100 rows, hyperparameter searches on datasets exceeding 50,000 rows automatically extract a 20,000-row chronological slice, and image feature extraction is capped at 2,000 images per run with on-disk `.npy` caching.

---

## Troubleshooting

| Issue | Likely Cause | Solution |
| :--- | :--- | :--- |
| `ERR_CONNECTION_REFUSED` | The backend or frontend dev server is not running. | Verify both processes are running via `npm run dev:all`. Confirm backend is active at `http://127.0.0.1:8000/api/health`. |
| `502 Bad Gateway` | Vite development server proxy cannot reach FastAPI. | The backend is still initializing or encountered a startup crash. Check the backend PowerShell console output for Python exceptions. |
| `404 Not Found` on `/api/*` | Route path mismatch or backend started from the wrong working directory. | Ensure Uvicorn is launched from the `backend/` directory (`app.main:app`). Verify the target route in `http://127.0.0.1:8000/api/docs`. |
| `ModuleNotFoundError: No module named 'app'` | Python is executed from the repository root rather than `backend/`. | Activate `backend\.venv` and set the working directory to `backend/` before running Uvicorn: `Set-Location backend; uvicorn app.main:app`. |
| `DLL load failed` on Windows | Missing Microsoft Visual C++ Redistributable runtimes. | Install the latest 64-bit Visual C++ Redistributable (`vc_redist.x64.exe`) required by PyTorch, scikit-learn, and XGBoost C-extensions. |
| `Port already in use` (`8000` or `5173`) | A lingering Python or Node process is already bound to the port. | In PowerShell, find the PID: `netstat -ano \| findstr :8000`. Kill the offending process: `Stop-Process -Id <PID> -Force`, or configure `BACKEND_PORT=8001` in `.env`. |

---

## Known Limitations and Future Work

- **Task Queue Architecture**: Background jobs currently rely on Python's in-process `ThreadPoolExecutor`. For clustered production deployments, this should be migrated to a distributed task queue (e.g. Celery or ARQ with Redis).
- **Concurrency & Storage**: The current data layer uses SQLite with file-based locking. High concurrent write throughput will benefit from PostgreSQL and object storage (e.g. AWS S3 or MinIO).
- **Inference Serving**: Models are currently loaded into memory per inference request. Deploying dedicated low-latency serving runtimes (e.g. Triton Inference Server or TorchServe) is recommended for production traffic.
- **Multivariate Forecasting**: The time-series engine supports autoregressive lag and rolling feature regression (Linear Regression, Random Forest, XGBoost). Future releases will incorporate dedicated neural forecasting architectures (such as PatchTST, N-BEATS, and DLinear).
- **GPU Acceleration**: PyTorch components automatically utilize NVIDIA CUDA if available; when running on systems without a dedicated GPU, processing defaults to CPU execution.

---

## Screenshots

Visual documentation placeholders and guidelines for capturing key user workflows are located in the [docs/screenshots](docs/screenshots/README.md) directory.
