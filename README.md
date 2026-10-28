# Tabular AI Research Lab

A local research application for exploring tabular datasets, configuring preprocessing, training classification and regression models, and inspecting experiment results.

## Overview

Tabular AI Research Lab is a local-first research prototype for an end-to-end tabular machine-learning workflow. Use it to upload data, explore data quality, configure preprocessing, train and compare models, explain predictions, and export reports. The Overview page includes a guided 120-row churn demo.

## Features

- Upload CSV datasets and load built-in samples.
- Explore dataset profiles, columns, missing values, and row previews.
- Configure target columns, preprocessing, and train/test splits.
- Train compatible scikit-learn and PyTorch models.
- Track experiment status and metrics.
- Review feature importance, make single-row or CSV predictions, and export reports.
- Store dataset and experiment metadata in SQLite and model artifacts on disk.
- Optionally train TabNet and FT-Transformer for tabular data, with device-aware training and early stopping.
- Run the guided demo from Overview to load a 120-row churn sample and train Logistic Regression and Random Forest.
- Generate synthetic datasets, import Kaggle datasets, and explore supported text, date, image-path, and time-series features.

## Tech stack

- **Frontend:** React, Vite, React Router, Tailwind CSS, Recharts, Axios.
- **Backend:** Python, FastAPI, Uvicorn, Pydantic.
- **Machine learning:** pandas, NumPy, scikit-learn; optional PyTorch, TabNet, FT-Transformer, SHAP, Kaggle, and synthetic-data packages.
- **Persistence:** SQLite metadata, local CSV uploads, and joblib/model artifacts.

### Optional deep-learning models

The base app does not require either architecture package. Install the optional packages from `backend` to enable both models:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-optional.txt
```

Or install one at a time:

```powershell
.\.venv\Scripts\python.exe -m pip install pytorch-tabnet
.\.venv\Scripts\python.exe -m pip install rtdl-revisiting-models
```

PyTorch is required for both optional models. Restart the backend after installing packages; Model Lab only lists architectures that are available in the active environment.

## Architecture

```text
┌─────────────────────────────┐       HTTP /api       ┌──────────────────────────────┐
│ React + Vite                 │ ────────────────────> │ FastAPI + Uvicorn             │
│ frontend/                    │ <── Vite proxy ────── │ backend/app/                  │
│ http://localhost:5173        │                       │ http://127.0.0.1:8000        │
└─────────────────────────────┘                       └──────────────┬───────────────┘
                                                                     │
                                               ┌─────────────────────┴────────────┐
                                               │                                  │
                                      ┌────────▼─────────┐              ┌─────────▼────────┐
                                      │ SQLite metadata   │              │ Local data and    │
                                      │ datasets/runs     │              │ model artifacts   │
                                      └──────────────────┘              └──────────────────┘
```

## Setup (Windows PowerShell)

Copy `.env.example` to `.env` in the project root and adjust values if needed.

### Backend

In a PowerShell terminal:

```powershell
Set-Location 'C:\path\to\tabular-ai-research-lab\backend'
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

If `BACKEND_PORT` in `.env` is not `8000`, pass that port to Uvicorn and update the Vite proxy target in `frontend/vite.config.js`.

### Frontend

In a second PowerShell terminal:

```powershell
Set-Location 'C:\path\to\tabular-ai-research-lab\frontend'
npm install
npm run dev
```

### Run both servers together

Create and activate `backend\.venv`, install backend requirements, then install the root and frontend Node dependencies as described above. From the project root run:

```powershell
Set-Location 'C:\path\to\tabular-ai-research-lab'
npm install
npm --prefix frontend install
npm run dev:all
```

The first start can take up to a minute while the backend initializes. The combined script waits for the backend health check before starting Vite, then serves FastAPI from `backend` on port 8000 and Vite on port 5173. Use Ctrl+C to stop both processes.

## URLs

- Frontend: <http://localhost:5173>
- Backend health: <http://127.0.0.1:8000/api/health>
- Interactive API documentation: <http://127.0.0.1:8000/api/docs>

## Tests and build

```powershell
Set-Location 'C:\path\to\tabular-ai-research-lab\backend'
.\.venv\Scripts\python.exe -m pytest

Set-Location 'C:\path\to\tabular-ai-research-lab\frontend'
npm run build
```

## Troubleshooting

### `ERR_CONNECTION_REFUSED`

- Confirm both server terminals are still running.
- Check that the backend is listening on `127.0.0.1:8000` and Vite on `localhost:5173`.
- If you changed `BACKEND_PORT`, use the same port for Uvicorn and the `/api` proxy target in `frontend/vite.config.js`.
- Run `http://127.0.0.1:8000/api/health` directly to check backend availability.

### Vite proxy returns `502`

- Confirm Uvicorn started without import or dependency errors in the backend terminal.
- Open `http://127.0.0.1:8000/api/health` directly. It should return JSON with `"status": "ok"`.
- Confirm `frontend/vite.config.js` targets `http://127.0.0.1:8000`, then restart Vite after changing the proxy.

### 404 for an `/api` request

- Start the backend from the `backend` directory with `app.main:app`.
- Confirm the request path begins with `/api` and is listed in the docs at `/api/docs`.
- The Vite development server proxies `/api` to the backend; restart Vite after changing its proxy configuration.
- Verify the route and HTTP method in `http://127.0.0.1:8000/api/docs`; an unsupported path or method returns 404/405.
- Use `/api/health` or `/api/docs` to verify the backend rather than assuming every endpoint exists.

### `ModuleNotFoundError: No module named 'app'`

- Change into the `backend` directory before running Uvicorn: `Set-Location .\backend`.
- Activate `.\.venv\Scripts\Activate.ps1` and install `python -m pip install -r requirements.txt` in that environment.
- Start with `python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000`. Running from the repository root can prevent Python from finding the `app` package.

## Limitations

- This is a local research/demo application, not a hardened multi-user or production deployment.
- Model quality depends on the dataset and split; the included demo is small and its metrics are illustrative.
- Optional integrations and architectures require their packages and, for Kaggle, valid credentials and accepted dataset rules.
- Image embeddings, sentence embeddings, SHAP, and deep-learning model availability depends on installed packages and compatible hardware.
- Large datasets and expensive model/tuning/explainability jobs can take significant time and memory.
- Some generated features and optional model implementations are intentionally best-effort; check Settings for the current runtime capability status.

## Guided demo

See [docs/DEMO.md](docs/DEMO.md) for the exact click sequence and expected output.
