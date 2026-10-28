# Demo script

The Overview page's **Demo mode** button runs the first two modeling steps automatically with a deterministic, generated 120-row churn dataset. It uploads the CSV through the normal dataset endpoint, then runs preprocessing and trains Logistic Regression and Random Forest through the regular experiment workflow.

## Exact order of clicks

1. Start the backend and frontend using the Windows setup steps in the project [README](../README.md).
   The first start can take up to a minute while the backend initializes; the combined startup command waits for it automatically.
2. Open <http://localhost:5173> and wait for the Overview page to show **Backend Online**.
3. Click **Demo mode** once. Leave the page open while the progress bar reports dataset loading, preprocessing, and each model training.
4. When the success message appears, click **View Experiments**.
5. On Experiments, click **Refresh** if the page has not yet refreshed, then expand each completed run to inspect its metrics.
6. To continue the guided workflow, open **Explainability**, select a completed run, and inspect its feature importance. Use **Predictions** or **Reports** to explore the remaining workflow steps.

## Expected result

- A dataset named `Demo Churn 120` appears in Datasets with 120 rows and `churn` as its target column.
- Two completed classification experiments appear: Logistic Regression and Random Forest.
- Each run includes classification metrics; the Overview counters update with the dataset, experiment count, and highest completed accuracy.
- The demo ends with a **View Experiments** link. If upload or training fails, the Overview page displays a readable error instead of silently stopping.

The dataset is intentionally small and deterministic for demonstrations. Its scores are illustrative, not a benchmark.
