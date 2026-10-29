"""
Model implementations: Traditional ML + PyTorch MLP.
All models implement a common interface: fit, predict, predict_proba, save, load.
"""
import numpy as np
import time
import joblib
import importlib.util
from typing import Dict, Any, List, Optional

XGBOOST_AVAILABLE = importlib.util.find_spec("xgboost") is not None

torch = None
nn = None
optim = None
DataLoader = None
TensorDataset = None
TabNetClassifier = None
TabNetRegressor = None
rtdl_revisiting_models = None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
TABNET_AVAILABLE = TORCH_AVAILABLE and importlib.util.find_spec("pytorch_tabnet") is not None
FT_TRANSFORMER_AVAILABLE = TORCH_AVAILABLE and importlib.util.find_spec("rtdl_revisiting_models") is not None


def _load_torch():
    global torch, nn, optim, DataLoader, TensorDataset
    if torch is None:
        import torch as torch_module
        import torch.nn as nn_module
        import torch.optim as optim_module
        from torch.utils.data import DataLoader as data_loader, TensorDataset as tensor_dataset

        torch = torch_module
        nn = nn_module
        optim = optim_module
        DataLoader = data_loader
        TensorDataset = tensor_dataset
    return torch


def _load_tabnet():
    global TabNetClassifier, TabNetRegressor
    _load_torch()
    if TabNetClassifier is None:
        from pytorch_tabnet.tab_model import TabNetClassifier as classifier, TabNetRegressor as regressor

        TabNetClassifier = classifier
        TabNetRegressor = regressor
    return TabNetClassifier, TabNetRegressor


def _load_ft_transformer():
    global rtdl_revisiting_models
    _load_torch()
    if rtdl_revisiting_models is None:
        import rtdl_revisiting_models as models

        rtdl_revisiting_models = models
    return rtdl_revisiting_models


def get_torch():
    return _load_torch()


# ─────────────────────────────────────────────
#  Sklearn-based models
# ─────────────────────────────────────────────

class SklearnModelWrapper:
    """Wraps any sklearn estimator with a common interface."""

    def __init__(self, estimator: Any, task_type: str):
        self.estimator = estimator
        self.task_type = task_type
        self.training_history = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict:
        t0 = time.time()
        self.estimator.fit(X_train, y_train)
        duration = time.time() - t0
        return {"train_duration_s": duration, "training_history": []}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)
        if hasattr(self.estimator, "decision_function"):
            scores = self.estimator.decision_function(X)
            if scores.ndim == 1:
                proba = 1 / (1 + np.exp(-scores))
                return np.stack([1 - proba, proba], axis=1)
        return None

    def feature_importances(self) -> Optional[np.ndarray]:
        if hasattr(self.estimator, "feature_importances_"):
            return self.estimator.feature_importances_
        if hasattr(self.estimator, "coef_"):
            coef = self.estimator.coef_
            if coef.ndim == 2:
                return np.abs(coef).mean(axis=0)
            return np.abs(coef)
        return None

    def save(self, path: str):
        joblib.dump({"estimator": self.estimator, "task_type": self.task_type}, path)

    @classmethod
    def load(cls, path: str) -> "SklearnModelWrapper":
        data = joblib.load(path)
        obj = cls(data["estimator"], data["task_type"])
        return obj


# ─────────────────────────────────────────────
#  PyTorch MLP
# ─────────────────────────────────────────────

def _build_mlp_network(input_dim: int, hidden_layers: List[int], output_dim: int, dropout: float):
    _load_torch()
    layers = []
    previous_dim = input_dim
    for hidden_dim in hidden_layers:
        layers += [nn.Linear(previous_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


class PyTorchMLPWrapper:
    """PyTorch MLP with configurable layers, early stopping, and loss history."""

    def __init__(self, task_type: str, config: Dict[str, Any]):
        self.task_type = task_type
        self.config = config
        _load_torch()
        self.model = None
        self.training_history: List[Dict] = []
        self.n_classes: Optional[int] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self, input_dim: int, n_classes: int):
        hidden_layers = self.config.get("hidden_layers", [128, 64])
        dropout = self.config.get("dropout", 0.2)

        if self.task_type == "binary_classification":
            output_dim = 1
        elif self.task_type == "multiclass_classification":
            output_dim = n_classes
        else:
            output_dim = 1

        return _build_mlp_network(input_dim, hidden_layers, output_dim, dropout).to(self.device)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            progress_callback=None) -> Dict:
        t0 = time.time()
        lr = self.config.get("learning_rate", 1e-3)
        batch_size = self.config.get("batch_size", 64)
        epochs = self.config.get("epochs", 50)
        weight_decay = self.config.get("weight_decay", 1e-4)
        patience = self.config.get("patience", 10)
        seed = self.config.get("random_seed", 42)

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Determine n_classes
        if self.task_type == "multiclass_classification":
            self.n_classes = int(y_train.max()) + 1
        else:
            self.n_classes = 2

        input_dim = X_train.shape[1]
        self.model = self._build_model(input_dim, self.n_classes)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        # Loss function
        if self.task_type == "binary_classification":
            criterion = nn.BCEWithLogitsLoss()
        elif self.task_type == "multiclass_classification":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        # Tensors
        X_t = torch.tensor(X_train, dtype=torch.float32)
        if self.task_type == "binary_classification":
            y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        elif self.task_type == "multiclass_classification":
            y_t = torch.tensor(y_train, dtype=torch.long)
        else:
            y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        has_val = X_val is not None and y_val is not None
        if has_val:
            Xv_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            if self.task_type == "binary_classification":
                yv_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(self.device)
            elif self.task_type == "multiclass_classification":
                yv_t = torch.tensor(y_val, dtype=torch.long).to(self.device)
            else:
                yv_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(self.device)

        self.training_history = []
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = self.model(Xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(Xb)

            epoch_loss /= len(X_train)
            record: Dict[str, Any] = {"epoch": epoch + 1, "train_loss": round(epoch_loss, 6)}

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(Xv_t)
                    val_loss = criterion(val_out, yv_t).item()
                record["val_loss"] = round(val_loss, 6)
                scheduler.step(val_loss)

                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        self.training_history.append(record)
                        break
            else:
                scheduler.step(epoch_loss)

            self.training_history.append(record)

            if progress_callback:
                progress_callback(epoch + 1, epochs, epoch_loss)

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        duration = time.time() - t0
        return {"train_duration_s": duration, "training_history": self.training_history}

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self.model(X_t)
            if self.task_type == "binary_classification":
                probs = torch.sigmoid(out).cpu().numpy().flatten()
                return (probs >= 0.5).astype(int)
            elif self.task_type == "multiclass_classification":
                return torch.argmax(out, dim=1).cpu().numpy()
            else:
                return out.cpu().numpy().flatten()

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self.model(X_t)
            if self.task_type == "binary_classification":
                probs = torch.sigmoid(out).cpu().numpy().flatten()
                return np.stack([1 - probs, probs], axis=1)
            elif self.task_type == "multiclass_classification":
                probs = torch.softmax(out, dim=1).cpu().numpy()
                return probs
        return None

    def feature_importances(self) -> Optional[np.ndarray]:
        # MLP doesn't have built-in importances; handled by explainability via permutation
        return None

    def save(self, path: str):
        state = {
            "model_state": self.model.state_dict() if self.model else None,
            "task_type": self.task_type,
            "config": self.config,
            "n_classes": self.n_classes,
            "training_history": self.training_history,
            "input_dim": next(iter(self.model.parameters())).shape[1] if self.model else None,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str) -> "PyTorchMLPWrapper":
        state = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(state["task_type"], state["config"])
        obj.n_classes = state["n_classes"]
        obj.training_history = state.get("training_history", [])
        input_dim = state["input_dim"]
        n_classes = state["n_classes"] or 2
        obj.model = obj._build_model(input_dim, n_classes)
        obj.model.load_state_dict(state["model_state"])
        obj.model.eval()
        return obj


class TabNetModelWrapper:
    def __init__(self, task_type, config):
        if not TABNET_AVAILABLE:
            raise RuntimeError("TabNet requires torch and pytorch-tabnet. Install with: pip install pytorch-tabnet")
        _load_torch()
        self.task_type = task_type
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.estimator = None
        self.training_history = []

    def fit(self, X_train, y_train, X_val=None, y_val=None, progress_callback=None):
        started = time.time()
        classification = self.task_type not in ("regression", "time_series")
        tabnet_classifier, tabnet_regressor = _load_tabnet()
        estimator_cls = tabnet_classifier if classification else tabnet_regressor
        self.estimator = estimator_cls(
            n_d=int(self.config.get("n_d", 8)),
            n_a=int(self.config.get("n_a", 8)),
            n_steps=int(self.config.get("n_steps", 3)),
            optimizer_fn=optim.Adam,
            optimizer_params={"lr": float(self.config.get("learning_rate", 0.02))},
            device_name=self.device,
            verbose=0,
            seed=int(self.config.get("random_seed", 42)),
        )
        callback_base = __import__("pytorch_tabnet.callbacks", fromlist=["Callback"]).Callback
        progress = progress_callback or (lambda *args: None)
        max_epochs = int(self.config.get("max_epochs", 50))
        task_type = self.task_type

        class TabNetProgressCallback(callback_base):
            def __init__(self):
                super().__init__()
                self.history = []

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                train_loss = float(logs.get("loss", 0.0))
                validation_score = next(
                    (float(value) for key, value in logs.items() if key.startswith(("val_", "valid_"))),
                    None,
                )
                self.history.append({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": validation_score if task_type in ("regression", "time_series") else None,
                    "val_score": validation_score,
                })
                progress(epoch + 1, max_epochs, train_loss, validation_score)

        callback = TabNetProgressCallback()
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        y_train_fit = np.asarray(y_train).reshape(-1, 1) if not classification else np.asarray(y_train)
        y_val_fit = np.asarray(y_val).reshape(-1, 1) if eval_set and not classification else y_val
        effective_batch_size = max(2, min(int(self.config.get("batch_size", 1024)), len(X_train)))
        virtual_batch_size = min(128, effective_batch_size)
        while effective_batch_size % virtual_batch_size:
            virtual_batch_size -= 1
        self.estimator.fit(
            np.asarray(X_train, dtype=np.float32),
            y_train_fit,
            eval_set=eval_set if classification else (
                [(X_val, y_val_fit)] if eval_set else None
            ),
            eval_name=["valid"],
            eval_metric=["accuracy" if classification else "rmse"],
            max_epochs=int(self.config.get("max_epochs", 50)),
            patience=int(self.config.get("patience", 10)),
            batch_size=effective_batch_size,
            virtual_batch_size=virtual_batch_size,
            drop_last=False,
            callbacks=[callback],
        )
        self.training_history = callback.history
        return {"train_duration_s": time.time() - started, "training_history": self.training_history}

    def predict(self, X):
        prediction = self.estimator.predict(np.asarray(X, dtype=np.float32))
        return np.asarray(prediction).reshape(-1)

    def predict_proba(self, X):
        if self.task_type in ("regression", "time_series"):
            return None
        return self.estimator.predict_proba(np.asarray(X, dtype=np.float32))

    def feature_importances(self):
        return getattr(self.estimator, "feature_importances_", None)

    def explain(self, X):
        return self.estimator.explain(np.asarray(X, dtype=np.float32))

    def save(self, path):
        _load_torch()
        self.estimator._callback_container = None
        self.estimator.device = torch.device("cpu")
        self.estimator.network.to(self.estimator.device)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        _load_torch()
        model = joblib.load(path)
        model.device = "cuda" if torch.cuda.is_available() else "cpu"
        model.estimator.device = torch.device(model.device)
        model.estimator.network.to(model.estimator.device)
        return model


class FTTransformerModelWrapper:
    def __init__(self, task_type, config):
        if not FT_TRANSFORMER_AVAILABLE:
            raise RuntimeError("FT-Transformer requires torch and rtdl-revisiting-models. Install with: pip install rtdl-revisiting-models")
        _load_torch()
        self.task_type = task_type
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.training_history = []
        self.classes = 2

    def _build(self, input_dim, n_classes):
        output_dim = 1 if self.task_type in ("regression", "time_series") else n_classes
        d_token = int(self.config.get("d_token", 64))
        heads = int(self.config.get("attention_heads", 8))
        if d_token % heads:
            raise ValueError("d_token must be divisible by attention_heads.")
        return _load_ft_transformer().FTTransformer(
            n_cont_features=input_dim,
            cat_cardinalities=[],
            d_out=output_dim,
            n_blocks=int(self.config.get("n_blocks", 3)),
            d_block=d_token,
            attention_n_heads=heads,
            attention_dropout=0.1,
            ffn_d_hidden_multiplier=4 / 3,
            ffn_dropout=0.1,
            residual_dropout=0.0,
        ).to(self.device)

    def _forward(self, X):
        return self.model(X, None)

    def fit(self, X_train, y_train, X_val=None, y_val=None, progress_callback=None):
        _load_torch()
        started = time.time()
        classification = self.task_type not in ("regression", "time_series")
        y_train = np.asarray(y_train)
        self.classes = int(np.max(y_train)) + 1 if self.task_type == "multiclass_classification" else 2
        self.model = self._build(X_train.shape[1], self.classes)
        optimizer = optim.Adam(self.model.parameters(), lr=float(self.config.get("learning_rate", 1e-3)))
        criterion = nn.CrossEntropyLoss() if classification else nn.MSELoss()
        batch_size = int(self.config.get("batch_size", 1024))
        epochs = int(self.config.get("max_epochs", 50))
        patience = int(self.config.get("patience", 10))
        X_tensor = torch.tensor(np.asarray(X_train), dtype=torch.float32)
        y_dtype = torch.long if classification else torch.float32
        y_tensor = torch.tensor(y_train, dtype=y_dtype)
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)
        Xv = torch.tensor(np.asarray(X_val), dtype=torch.float32).to(self.device) if X_val is not None else None
        yv = torch.tensor(np.asarray(y_val), dtype=y_dtype).to(self.device) if y_val is not None else None
        best_loss, best_state, stale = float("inf"), None, 0
        self.training_history = []
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                output = self._forward(xb)
                if not classification:
                    loss = criterion(output.squeeze(-1), yb)
                else:
                    loss = criterion(output, yb)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(xb)
            train_loss = total_loss / len(X_train)
            val_loss = None
            if Xv is not None and yv is not None:
                self.model.eval()
                with torch.no_grad():
                    prediction = self._forward(Xv)
                    if not classification:
                        prediction = prediction.squeeze(-1)
                    val_loss = float(criterion(prediction, yv).item())
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
            self.training_history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
            if progress_callback:
                progress_callback(epoch + 1, epochs, train_loss, val_loss)
            if stale >= patience:
                break
        if best_state:
            self.model.load_state_dict(best_state)
        return {"train_duration_s": time.time() - started, "training_history": self.training_history}

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            logits = self._forward(torch.tensor(np.asarray(X), dtype=torch.float32).to(self.device))
        if self.task_type != "regression":
            return logits.argmax(dim=1).cpu().numpy()
        return logits.squeeze(-1).cpu().numpy()

    def predict_proba(self, X):
        if self.task_type in ("regression", "time_series"):
            return None
        self.model.eval()
        with torch.no_grad():
            logits = self._forward(torch.tensor(np.asarray(X), dtype=torch.float32).to(self.device))
        probabilities = torch.softmax(logits, dim=1)
        probabilities = probabilities.cpu().numpy()
        return probabilities

    def feature_importances(self):
        return None

    def save(self, path):
        training_device = self.device
        self.model.to(torch.device("cpu"))
        joblib.dump(self, path)
        self.model.to(training_device)

    @classmethod
    def load(cls, path):
        model = joblib.load(path)
        model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.model.to(model.device)
        return model


# ─────────────────────────────────────────────
#  Model registry / factory
# ─────────────────────────────────────────────

def get_available_models() -> List[Dict]:
    models = [
        {
            "id": "linear_regression",
            "name": "Linear Regression",
            "description": "Standard ordinary least squares regression for continuous targets and time-series forecasting.",
            "tasks": ["regression", "time_series"],
            "type": "sklearn",
            "params": [
                {"name": "fit_intercept", "label": "Fit Intercept", "type": "bool", "default": True},
            ],
        },
        {
            "id": "logistic_regression",
            "name": "Logistic Regression",
            "description": "Linear model for classification. Fast, interpretable, good baseline.",
            "tasks": ["binary_classification", "multiclass_classification"],
            "type": "sklearn",
            "params": [
                {"name": "C", "label": "Regularization (C)", "type": "float", "default": 1.0, "min": 0.001, "max": 100},
                {"name": "max_iter", "label": "Max Iterations", "type": "int", "default": 1000, "min": 100, "max": 5000},
            ],
        },
        {
            "id": "knn",
            "name": "K-Nearest Neighbors",
            "description": "Distance-based model for classification and regression.",
            "tasks": ["binary_classification", "multiclass_classification", "regression", "time_series"],
            "type": "sklearn",
            "params": [
                {"name": "n_neighbors", "label": "Neighbors", "type": "int", "default": 5, "min": 1, "max": 50},
                {"name": "weights", "label": "Weights (uniform or distance)", "type": "string", "default": "uniform"},
            ],
        },
        {
            "id": "random_forest",
            "name": "Random Forest",
            "description": "Ensemble of decision trees. Robust, handles mixed types well.",
            "tasks": ["binary_classification", "multiclass_classification", "regression", "time_series"],
            "type": "sklearn",
            "params": [
                {"name": "n_estimators", "label": "Number of Trees", "type": "int", "default": 100, "min": 10, "max": 500},
                {"name": "max_depth", "label": "Max Depth", "type": "int", "default": 10, "min": 1, "max": 50},
                {"name": "min_samples_split", "label": "Min Samples Split", "type": "int", "default": 2, "min": 2, "max": 20},
            ],
        },
        {
            "id": "gradient_boosting",
            "name": "Gradient Boosting",
            "description": "Sequential ensemble using gradient descent. Strong performer on tabular data.",
            "tasks": ["binary_classification", "multiclass_classification", "regression", "time_series"],
            "type": "sklearn",
            "params": [
                {"name": "n_estimators", "label": "Number of Estimators", "type": "int", "default": 100, "min": 10, "max": 500},
                {"name": "learning_rate", "label": "Learning Rate", "type": "float", "default": 0.1, "min": 0.001, "max": 1.0},
                {"name": "max_depth", "label": "Max Depth", "type": "int", "default": 3, "min": 1, "max": 10},
            ],
        },
        {
            "id": "pytorch_mlp",
            "name": "Neural Network (MLP)",
            "description": "Configurable multi-layer perceptron. Trained with PyTorch, supports early stopping.",
            "tasks": ["binary_classification", "multiclass_classification", "regression", "time_series"],
            "type": "pytorch",
            "params": [
                {"name": "hidden_layers", "label": "Hidden Layers (comma-separated sizes)", "type": "list_int", "default": [128, 64]},
                {"name": "learning_rate", "label": "Learning Rate", "type": "float", "default": 0.001, "min": 0.00001, "max": 0.1},
                {"name": "epochs", "label": "Max Epochs", "type": "int", "default": 50, "min": 5, "max": 200},
                {"name": "batch_size", "label": "Batch Size", "type": "int", "default": 64, "min": 16, "max": 512},
                {"name": "dropout", "label": "Dropout Rate", "type": "float", "default": 0.2, "min": 0.0, "max": 0.5},
                {"name": "patience", "label": "Early Stopping Patience", "type": "int", "default": 10, "min": 3, "max": 30},
            ],
        },
    ]
    if XGBOOST_AVAILABLE:
        models.append({
            "id": "xgboost",
            "name": "XGBoost",
            "description": "Extreme Gradient Boosting. State-of-the-art for tabular data.",
            "tasks": ["binary_classification", "multiclass_classification", "regression", "time_series"],
            "type": "xgboost",
            "params": [
                {"name": "n_estimators", "label": "Number of Estimators", "type": "int", "default": 100, "min": 10, "max": 500},
                {"name": "max_depth", "label": "Max Depth", "type": "int", "default": 6, "min": 1, "max": 15},
                {"name": "learning_rate", "label": "Learning Rate", "type": "float", "default": 0.1, "min": 0.001, "max": 1.0},
                {"name": "subsample", "label": "Subsample", "type": "float", "default": 0.8, "min": 0.1, "max": 1.0},
            ],
        })
    if not TORCH_AVAILABLE:
        models = [model for model in models if model["id"] != "pytorch_mlp"]
    else:
        torch_module = _load_torch()
        for model in models:
            if model["id"] == "pytorch_mlp":
                model["device"] = "CUDA" if torch_module.cuda.is_available() else "CPU"
        deep_tasks = ["binary_classification", "multiclass_classification", "regression", "time_series"]
        if TABNET_AVAILABLE:
            models.append({
                "id": "tabnet",
                "name": "TabNet",
                "description": "Attentive neural network for tabular data with validation-based early stopping.",
                "tasks": deep_tasks,
                "type": "pytorch",
                "device": "CUDA" if torch_module.cuda.is_available() else "CPU",
                "params": [
                    {"name": "n_d", "label": "Decision Dimension", "type": "int", "default": 8, "min": 4, "max": 64},
                    {"name": "n_a", "label": "Attention Dimension", "type": "int", "default": 8, "min": 4, "max": 64},
                    {"name": "n_steps", "label": "Steps", "type": "int", "default": 3, "min": 3, "max": 10},
                    {"name": "max_epochs", "label": "Max Epochs", "type": "int", "default": 50, "min": 5, "max": 300},
                    {"name": "batch_size", "label": "Batch Size", "type": "int", "default": 1024, "min": 128, "max": 8192},
                    {"name": "learning_rate", "label": "Learning Rate", "type": "float", "default": 0.02, "min": 0.0001, "max": 0.1},
                ],
            })
        if FT_TRANSFORMER_AVAILABLE:
            models.append({
                "id": "ft_transformer",
                "name": "FT-Transformer",
                "description": "Feature-tokenizing Transformer for tabular classification and regression.",
                "tasks": deep_tasks,
                "type": "pytorch",
                "device": "CUDA" if torch_module.cuda.is_available() else "CPU",
                "params": [
                    {"name": "d_token", "label": "Token Dimension", "type": "int", "default": 64, "min": 16, "max": 256},
                    {"name": "n_blocks", "label": "Transformer Blocks", "type": "int", "default": 3, "min": 1, "max": 12},
                    {"name": "attention_heads", "label": "Attention Heads", "type": "int", "default": 8, "min": 1, "max": 16},
                    {"name": "max_epochs", "label": "Max Epochs", "type": "int", "default": 50, "min": 5, "max": 300},
                    {"name": "batch_size", "label": "Batch Size", "type": "int", "default": 1024, "min": 128, "max": 8192},
                    {"name": "learning_rate", "label": "Learning Rate", "type": "float", "default": 0.001, "min": 0.0001, "max": 0.1},
                ],
            })

    from app.core.config import LIGHT_MODE
    if LIGHT_MODE:
        for m in models:
            if m["id"] in {"pytorch_mlp", "tabnet", "ft_transformer"}:
                m["disabled"] = True
                m["disabled_reason"] = "Not available in light mode"

    return models


def build_model(model_id: str, task_type: str, model_config: Dict[str, Any]):
    """Factory: return initialized model wrapper."""
    from app.core.config import LIGHT_MODE

    if LIGHT_MODE and model_id in {"pytorch_mlp", "tabnet", "ft_transformer"}:
        raise ValueError(f"Model '{model_id}' is not available in light mode.")

    from sklearn.ensemble import (
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    seed = model_config.get("random_seed", 42)

    if model_id == "linear_regression":
        fit_intercept = bool(model_config.get("fit_intercept", True))
        est = LinearRegression(fit_intercept=fit_intercept)
        return SklearnModelWrapper(est, task_type)

    elif model_id == "logistic_regression":
        est = LogisticRegression(
            C=float(model_config.get("C", 1.0)),
            max_iter=int(model_config.get("max_iter", 1000)),
            random_state=seed,
        )
        return SklearnModelWrapper(est, task_type)

    elif model_id == "knn":
        neighbors = int(model_config.get("n_neighbors", 5))
        weights = model_config.get("weights", "uniform")
        if weights not in ("uniform", "distance"):
            raise ValueError("KNN weights must be 'uniform' or 'distance'.")
        n_jobs = 1 if LIGHT_MODE else -1
        estimator = (
            KNeighborsRegressor(n_neighbors=neighbors, weights=weights, n_jobs=n_jobs)
            if task_type in ("regression", "time_series")
            else KNeighborsClassifier(n_neighbors=neighbors, weights=weights, n_jobs=n_jobs)
        )
        return SklearnModelWrapper(estimator, task_type)

    elif model_id == "random_forest":
        n_est = int(model_config.get("n_estimators", 100))
        max_d = int(model_config.get("max_depth", 10)) if model_config.get("max_depth") else None
        n_jobs = -1
        if LIGHT_MODE:
            n_est = min(100, n_est)
            max_d = 12 if max_d is None else min(12, max_d)
            n_jobs = 1
        if task_type in ("regression", "time_series"):
            est = RandomForestRegressor(n_estimators=n_est, max_depth=max_d,
                                        min_samples_split=int(model_config.get("min_samples_split", 2)),
                                        random_state=seed, n_jobs=n_jobs)
        else:
            est = RandomForestClassifier(n_estimators=n_est, max_depth=max_d,
                                          min_samples_split=int(model_config.get("min_samples_split", 2)),
                                          random_state=seed, n_jobs=n_jobs)
        return SklearnModelWrapper(est, task_type)

    elif model_id == "gradient_boosting":
        n_est = int(model_config.get("n_estimators", 100))
        if LIGHT_MODE:
            n_est = min(100, n_est)
        lr = float(model_config.get("learning_rate", 0.1))
        max_d = int(model_config.get("max_depth", 3))
        if task_type in ("regression", "time_series"):
            est = GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr,
                                             max_depth=max_d, random_state=seed)
        else:
            est = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr,
                                              max_depth=max_d, random_state=seed)
        return SklearnModelWrapper(est, task_type)

    elif model_id == "xgboost" and XGBOOST_AVAILABLE:
        import xgboost as xgb

        n_est = int(model_config.get("n_estimators", 100))
        n_jobs = 1 if LIGHT_MODE else -1
        if LIGHT_MODE:
            n_est = min(100, n_est)
        max_d = int(model_config.get("max_depth", 6))
        lr = float(model_config.get("learning_rate", 0.1))
        subsample = float(model_config.get("subsample", 0.8))
        if task_type == "binary_classification":
            est = xgb.XGBClassifier(n_estimators=n_est, max_depth=max_d, learning_rate=lr,
                                     subsample=subsample, random_state=seed,
                                     use_label_encoder=False, eval_metric="logloss", verbosity=0,
                                     n_jobs=n_jobs)
        elif task_type == "multiclass_classification":
            est = xgb.XGBClassifier(n_estimators=n_est, max_depth=max_d, learning_rate=lr,
                                     subsample=subsample, random_state=seed,
                                     objective="multi:softprob", verbosity=0,
                                     n_jobs=n_jobs)
        else:
            est = xgb.XGBRegressor(n_estimators=n_est, max_depth=max_d, learning_rate=lr,
                                    subsample=subsample, random_state=seed, verbosity=0,
                                    n_jobs=n_jobs)
        return SklearnModelWrapper(est, task_type)

    elif model_id == "pytorch_mlp":
        if not TORCH_AVAILABLE:
            raise RuntimeError("The MLP model requires torch. Install with: pip install torch")
        return PyTorchMLPWrapper(task_type, model_config)

    elif model_id == "tabnet" and TABNET_AVAILABLE:
        return TabNetModelWrapper(task_type, model_config)

    elif model_id == "ft_transformer" and FT_TRANSFORMER_AVAILABLE:
        return FTTransformerModelWrapper(task_type, model_config)

    else:
        raise ValueError(f"Unknown model: {model_id}")


def load_model_wrapper(path: str, model_type: str):
    if model_type == "pytorch_mlp":
        return PyTorchMLPWrapper.load(path)
    if model_type == "tabnet":
        if not TABNET_AVAILABLE:
            raise RuntimeError("This experiment requires pytorch-tabnet. Install it with: pip install pytorch-tabnet")
        return TabNetModelWrapper.load(path)
    if model_type == "ft_transformer":
        if not FT_TRANSFORMER_AVAILABLE:
            raise RuntimeError("This experiment requires rtdl-revisiting-models. Install it with: pip install rtdl-revisiting-models")
        return FTTransformerModelWrapper.load(path)
    return SklearnModelWrapper.load(path)
