import numpy as np
from typing import Dict, Any


class XGBoostWrapper:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None

    def _flatten_sequences(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        return X.reshape(n_samples, -1)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ):
        try:
            import xgboost as xgb
        except ImportError:
            print("erreur xgboost")
            return self

        X_flat = self._flatten_sequences(X_train)

        self.model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=self.config.get("n_estimators", 500),
            max_depth=self.config.get("max_depth", 6),
            learning_rate=self.config.get("learning_rate", 0.05),
            subsample=self.config.get("subsample", 0.8),
            colsample_bytree=self.config.get("colsample_bytree", 0.8),
            reg_alpha=self.config.get("reg_alpha", 0.1),
            reg_lambda=self.config.get("reg_lambda", 1.0),
            random_state=42,
            verbosity=0,
        )

        fit_params = {}
        if X_val is not None:
            X_val_flat = self._flatten_sequences(X_val)
            fit_params["eval_set"] = [(X_val_flat, y_val)]

        self.model.fit(X_flat, y_train, **fit_params)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Le modèle n'a pas été entraîné.")
        X_flat = self._flatten_sequences(X)
        return self.model.predict_proba(X_flat)[:, 1]
