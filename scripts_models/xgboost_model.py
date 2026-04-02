import numpy as np
from typing import Dict, Any


class XGBoostWrapper:
    """Wrapper autour de XGBoost pour l'adapter à l'interface du projet.
    XGBoost est un modèle à base d'arbres de décision boostés par gradient.
    Il ne traite pas les séquences : les données sont d'abord "aplaties".
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None

    def _flatten_sequences(self, X: np.ndarray) -> np.ndarray:
        # Transforme (n_samples, seq_len, n_features) → (n_samples, seq_len * n_features)
        # XGBoost ne comprend pas les données 3D, on aplatit la séquence en vecteur
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

        # Aplatissement de la séquence 3D en matrice 2D
        X_flat = self._flatten_sequences(X_train)

        self.model = xgb.XGBClassifier(
            objective="binary:logistic",   # sortie : probabilité entre 0 et 1
            eval_metric="logloss",
            n_estimators=self.config.get("n_estimators", 500),  # nombre d'arbres
            max_depth=self.config.get("max_depth", 6),           # profondeur max de chaque arbre
            learning_rate=self.config.get("learning_rate", 0.05),
            subsample=self.config.get("subsample", 0.8),          # fraction d'échantillons par arbre
            colsample_bytree=self.config.get("colsample_bytree", 0.8),  # fraction de features par arbre
            reg_alpha=self.config.get("reg_alpha", 0.1),          # régularisation L1
            reg_lambda=self.config.get("reg_lambda", 1.0),         # régularisation L2
            random_state=42,
            verbosity=0,
        )

        fit_params = {}
        if X_val is not None:
            # Passe un jeu de validation pour surveiller la loss pendant l'entraînement
            X_val_flat = self._flatten_sequences(X_val)
            fit_params["eval_set"] = [(X_val_flat, y_val)]

        self.model.fit(X_flat, y_train, **fit_params)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Le modèle n'a pas été entraîné.")
        X_flat = self._flatten_sequences(X)
        # predict_proba retourne [[P(classe 0), P(classe 1)]], on prend la colonne 1
        return self.model.predict_proba(X_flat)[:, 1]
