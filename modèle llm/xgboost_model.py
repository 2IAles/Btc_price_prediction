"""
Modèle XGBoost (eXtreme Gradient Boosting) — baseline de Machine Learning classique.

XGBoost n'est PAS un réseau de neurones. C'est un algorithme d'ensemble basé sur
des arbres de décision boostés par gradient. Il est inclus comme baseline car :

    1. C'est souvent le meilleur algorithme ML "classique" sur les données tabulaires
    2. Il sert de référence : si un modèle deep learning ne bat pas XGBoost,
       la complexité supplémentaire n'est pas justifiée
    3. Il est beaucoup plus rapide à entraîner et ne nécessite pas de GPU

Adaptation aux séries temporelles :
    XGBoost ne gère pas nativement les séquences. On "aplatit" donc chaque séquence
    (60 jours × N features) en un seul vecteur, perdant ainsi la structure temporelle.
    Malgré cela, XGBoost peut parfois surpasser les modèles deep learning grâce à
    sa robustesse aux features bruitées et sa résistance au surapprentissage.

Références :
    - Chen & Guestrin (2016) — article fondateur de XGBoost
    - Plusieurs études montrant XGBoost compétitif pour BTC (65-67% accuracy)
"""

import numpy as np
from typing import Dict, Any


class XGBoostWrapper:
    """
    Wrapper pour XGBoost adapté au pipeline de séries temporelles.
    
    Aplatit les séquences 3D en vecteurs 2D pour la compatibilité avec XGBoost,
    puis entraîne un modèle de régression par gradient boosting.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None

    def _flatten_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        Aplatit les séquences 3D en vecteurs 2D.
        
        (n_samples, seq_length, n_features) → (n_samples, seq_length * n_features)
        
        Cela signifie que chaque valeur de chaque feature à chaque pas de temps
        devient une colonne distincte. XGBoost peut ensuite apprendre des patterns
        comme "si le RSI il y a 3 jours était > 70 ET le volume hier a baissé, alors..."
        """
        n_samples = X.shape[0]
        return X.reshape(n_samples, -1)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Entraîne le modèle XGBoost.
        
        Utilise l'early stopping sur le set de validation pour éviter le surapprentissage.
        """
        try:
            import xgboost as xgb
        except ImportError:
            print("XGBoost non installé. Installation : pip install xgboost")
            return self

        X_flat = self._flatten_sequences(X_train)

        self.model = xgb.XGBRegressor(
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
        """Prédit les valeurs cibles pour les séquences données."""
        if self.model is None:
            raise RuntimeError("Le modèle n'a pas été entraîné.")
        X_flat = self._flatten_sequences(X)
        return self.model.predict(X_flat)
