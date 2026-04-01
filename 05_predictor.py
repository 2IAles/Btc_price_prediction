"""
BTCPredictor — Pipeline de prédiction de direction journalière du Bitcoin.

Utilisation :
    from predictor import BTCPredictor
    p = BTCPredictor.load("models/")
    result = p.predict(df)
    # → {"direction": 1, "probability": 0.67, "horizon": "1d"}

Input attendu (df) :
    DataFrame avec au minimum les colonnes OHLCV pour BTC + actifs macro sur
    les 60 derniers jours. Le preprocessing (features engineering + scaling)
    est appliqué automatiquement.

Prérequis :
    - 03_models.ipynb et 04_results_analysis.ipynb doivent avoir été exécutés
      (génèrent models/best_model.pkl et models/scaler.pkl).
    - 02_preprocessing.py doit avoir été exécuté (génère data/X_train.pkl
      pour connaître les colonnes attendues).
"""

import sys
import pickle
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ajouter le dossier des modèles au path Python
sys.path.insert(0, str(Path(__file__).parent / "modèle llm"))


class BTCPredictor:
    """Prédicteur de direction journalière Bitcoin (J+1 : 0 = baisse, 1 = hausse)."""

    SEQUENCE_LENGTH = 60  # doit correspondre à config.SEQUENCE_LENGTH

    def __init__(self, model, scaler, feature_columns: list, model_name: str,
                 threshold: float = 0.5):
        self._model          = model       # Trainer PyTorch ou XGBoostWrapper
        self._scaler         = scaler      # RobustScaler fitté sur X_train
        self._feature_columns = feature_columns
        self._model_name     = model_name
        self._threshold      = threshold

    # ──────────────────────────────────────────────────────────────────────────
    # Chargement
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, models_dir: Union[str, Path] = "models/",
             data_dir: Union[str, Path] = "data/") -> "BTCPredictor":
        """
        Charge le meilleur modèle et le scaler depuis les répertoires indiqués.

        Args:
            models_dir : répertoire contenant best_model.pkl (et best_model.pt
                         pour les modèles PyTorch).
            data_dir   : répertoire contenant X_train.pkl (pour les noms de
                         colonnes) et scaler.pkl.
        """
        models_dir = Path(models_dir)
        data_dir   = Path(data_dir)

        # ── Scaler ────────────────────────────────────────────────────────────
        scaler_path = models_dir / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler introuvable : {scaler_path}\n"
                "Exécutez 02_preprocessing.py pour le générer."
            )
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        # ── Colonnes attendues ────────────────────────────────────────────────
        x_train_path = data_dir / "X_train.pkl"
        if not x_train_path.exists():
            raise FileNotFoundError(
                f"X_train.pkl introuvable : {x_train_path}\n"
                "Exécutez 02_preprocessing.py pour le générer."
            )
        with open(x_train_path, "rb") as f:
            x_train = pickle.load(f)
        feature_columns = list(x_train.columns)

        # ── Modèle ────────────────────────────────────────────────────────────
        best_path = models_dir / "best_model.pkl"
        if not best_path.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {best_path}\n"
                "Exécutez 03_models.ipynb puis 04_results_analysis.ipynb."
            )
        with open(best_path, "rb") as f:
            saved = pickle.load(f)

        from xgboost_model import XGBoostWrapper

        # Cas XGBoost : best_model.pkl contient {"model": wrapper, "meta": {...}}
        if isinstance(saved, dict) and "model" in saved:
            model      = saved["model"]
            meta       = saved.get("meta", {})
            model_name = meta.get("name", "XGBoost")
            threshold  = meta.get("threshold", 0.5)

        # Cas PyTorch : best_model.pkl contient {"name": ..., "threshold": ..., "n_features": ...}
        elif isinstance(saved, dict) and "name" in saved:
            meta       = saved
            model_name = meta["name"]
            threshold  = meta.get("threshold", 0.5)
            n_features = meta.get("n_features", len(feature_columns))
            model      = cls._load_pytorch_model(model_name, n_features, models_dir)

        else:
            raise ValueError(f"Format inattendu dans best_model.pkl : {type(saved)}")

        print(f"Modèle chargé : {model_name}  |  seuil : {threshold:.2f}")
        return cls(model, scaler, feature_columns, model_name, threshold)

    # ──────────────────────────────────────────────────────────────────────────
    # Prédiction
    # ──────────────────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Prédit la direction de BTC pour le lendemain.

        Args:
            df : DataFrame brut avec les colonnes sources (au moins les mêmes
                 que celles utilisées lors du preprocessing). Doit contenir
                 au minimum SEQUENCE_LENGTH + 1 lignes.

        Returns:
            {"direction": 0|1, "probability": float, "horizon": "1d"}
        """
        prob = self.predict_proba(df)
        direction = int(prob >= self._threshold)
        return {
            "direction":   direction,
            "probability": round(float(prob), 4),
            "horizon":     "1d",
            "model":       self._model_name,
            "threshold":   self._threshold,
        }

    def predict_proba(self, df: pd.DataFrame) -> float:
        """Retourne la probabilité brute P(hausse J+1) ∈ [0, 1]."""
        seq = self._prepare_sequence(df)
        probs = self._model.predict(seq)
        return float(probs[-1])

    # ──────────────────────────────────────────────────────────────────────────
    # Preprocessing interne
    # ──────────────────────────────────────────────────────────────────────────

    def _prepare_sequence(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforme le DataFrame brut en une séquence (1, SEQ_LEN, n_features)
        prête pour l'inférence.
        """
        features = self._extract_features(df)
        features = features[self._feature_columns]

        if len(features) < self.SEQUENCE_LENGTH:
            raise ValueError(
                f"Pas assez de données : {len(features)} lignes disponibles, "
                f"{self.SEQUENCE_LENGTH} requises."
            )

        # Prendre les SEQ_LEN dernières lignes
        window = features.iloc[-self.SEQUENCE_LENGTH:].copy()

        # Appliquer le scaler (même transformation que Phase 2)
        window_scaled = self._scaler.transform(window)

        # Shape (1, SEQ_LEN, n_features)
        return window_scaled[np.newaxis, :, :].astype(np.float32)

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reproduit le feature engineering de 02_preprocessing.py.

        Le DataFrame d'entrée doit contenir au minimum :
          close_btc, (close_xau, close_eth, close_snp500, close_dxy, close_vix,
          close_us10y, close_oil, close_silver — optionnels)
          + fedfunds, funding_rate, hashrate, mvrv, nupl, google_trends — optionnels

        Toutes les features sont décalées de 1 jour (shift(1)) pour éviter
        le data leakage.
        """
        out = pd.DataFrame(index=df.index)

        assets = {
            "btc":     "close_btc",
            "xau":     "close_xau",
            "eth":     "close_eth",
            "snp500":  "close_snp500",
            "dxy":     "close_dxy",
            "vix":     "close_vix",
            "us10y":   "close_us10y",
            "oil":     "close_oil",
            "silver":  "close_silver",
        }

        for key, col in assets.items():
            if col not in df.columns:
                continue
            s = df[col]
            # Rendement log J-1
            ret = np.log(s / s.shift(1)).shift(1)
            out[f"ret_1d_{key}"] = ret
            # Rendements multi-fenêtres
            for w in [3, 7, 14, 30]:
                out[f"ret_{w}d_{key}"] = np.log(s / s.shift(w)).shift(1)
            # Volatilité roulante
            for w in [7, 30]:
                out[f"vol_{w}d_{key}"] = ret.rolling(w).std()
            # Momentum
            for w in [7, 30]:
                sma = s.rolling(w).mean()
                out[f"mom_{w}d_{key}"] = (s / sma).shift(1)

        # Volume BTC
        if "volume_btc" in df.columns:
            vol = df["volume_btc"]
            out["vol_ratio_7d_btc"] = (vol / vol.rolling(7).mean()).shift(1)

        # Encodage cyclique jour de semaine
        dow = df.index.dayofweek
        out["dow_sin"] = np.sin(dow * 2 * np.pi / 7)
        out["dow_cos"] = np.cos(dow * 2 * np.pi / 7)

        # Indicateurs BTC-spécifiques (shift(1) déjà fait lors de la collecte)
        for col in ["fedfunds", "funding_rate", "hashrate", "mvrv", "nupl", "google_trends"]:
            if col in df.columns:
                out[col] = df[col].shift(1)

        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.ffill()
        out = out.dropna()
        return out

    # ──────────────────────────────────────────────────────────────────────────
    # Utilitaires
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_pytorch_model(model_name: str, n_features: int,
                            models_dir: Path):
        """Reconstruit et charge les poids d'un modèle PyTorch."""
        import torch
        from config import (LSTM_CONFIG, GRU_CONFIG, CNN_LSTM_CONFIG,
                            TRANSFORMER_CONFIG, TFT_CONFIG)
        from trainer import Trainer

        name = model_name.lower()
        if name == "lstm":
            from lstm_model import LSTMModel
            model = LSTMModel(n_features, LSTM_CONFIG["hidden_size"],
                              LSTM_CONFIG["num_layers"], LSTM_CONFIG["dropout"])
        elif name == "bilstm":
            from lstm_model import LSTMModel
            model = LSTMModel(n_features, LSTM_CONFIG["hidden_size"],
                              LSTM_CONFIG["num_layers"], LSTM_CONFIG["dropout"],
                              bidirectional=True)
        elif name == "gru":
            from gru_model import GRUModel
            model = GRUModel(n_features, GRU_CONFIG["hidden_size"],
                             GRU_CONFIG["num_layers"], GRU_CONFIG["dropout"])
        elif name == "cnn-lstm":
            from cnn_lstm_model import CNNLSTMModel
            model = CNNLSTMModel(n_features, CNN_LSTM_CONFIG["cnn_filters"],
                                 CNN_LSTM_CONFIG["cnn_kernel_size"],
                                 CNN_LSTM_CONFIG["lstm_hidden_size"],
                                 CNN_LSTM_CONFIG["lstm_num_layers"],
                                 CNN_LSTM_CONFIG["dropout"])
        elif name == "transformer":
            from transformer_model import TransformerModel
            model = TransformerModel(n_features, TRANSFORMER_CONFIG["d_model"],
                                     TRANSFORMER_CONFIG["nhead"],
                                     TRANSFORMER_CONFIG["num_encoder_layers"],
                                     TRANSFORMER_CONFIG["dim_feedforward"],
                                     TRANSFORMER_CONFIG["dropout"])
        elif name == "tft":
            from tft_model import SimplifiedTFT
            model = SimplifiedTFT(n_features, TFT_CONFIG["hidden_size"],
                                  TFT_CONFIG["lstm_layers"],
                                  TFT_CONFIG["attention_heads"],
                                  TFT_CONFIG["dropout"])
        else:
            raise ValueError(f"Modèle inconnu : {model_name}")

        weights_path = models_dir / "best_model.pt"
        model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
        model.eval()

        trainer = Trainer(model=model)
        return trainer

    def __repr__(self) -> str:
        return (f"BTCPredictor(model={self._model_name}, "
                f"features={len(self._feature_columns)}, "
                f"threshold={self._threshold:.2f})")
