import sys
import pickle
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Permet d'importer les modules scripts_models depuis n'importe où
sys.path.insert(0, str(Path(__file__).parent))


class BTCPredictor:
    """Classe principale pour effectuer des prédictions de direction BTC.
    Encapsule le modèle entraîné, le scaler et la logique de feature engineering.
    """

    # Nombre de jours historiques nécessaires pour former une séquence d'inférence
    SEQUENCE_LENGTH = 60

    def __init__(
        self,
        model,
        scaler,
        feature_columns: list,
        model_name: str,
        threshold: float = 0.5,
    ):
        self._model = model
        self._scaler = scaler
        self._feature_columns = feature_columns
        self._model_name = model_name
        self._threshold = threshold  # seuil de décision : prob >= threshold → hausse

    @classmethod
    def load(
        cls,
        models_dir: Union[str, Path] = "models/",
        data_dir: Union[str, Path] = "data/",
    ) -> "BTCPredictor":
        """Charge le meilleur modèle, le scaler et les colonnes depuis les fichiers sauvegardés."""
        models_dir = Path(models_dir)
        data_dir = Path(data_dir)

        # Charge le scaler ajusté pendant le preprocessing
        scaler_path = models_dir / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler introuvable : {scaler_path}\n"
                "Exécutez 02_preprocessing.py pour le générer."
            )
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        # Récupère la liste des features depuis X_train pour garantir la cohérence
        x_train_path = data_dir / "X_train.pkl"
        if not x_train_path.exists():
            raise FileNotFoundError(
                f"X_train.pkl introuvable : {x_train_path}\n"
                "Exécutez 02_preprocessing.py pour le générer."
            )
        with open(x_train_path, "rb") as f:
            x_train = pickle.load(f)
        feature_columns = list(x_train.columns)

        # Charge les métadonnées du meilleur modèle (nom, seuil, architecture)
        best_path = models_dir / "best_model.pkl"
        if not best_path.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {best_path}\n"
                "Exécutez 03_models.ipynb puis 04_results_analysis.ipynb."
            )
        with open(best_path, "rb") as f:
            saved = pickle.load(f)

        # Cas PyTorch : le fichier .pkl contient les métadonnées, les poids sont dans .pt
        if isinstance(saved, dict) and "name" in saved:
            meta = saved
            model_name = meta["name"]
            threshold = meta.get("threshold", 0.5)
            n_features = meta.get("n_features", len(feature_columns))
            model = cls._load_pytorch_model(model_name, n_features, models_dir)

        else:
            raise ValueError(f"Format inattendu dans best_model.pkl : {type(saved)}")

        print(f"Modèle chargé : {model_name}  |  seuil : {threshold:.2f}")
        return cls(model, scaler, feature_columns, model_name, threshold)

    def predict(self, df: pd.DataFrame) -> dict:
        """Retourne la direction prédite (0=baisse, 1=hausse) et la probabilité associée."""
        prob = self.predict_proba(df)
        direction = int(prob >= self._threshold)
        return {
            "direction": direction,
            "probability": round(float(prob), 4),
            "horizon": "1d",
            "model": self._model_name,
            "threshold": self._threshold,
        }

    def predict_proba(self, df: pd.DataFrame) -> float:
        """Retourne la probabilité brute de hausse (entre 0 et 1)."""
        seq = self._prepare_sequence(df)
        probs = self._model.predict(seq)
        return float(np.atleast_1d(probs)[0])

    def _prepare_sequence(self, df: pd.DataFrame) -> np.ndarray:
        """Construit la séquence d'entrée normalisée à partir du DataFrame brut."""
        features = self._extract_features(df)
        # Garantit que les colonnes sont dans le même ordre qu'à l'entraînement
        features = features[self._feature_columns]

        if len(features) < self.SEQUENCE_LENGTH:
            raise ValueError(
                f"Pas assez de données : {len(features)} lignes disponibles, "
                f"{self.SEQUENCE_LENGTH} requises."
            )
        # Prend les 60 derniers jours disponibles
        window = features.iloc[-self.SEQUENCE_LENGTH :].copy()

        # Normalise avec le scaler ajusté sur le train (même échelle qu'à l'entraînement)
        window_scaled = self._scaler.transform(window)

        # Ajoute la dimension batch : (1, 60, n_features)
        return window_scaled[np.newaxis, :, :].astype(np.float32)

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recalcule les features exactement comme dans 02_preprocessing.py."""
        out = pd.DataFrame(index=df.index)

        # Correspondance entre le nom interne de la feature et le nom de colonne dans df
        assets = {
            "btc":    "close_btc",
            "gold":   "close_xau",
            "eth":    "close_eth",
            "sp500":  "close_snp500",
            "dxy":    "close_dxy",
            "vix":    "close_vix",
            "us10y":  "close_us10y",
            "oil":    "close_oil",
            "silver": "close_silver",
        }

        for key, col in assets.items():
            if col not in df.columns:
                continue
            s = df[col]
            # Rendement log journalier, décalé d'un jour (anti-leakage)
            ret = np.log(s / s.shift(1)).shift(1)
            out[f"ret_1d_{key}"] = ret
            # Rendements sur plusieurs fenêtres
            for w in [3, 7, 14, 30]:
                out[f"ret_{w}d_{key}"] = np.log(s / s.shift(w)).shift(1)
            # Volatilité roulante
            for w in [7, 30]:
                out[f"vol_{w}d_{key}"] = ret.rolling(w).std()

        # Momentum BTC uniquement (close / SMA)
        if "close_btc" in df.columns:
            s = df["close_btc"]
            for w in [7, 30]:
                sma = s.rolling(w).mean()
                out[f"momentum_{w}d_btc"] = (s / sma).shift(1)

        # Ratio volume : détecte les pics d'activité
        if "volume_btc" in df.columns:
            vol = df["volume_btc"]
            out["vol_ratio_7d_btc"] = (vol / vol.rolling(7).mean()).shift(1)

        # Encodage cyclique du jour de la semaine
        dow = df.index.dayofweek
        out["dow_sin"] = np.sin(dow * 2 * np.pi / 7)
        out["dow_cos"] = np.cos(dow * 2 * np.pi / 7)

        # Indicateurs on-chain et macro (décalés d'un jour pour éviter le leakage)
        for col in [
            "fedfunds",
            "funding_rate",
            "hashrate",
            "mvrv",
            "nupl",
            "google_trends",
        ]:
            if col in df.columns:
                out[col] = df[col].shift(1)

        # Nettoyage : remplace les infinis, propage les dernières valeurs, supprime les NaN
        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.ffill()
        out = out.dropna()
        return out

    @staticmethod
    def _load_pytorch_model(model_name: str, n_features: int, models_dir: Path):
        """Instancie l'architecture PyTorch et charge les poids depuis best_model.pt."""
        import torch
        from scripts_models.config import (
            LSTM_CONFIG,
            GRU_CONFIG,
            CNN_LSTM_CONFIG,
            TRANSFORMER_CONFIG,
            TFT_CONFIG,
        )
        from scripts_models.trainer import Trainer

        # Sélectionne et instancie la bonne architecture selon le nom du modèle
        name = model_name.lower()
        if name == "lstm":
            from scripts_models.lstm_model import LSTMModel

            model = LSTMModel(
                n_features,
                LSTM_CONFIG["hidden_size"],
                LSTM_CONFIG["num_layers"],
                LSTM_CONFIG["dropout"],
            )
        elif name == "bilstm":
            from scripts_models.lstm_model import LSTMModel

            model = LSTMModel(
                n_features,
                LSTM_CONFIG["hidden_size"],
                LSTM_CONFIG["num_layers"],
                LSTM_CONFIG["dropout"],
                bidirectional=True,
            )
        elif name == "gru":
            from scripts_models.gru_model import GRUModel

            model = GRUModel(
                n_features,
                GRU_CONFIG["hidden_size"],
                GRU_CONFIG["num_layers"],
                GRU_CONFIG["dropout"],
            )
        elif name == "cnn-lstm":
            from scripts_models.cnn_lstm_model import CNNLSTMModel

            model = CNNLSTMModel(
                n_features,
                CNN_LSTM_CONFIG["cnn_filters"],
                CNN_LSTM_CONFIG["cnn_kernel_size"],
                CNN_LSTM_CONFIG["lstm_hidden_size"],
                CNN_LSTM_CONFIG["lstm_num_layers"],
                CNN_LSTM_CONFIG["dropout"],
            )
        elif name == "transformer":
            from scripts_models.transformer_model import TransformerModel

            model = TransformerModel(
                n_features,
                TRANSFORMER_CONFIG["d_model"],
                TRANSFORMER_CONFIG["nhead"],
                TRANSFORMER_CONFIG["num_encoder_layers"],
                TRANSFORMER_CONFIG["dim_feedforward"],
                TRANSFORMER_CONFIG["dropout"],
            )
        elif name == "tft":
            from scripts_models.tft_model import SimplifiedTFT

            model = SimplifiedTFT(
                n_features,
                TFT_CONFIG["hidden_size"],
                TFT_CONFIG["lstm_layers"],
                TFT_CONFIG["attention_heads"],
                TFT_CONFIG["dropout"],
            )
        else:
            raise ValueError(f"Modèle inconnu : {model_name}")

        # Charge les poids entraînés dans l'architecture vide
        weights_path = models_dir / "best_model.pt"
        model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
        model.eval()  # désactive dropout pour l'inférence

        # Wrappé dans Trainer pour utiliser sa méthode predict() avec sigmoid
        trainer = Trainer(model=model)
        return trainer

    def __repr__(self) -> str:
        return (
            f"BTCPredictor(model={self._model_name}, "
            f"features={len(self._feature_columns)}, "
            f"threshold={self._threshold:.2f})"
        )
