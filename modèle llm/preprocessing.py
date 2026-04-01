"""
Module de preprocessing des données Bitcoin.
Gère le calcul des indicateurs techniques, la normalisation,
et la création des séquences temporelles pour l'entraînement.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule l'ensemble des indicateurs techniques à partir des données OHLCV brutes.
    
    Ces indicateurs servent de features supplémentaires pour capturer :
    - Les tendances (SMA, EMA)
    - Le momentum (RSI, MACD)  
    - La volatilité (Bandes de Bollinger, ATR)
    - Le volume (OBV)
    
    Args:
        df: DataFrame avec colonnes Open, High, Low, Close, Volume
    
    Returns:
        DataFrame enrichi avec tous les indicateurs techniques
    """
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # --- Moyennes Mobiles ---
    # SMA : moyenne simple sur N jours (lisse le bruit, montre la tendance)
    df["SMA_7"] = close.rolling(window=7).mean()
    df["SMA_21"] = close.rolling(window=21).mean()

    # EMA : moyenne exponentielle (réagit plus vite aux changements récents)
    df["EMA_7"] = close.ewm(span=7, adjust=False).mean()
    df["EMA_21"] = close.ewm(span=21, adjust=False).mean()

    # --- RSI (Relative Strength Index) ---
    # Mesure la force relative du mouvement haussier vs baissier sur 14 jours
    # RSI > 70 = surachat, RSI < 30 = survente
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)  # Évite la division par zéro
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # --- MACD (Moving Average Convergence Divergence) ---
    # Croisement de deux EMA pour détecter les changements de momentum
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # --- Bandes de Bollinger ---
    # Enveloppe de volatilité autour de la SMA 21
    bb_sma = close.rolling(window=21).mean()
    bb_std = close.rolling(window=21).std()
    df["BB_Upper"] = bb_sma + 2 * bb_std
    df["BB_Lower"] = bb_sma - 2 * bb_std

    # --- ATR (Average True Range) ---
    # Mesure la volatilité moyenne sur 14 jours
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR_14"] = true_range.rolling(window=14).mean()

    # --- OBV (On-Balance Volume) ---
    # Volume cumulé pondéré par la direction du prix
    obv = np.where(close > close.shift(1), volume, 
                   np.where(close < close.shift(1), -volume, 0))
    df["OBV"] = pd.Series(obv, index=df.index).cumsum()

    # --- Rendements ---
    df["Returns"] = close.pct_change()
    df["Log_Returns"] = np.log(close / close.shift(1))
    df["Volatility_21"] = df["Returns"].rolling(window=21).std()

    # Supprime les lignes avec des NaN créées par les calculs rolling
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def generate_synthetic_btc_data(n_days: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Génère des données synthétiques réalistes de prix Bitcoin pour le développement.
    
    Simule un processus stochastique avec :
    - Tendance haussière à long terme (drift)
    - Volatilité stochastique (clustering de volatilité type GARCH)
    - Cycles de marché (bull/bear)
    - Structure OHLCV réaliste
    
    Args:
        n_days: Nombre de jours à générer
        seed: Graine aléatoire pour reproductibilité
    
    Returns:
        DataFrame avec colonnes Date, Open, High, Low, Close, Volume
    """
    np.random.seed(seed)

    # Prix initial et paramètres du processus
    price = 10000.0
    prices_close = []
    prices_open = []
    prices_high = []
    prices_low = []
    volumes = []

    # Paramètres de la volatilité stochastique
    base_vol = 0.03  # Volatilité journalière de base (~3%)
    vol = base_vol

    for i in range(n_days):
        # Cycle de marché sinusoïdal (période ~365 jours)
        cycle = 0.0003 * np.sin(2 * np.pi * i / 365)

        # Mise à jour de la volatilité (effet GARCH simplifié)
        vol = 0.94 * vol + 0.06 * base_vol + 0.02 * abs(np.random.randn()) * base_vol

        # Rendement journalier : drift + cycle + bruit
        drift = 0.0002  # Tendance haussière légère
        daily_return = drift + cycle + vol * np.random.randn()

        # Structure OHLCV
        open_price = price * (1 + np.random.randn() * 0.002)
        close_price = price * (1 + daily_return)
        high_price = max(open_price, close_price) * (1 + abs(np.random.randn()) * vol * 0.5)
        low_price = min(open_price, close_price) * (1 - abs(np.random.randn()) * vol * 0.5)

        # Volume corrélé à la volatilité (plus de volume quand ça bouge)
        base_volume = 1e9
        vol_multiplier = 1 + 5 * (vol / base_vol - 1)
        volume = base_volume * vol_multiplier * (0.5 + np.random.rand())

        prices_open.append(open_price)
        prices_high.append(high_price)
        prices_low.append(low_price)
        prices_close.append(close_price)
        volumes.append(volume)

        price = close_price

    dates = pd.date_range(start="2019-01-01", periods=n_days, freq="D")
    df = pd.DataFrame({
        "Date": dates,
        "Open": prices_open,
        "High": prices_high,
        "Low": prices_low,
        "Close": prices_close,
        "Volume": volumes,
    })

    return df


class TimeSeriesScaler:
    """
    Normalisateur Min-Max adapté aux séries temporelles financières.
    
    Important : on fit UNIQUEMENT sur les données d'entraînement pour éviter
    le data leakage (fuite d'information du futur vers le passé).
    """
    
    def __init__(self):
        self.min_vals = None
        self.max_vals = None
        self.feature_names = None

    def fit(self, data: np.ndarray, feature_names: Optional[List[str]] = None):
        """Apprend les paramètres de normalisation sur les données d'entraînement."""
        self.min_vals = data.min(axis=0)
        self.max_vals = data.max(axis=0)
        self.feature_names = feature_names
        # Évite la division par zéro si une feature est constante
        self.range_vals = self.max_vals - self.min_vals
        self.range_vals[self.range_vals == 0] = 1.0
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Normalise les données dans [0, 1]."""
        return (data - self.min_vals) / self.range_vals

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Dé-normalise les données vers l'échelle originale."""
        return data * self.range_vals + self.min_vals

    def inverse_transform_column(self, data: np.ndarray, col_idx: int) -> np.ndarray:
        """Dé-normalise une seule colonne (utile pour la colonne cible)."""
        return data * self.range_vals[col_idx] + self.min_vals[col_idx]


def create_sequences(
    data: np.ndarray,
    target_col_idx: int,
    seq_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crée des paires (séquence d'entrée, valeur cible) pour l'apprentissage supervisé.
    
    Pour chaque position t, on prend les `seq_length` pas de temps précédents
    comme entrée, et la valeur cible au pas t comme sortie.
    
    Schéma : [t-59, t-58, ..., t-1, t] → prédiction pour t+1
    
    Args:
        data: Données normalisées (n_samples, n_features)
        target_col_idx: Index de la colonne cible dans les features
        seq_length: Longueur de la fenêtre d'entrée
    
    Returns:
        X: Séquences d'entrée (n_sequences, seq_length, n_features)
        y: Valeurs cibles (n_sequences,)
    """
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(data[i, target_col_idx])
    return np.array(X), np.array(y)


def prepare_data_splits(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    seq_length: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> dict:
    """
    Pipeline complet de préparation des données : features → normalisation → séquences → splits.
    
    ATTENTION : Le split est chronologique (pas aléatoire) car c'est une série temporelle.
    Un split aléatoire causerait du data leakage en mélangeant futur et passé.
    
    Args:
        df: DataFrame avec indicateurs techniques déjà calculés
        feature_columns: Liste des colonnes à utiliser comme features
        target_column: Nom de la colonne cible
        seq_length: Longueur des séquences d'entrée
        train_ratio: Proportion des données pour l'entraînement
        val_ratio: Proportion pour la validation
    
    Returns:
        Dictionnaire avec tous les splits et le scaler
    """
    # Sélection et validation des colonnes
    available_cols = [c for c in feature_columns if c in df.columns]
    data = df[available_cols].values
    target_col_idx = available_cols.index(target_column)

    # Split chronologique
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # Normalisation (fit uniquement sur train pour éviter le leakage)
    scaler = TimeSeriesScaler()
    scaler.fit(train_data, feature_names=available_cols)

    train_scaled = scaler.transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    # Création des séquences
    X_train, y_train = create_sequences(train_scaled, target_col_idx, seq_length)
    X_val, y_val = create_sequences(val_scaled, target_col_idx, seq_length)
    X_test, y_test = create_sequences(test_scaled, target_col_idx, seq_length)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler,
        "target_col_idx": target_col_idx,
        "feature_names": available_cols,
        "n_features": len(available_cols),
    }
