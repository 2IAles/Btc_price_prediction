"""
Configuration globale du projet de prédiction du prix du Bitcoin.
Centralise tous les hyperparamètres et chemins pour faciliter l'expérimentation.
"""

import os

# ============================================================
# CHEMINS DU PROJET
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ============================================================
# PARAMÈTRES DES DONNÉES
# ============================================================
# Fenêtre temporelle : combien de jours passés utiliser pour prédire le suivant
SEQUENCE_LENGTH = 60  # 60 jours de contexte historique

# Ratio de découpe train/validation/test
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Features à utiliser (colonnes du dataset)
# Ces noms correspondent aux colonnes typiques d'un dataset OHLCV Bitcoin
FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    # Indicateurs techniques calculés dans le preprocessing
    "SMA_7", "SMA_21", "EMA_7", "EMA_21",
    "RSI_14", "MACD", "MACD_Signal",
    "BB_Upper", "BB_Lower",
    "ATR_14", "OBV",
    "Returns", "Log_Returns", "Volatility_21"
]

# Colonne cible à prédire
TARGET_COLUMN = "Close"

# ============================================================
# HYPERPARAMÈTRES COMMUNS
# ============================================================
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15
RANDOM_SEED = 42

# ============================================================
# HYPERPARAMÈTRES PAR MODÈLE
# ============================================================

# --- LSTM ---
LSTM_CONFIG = {
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "bidirectional": False,
}

# --- BiLSTM ---
BILSTM_CONFIG = {
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "bidirectional": True,  # Lecture dans les deux sens temporels
}

# --- GRU ---
GRU_CONFIG = {
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
}

# --- CNN-LSTM Hybride ---
CNN_LSTM_CONFIG = {
    "cnn_filters": 64,
    "cnn_kernel_size": 3,
    "lstm_hidden_size": 128,
    "lstm_num_layers": 1,
    "dropout": 0.2,
}

# --- Temporal Convolutional Network (TCN) ---
TCN_CONFIG = {
    "num_channels": [64, 64, 64, 64],  # Canaux par couche dilatée
    "kernel_size": 3,
    "dropout": 0.2,
}

# --- Transformer ---
TRANSFORMER_CONFIG = {
    "d_model": 128,        # Dimension du modèle
    "nhead": 8,            # Nombre de têtes d'attention
    "num_encoder_layers": 3,
    "dim_feedforward": 256,
    "dropout": 0.1,
}

# --- Temporal Fusion Transformer (simplifié) ---
TFT_CONFIG = {
    "hidden_size": 128,
    "lstm_layers": 1,
    "attention_heads": 4,
    "dropout": 0.1,
}

# --- XGBoost (baseline ML classique) ---
XGBOOST_CONFIG = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}

# --- ARIMA (baseline statistique) ---
ARIMA_CONFIG = {
    "order": (5, 1, 2),  # (p, d, q)
}
