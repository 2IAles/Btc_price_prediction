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

# Mode classification binaire (label_dir_1d : 0 = baisse, 1 = hausse)
CLASSIFICATION = True

# Splits temporels fixes (pas de ratio aléatoire — données de séries temporelles)
TRAIN_END = "2022-12-31"
VAL_END   = "2023-12-31"
# Test : tout ce qui est > VAL_END

# Colonne cible
TARGET_COLUMN = "label_dir_1d"

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
