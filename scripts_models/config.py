import os

# Chemins vers les dossiers principaux du projet
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


# Nombre de jours passés fournis au modèle pour prédire le suivant
SEQUENCE_LENGTH = 60

# True = classification (hausse/baisse), False = régression (valeur continue)
CLASSIFICATION = True

# Dates de coupure pour les splits train / validation / test
TRAIN_END = "2022-12-31"
VAL_END = "2023-12-31"

# Nom de la colonne cible dans le dataset
TARGET_COLUMN = "label_dir_1d"

# Hyperparamètres d'entraînement communs à tous les modèles
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.0003
EARLY_STOPPING_PATIENCE = 25  # arrêt si pas d'amélioration pendant N époques
RANDOM_SEED = 42


# Hyperparamètres spécifiques au LSTM (unidirectionnel)
LSTM_CONFIG = {
    "hidden_size": 64,    # taille de l'état caché
    "num_layers": 1,      # nombre de couches LSTM empilées
    "dropout": 0.3,
    "bidirectional": False,
}

# Même architecture mais lit la séquence dans les deux sens
BILSTM_CONFIG = {
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": 0.3,
    "bidirectional": True,
}

# Hyperparamètres du GRU (variante plus simple du LSTM)
GRU_CONFIG = {
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": 0.3,
}

# CNN extrait des patterns locaux, puis LSTM capture la dynamique temporelle
CNN_LSTM_CONFIG = {
    "cnn_filters": 32,
    "cnn_kernel_size": 3,
    "lstm_hidden_size": 64,
    "lstm_num_layers": 1,
    "dropout": 0.3,
}

# TCN : convolutions causales dilatées (champ récepteur exponentiel)
TCN_CONFIG = {
    "num_channels": [32, 32, 32],  # canaux par niveau de dilatation
    "kernel_size": 3,
    "dropout": 0.3,
}

# Transformer : attention multi-têtes, pas de récurrence
TRANSFORMER_CONFIG = {
    "d_model": 64,            # dimension des embeddings internes
    "nhead": 4,               # nombre de têtes d'attention
    "num_encoder_layers": 2,
    "dim_feedforward": 128,
    "dropout": 0.2,
}

# TFT (Temporal Fusion Transformer) : sélection de variables + attention
TFT_CONFIG = {
    "hidden_size": 64,
    "lstm_layers": 1,
    "attention_heads": 4,
    "dropout": 0.2,
}

# ARIMA : modèle statistique de référence (ordre p, d, q)
ARIMA_CONFIG = {
    "order": (5, 1, 2),
}
