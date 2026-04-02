import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


SEQUENCE_LENGTH = 60

CLASSIFICATION = True

TRAIN_END = "2022-12-31"
VAL_END = "2023-12-31"

TARGET_COLUMN = "label_dir_1d"

BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.0003
EARLY_STOPPING_PATIENCE = 25
RANDOM_SEED = 42


LSTM_CONFIG = {
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": 0.3,
    "bidirectional": False,
}

BILSTM_CONFIG = {
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": 0.3,
    "bidirectional": True,
}

GRU_CONFIG = {
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": 0.3,
}

CNN_LSTM_CONFIG = {
    "cnn_filters": 32,
    "cnn_kernel_size": 3,
    "lstm_hidden_size": 64,
    "lstm_num_layers": 1,
    "dropout": 0.3,
}

TCN_CONFIG = {
    "num_channels": [32, 32, 32],
    "kernel_size": 3,
    "dropout": 0.3,
}

TRANSFORMER_CONFIG = {
    "d_model": 64,
    "nhead": 4,
    "num_encoder_layers": 2,
    "dim_feedforward": 128,
    "dropout": 0.2,
}

TFT_CONFIG = {
    "hidden_size": 64,
    "lstm_layers": 1,
    "attention_heads": 4,
    "dropout": 0.2,
}

XGBOOST_CONFIG = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}

ARIMA_CONFIG = {
    "order": (5, 1, 2),
}
