"""
Pipeline principal — Benchmark complet de 7 architectures ML/DL pour la prédiction du BTC.

Ce script orchestre tout le pipeline :
    1. Génération de données synthétiques (remplaçable par de vraies données)
    2. Calcul des indicateurs techniques
    3. Préparation des séquences temporelles
    4. Entraînement et évaluation de chaque modèle
    5. Comparaison des performances
    6. Export des résultats

Modèles évalués :
    ┌─────────────────────────────────────────────────────────────────┐
    │  Deep Learning (réseaux récurrents)                            │
    │  ├── LSTM          — référence pour les séries temporelles     │
    │  ├── BiLSTM        — LSTM bidirectionnel                       │
    │  └── GRU           — variante légère du LSTM                   │
    │                                                                │
    │  Deep Learning (convolutionnel)                                │
    │  ├── CNN-LSTM      — hybride extraction locale + mémoire       │
    │  └── TCN           — convolutions causales dilatées            │
    │                                                                │
    │  Deep Learning (attention)                                     │
    │  ├── Transformer   — attention multi-têtes pure                │
    │  └── TFT           — fusion temporelle avec sélection de vars  │
    │                                                                │
    │  Machine Learning classique                                    │
    │  └── XGBoost       — gradient boosting (baseline)              │
    └─────────────────────────────────────────────────────────────────┘

Usage :
    python main.py
"""

import sys
import os
import numpy as np
import pandas as pd
import time
import json

# Ajout du répertoire racine au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from utils.preprocessing import (
    generate_synthetic_btc_data,
    compute_technical_indicators,
    prepare_data_splits,
)
from utils.metrics import evaluate_model, print_evaluation
from utils.trainer import Trainer


def run_pytorch_model(model, model_name, data_splits, config_epochs=EPOCHS):
    """
    Entraîne et évalue un modèle PyTorch donné.
    
    Returns:
        Dictionnaire avec les métriques d'évaluation et les métadonnées
    """
    print(f"\n{'#'*60}")
    print(f"  MODÈLE : {model_name}")
    print(f"{'#'*60}")

    # Compte des paramètres
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Paramètres entraînables : {n_params:,}")

    # Entraînement
    trainer = Trainer(
        model=model,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        epochs=config_epochs,
        patience=EARLY_STOPPING_PATIENCE,
    )

    history = trainer.train(
        data_splits["X_train"], data_splits["y_train"],
        data_splits["X_val"], data_splits["y_val"],
        verbose=True,
    )

    # Prédictions sur le jeu de test
    predictions = trainer.predict(data_splits["X_test"])

    # Dé-normalisation pour obtenir les prix en dollars
    scaler = data_splits["scaler"]
    target_idx = data_splits["target_col_idx"]
    y_test_real = scaler.inverse_transform_column(data_splits["y_test"], target_idx)
    y_pred_real = scaler.inverse_transform_column(predictions, target_idx)

    # Évaluation
    results = evaluate_model(y_test_real, y_pred_real, model_name)
    results["n_params"] = n_params
    results["training_time"] = history.get("training_time", 0)
    results["best_val_loss"] = min(history["val_loss"])
    results["n_epochs_trained"] = len(history["val_loss"])

    print_evaluation(results)
    return results


def run_xgboost_model(data_splits):
    """Entraîne et évalue le modèle XGBoost."""
    from models.xgboost_model import XGBoostWrapper

    print(f"\n{'#'*60}")
    print(f"  MODÈLE : XGBoost (Baseline ML)")
    print(f"{'#'*60}")

    start_time = time.time()

    wrapper = XGBoostWrapper(XGBOOST_CONFIG)
    wrapper.fit(
        data_splits["X_train"], data_splits["y_train"],
        data_splits["X_val"], data_splits["y_val"],
    )

    training_time = time.time() - start_time
    predictions = wrapper.predict(data_splits["X_test"])

    # Dé-normalisation
    scaler = data_splits["scaler"]
    target_idx = data_splits["target_col_idx"]
    y_test_real = scaler.inverse_transform_column(data_splits["y_test"], target_idx)
    y_pred_real = scaler.inverse_transform_column(predictions, target_idx)

    results = evaluate_model(y_test_real, y_pred_real, "XGBoost")
    results["n_params"] = "N/A (arbres)"
    results["training_time"] = training_time
    results["best_val_loss"] = "N/A"
    results["n_epochs_trained"] = XGBOOST_CONFIG["n_estimators"]

    print_evaluation(results)
    return results


def main():
    """
    Point d'entrée principal — exécute le benchmark complet.
    """
    print("=" * 70)
    print("  BENCHMARK : Architectures ML/DL pour la prédiction du prix du BTC")
    print("=" * 70)

    # ====================================================================
    # ÉTAPE 1 : Génération et préparation des données
    # ====================================================================
    print("\n[1/3] Génération des données synthétiques BTC...")
    df_raw = generate_synthetic_btc_data(n_days=2000, seed=RANDOM_SEED)
    print(f"  → {len(df_raw)} jours de données brutes générées")

    print("\n[1/3] Calcul des indicateurs techniques...")
    df = compute_technical_indicators(df_raw)
    print(f"  → {len(df)} jours après suppression des NaN")
    print(f"  → {len(FEATURE_COLUMNS)} features disponibles")

    print("\n[1/3] Préparation des séquences temporelles...")
    data_splits = prepare_data_splits(
        df=df,
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        seq_length=SEQUENCE_LENGTH,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
    )
    print(f"  → Séquences : longueur={SEQUENCE_LENGTH}, features={data_splits['n_features']}")
    print(f"  → Train: {len(data_splits['X_train'])}, Val: {len(data_splits['X_val'])}, Test: {len(data_splits['X_test'])}")

    n_features = data_splits["n_features"]

    # ====================================================================
    # ÉTAPE 2 : Entraînement et évaluation de chaque modèle
    # ====================================================================
    print("\n[2/3] Lancement des entraînements...")
    all_results = []

    # --- 1. LSTM ---
    from models.lstm_model import LSTMModel
    model = LSTMModel(
        input_size=n_features,
        hidden_size=LSTM_CONFIG["hidden_size"],
        num_layers=LSTM_CONFIG["num_layers"],
        dropout=LSTM_CONFIG["dropout"],
        bidirectional=False,
    )
    all_results.append(run_pytorch_model(model, "LSTM", data_splits))

    # --- 2. BiLSTM ---
    model = LSTMModel(
        input_size=n_features,
        hidden_size=BILSTM_CONFIG["hidden_size"],
        num_layers=BILSTM_CONFIG["num_layers"],
        dropout=BILSTM_CONFIG["dropout"],
        bidirectional=True,
    )
    all_results.append(run_pytorch_model(model, "BiLSTM", data_splits))

    # --- 3. GRU ---
    from models.gru_model import GRUModel
    model = GRUModel(
        input_size=n_features,
        hidden_size=GRU_CONFIG["hidden_size"],
        num_layers=GRU_CONFIG["num_layers"],
        dropout=GRU_CONFIG["dropout"],
    )
    all_results.append(run_pytorch_model(model, "GRU", data_splits))

    # --- 4. CNN-LSTM ---
    from models.cnn_lstm_model import CNNLSTMModel
    model = CNNLSTMModel(
        input_size=n_features,
        cnn_filters=CNN_LSTM_CONFIG["cnn_filters"],
        cnn_kernel_size=CNN_LSTM_CONFIG["cnn_kernel_size"],
        lstm_hidden_size=CNN_LSTM_CONFIG["lstm_hidden_size"],
        lstm_num_layers=CNN_LSTM_CONFIG["lstm_num_layers"],
        dropout=CNN_LSTM_CONFIG["dropout"],
    )
    all_results.append(run_pytorch_model(model, "CNN-LSTM", data_splits))

    # --- 5. TCN ---
    from models.tcn_model import TCNModel
    model = TCNModel(
        input_size=n_features,
        num_channels=TCN_CONFIG["num_channels"],
        kernel_size=TCN_CONFIG["kernel_size"],
        dropout=TCN_CONFIG["dropout"],
    )
    all_results.append(run_pytorch_model(model, "TCN", data_splits))

    # --- 6. Transformer ---
    from models.transformer_model import TransformerModel
    model = TransformerModel(
        input_size=n_features,
        d_model=TRANSFORMER_CONFIG["d_model"],
        nhead=TRANSFORMER_CONFIG["nhead"],
        num_encoder_layers=TRANSFORMER_CONFIG["num_encoder_layers"],
        dim_feedforward=TRANSFORMER_CONFIG["dim_feedforward"],
        dropout=TRANSFORMER_CONFIG["dropout"],
    )
    all_results.append(run_pytorch_model(model, "Transformer", data_splits))

    # --- 7. TFT (Temporal Fusion Transformer) ---
    from models.tft_model import SimplifiedTFT
    model = SimplifiedTFT(
        input_size=n_features,
        hidden_size=TFT_CONFIG["hidden_size"],
        lstm_layers=TFT_CONFIG["lstm_layers"],
        attention_heads=TFT_CONFIG["attention_heads"],
        dropout=TFT_CONFIG["dropout"],
    )
    all_results.append(run_pytorch_model(model, "TFT (Simplifié)", data_splits))

    # --- 8. XGBoost ---
    try:
        all_results.append(run_xgboost_model(data_splits))
    except Exception as e:
        print(f"  ⚠ XGBoost non disponible : {e}")

    # ====================================================================
    # ÉTAPE 3 : Comparaison et export des résultats
    # ====================================================================
    print("\n[3/3] Comparaison finale des modèles...")
    print("\n" + "=" * 90)
    print(f"{'Modèle':<20} {'MAE ($)':>10} {'RMSE ($)':>10} {'MAPE (%)':>10} {'R²':>8} {'Dir. Acc.':>10} {'Temps (s)':>10}")
    print("-" * 90)

    for r in all_results:
        print(
            f"{r['model']:<20} "
            f"{r['MAE']:>10,.2f} "
            f"{r['RMSE']:>10,.2f} "
            f"{r['MAPE (%)']:>10.2f} "
            f"{r['R²']:>8.4f} "
            f"{r['Direction Accuracy (%)']:>9.2f}% "
            f"{r.get('training_time', 0):>10.1f}"
        )
    print("=" * 90)

    # Meilleur modèle par métrique
    print("\n  🏆 MEILLEURS MODÈLES PAR MÉTRIQUE :")
    metrics_to_check = [
        ("MAE", False), ("RMSE", False), ("MAPE (%)", False),
        ("R²", True), ("Direction Accuracy (%)", True),
    ]
    for metric, higher_is_better in metrics_to_check:
        if higher_is_better:
            best = max(all_results, key=lambda x: x[metric])
        else:
            best = min(all_results, key=lambda x: x[metric])
        print(f"    {metric:<25}: {best['model']} ({best[metric]:.4f})")

    # Export JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "benchmark_results.json")

    # Conversion pour JSON (gérer les types non-sérialisables)
    json_results = []
    for r in all_results:
        jr = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                jr[k] = float(v)
            else:
                jr[k] = v
        json_results.append(jr)

    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"\n  → Résultats exportés dans : {results_path}")

    return all_results


if __name__ == "__main__":
    results = main()
