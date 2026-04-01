"""
Phase 2 — Preprocessing & Feature Engineering
==============================================
Input  : data/merged_daily.csv  (généré par 01_eda.ipynb)
Outputs: data/X_train.pkl  data/X_val.pkl  data/X_test.pkl
         data/y_train.pkl  data/y_val.pkl  data/y_test.pkl
         models/scaler.pkl

Règles anti-leakage
-------------------
- Split chronologique (pas de shuffle) :
    Train : jusqu'au 2022-12-31
    Val   : 2023-01-01 – 2023-12-31
    Test  : 2024-01-01 → fin
- Toutes les features sont shift(1) : on utilise les données de J-1 pour prédire J
- RobustScaler fitté uniquement sur X_train, appliqué sur val et test
- label_dir_1d jamais inclus dans les features
- La colonne date reste en index
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
DATA_DIR   = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

TRAIN_END = "2022-12-31"
VAL_END   = "2023-12-31"

# ---------------------------------------------------------------------------
# 1. Chargement du dataset fusionné
# ---------------------------------------------------------------------------

def load_merged() -> pd.DataFrame:
    path = DATA_DIR / "merged_daily.csv"
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df.index = pd.to_datetime(df.index).normalize()
    print(f"[load] {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"       {df.index.min().date()} → {df.index.max().date()}")
    return df

# ---------------------------------------------------------------------------
# 2. Nettoyage
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Supprimer les lignes sans prix BTC (référence indispensable)
    n_before = len(df)
    df = df.dropna(subset=["close_btc"])
    print(f"[clean] lignes sans close_btc supprimées : {n_before - len(df)}")

    # Supprimer les doublons de dates
    n_dup = df.index.duplicated().sum()
    df = df[~df.index.duplicated(keep="first")]
    if n_dup:
        print(f"[clean] doublons de dates supprimés : {n_dup}")

    # Forward-fill sur les actifs marchés fermés et google trends
    ffill_cols = [c for c in df.columns if any(
        c.endswith(s) for s in ["_gold", "_sp500", "_dxy", "_vix", "_us10y", "_oil", "_silver"]
    )] + ["fedfunds", "google_trends", "hashrate", "mvrv", "nupl"]
    df[ffill_cols] = df[ffill_cols].ffill()

    print(f"[clean] shape après nettoyage : {df.shape}")
    return df

# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Toutes les features sont calculées puis décalées d'un jour (shift(1))
    pour éviter tout leakage : on ne peut utiliser que des données de J-1
    pour prédire la direction de J.
    """
    feat = pd.DataFrame(index=df.index)

    close_assets = {
        "btc":   df["close_btc"],
        "gold":  df["close_gold"],
        "eth":   df["close_eth"],
        "sp500": df["close_sp500"],
        "dxy":   df["close_dxy"],
        "vix":   df["close_vix"],
        "us10y": df["close_us10y"],
        "oil":   df["close_oil"],
        "silver":df["close_silver"],
    }

    for name, close in close_assets.items():
        # Rendement log journalier
        lr = np.log(close / close.shift(1))
        feat[f"ret_1d_{name}"]  = lr

        # Rendements multi-fenêtres (rolling sum of log returns)
        for w in [3, 7, 14, 30]:
            feat[f"ret_{w}d_{name}"] = lr.rolling(w).sum()

        # Volatilité roulante
        for w in [7, 30]:
            feat[f"vol_{w}d_{name}"] = lr.rolling(w).std()

    # Momentum BTC : close / SMA
    for w in [7, 30]:
        sma = df["close_btc"].rolling(w).mean()
        feat[f"momentum_{w}d_btc"] = df["close_btc"] / sma

    # Ratio volume BTC / moyenne 7j
    feat["vol_ratio_7d_btc"] = (
        df["volume_btc"] / df["volume_btc"].rolling(7).mean()
    )

    # Encodage cyclique du jour de la semaine
    dow = df.index.dayofweek.astype(float)
    feat["dow_sin"] = np.sin(dow * 2 * np.pi / 7)
    feat["dow_cos"] = np.cos(dow * 2 * np.pi / 7)

    # Indicateurs on-chain et sentiment (déjà daily ou forward-fillé)
    for col in ["fedfunds", "funding_rate", "hashrate", "mvrv", "nupl", "google_trends"]:
        feat[col] = df[col]

    # -----------------------------------------------------------------------
    # ANTI-LEAKAGE : décalage d'un jour
    # Toutes les features representent J-1, la cible représente J
    # -----------------------------------------------------------------------
    feat = feat.shift(1)

    # Récupérer la cible (non décalée)
    feat["label_dir_1d"] = df["label_dir_1d"]

    # Supprimer les NaN introduits par shift(1) et les rolling windows
    n_before = len(feat)
    feat = feat.dropna()
    print(f"[features] lignes supprimées (NaN post-shift/rolling) : {n_before - len(feat)}")
    print(f"[features] shape finale : {feat.shape}")
    print(f"           {feat.index.min().date()} → {feat.index.max().date()}")
    print(f"           {feat.shape[1] - 1} features + 1 cible")

    return feat

# ---------------------------------------------------------------------------
# 4. Split temporel
# ---------------------------------------------------------------------------

def split(feat: pd.DataFrame):
    X = feat.drop(columns=["label_dir_1d"])
    y = feat["label_dir_1d"]

    train_mask = feat.index <= TRAIN_END
    val_mask   = (feat.index > TRAIN_END) & (feat.index <= VAL_END)
    test_mask  = feat.index > VAL_END

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    print(f"\n[split] Train : {len(X_train):>5} lignes  "
          f"({X_train.index.min().date()} → {X_train.index.max().date()})")
    print(f"[split] Val   : {len(X_val):>5} lignes  "
          f"({X_val.index.min().date()} → {X_val.index.max().date()})")
    print(f"[split] Test  : {len(X_test):>5} lignes  "
          f"({X_test.index.min().date()} → {X_test.index.max().date()})")

    # Vérification anti-leakage
    assert X_train.index.max() < X_val.index.min(), "LEAKAGE : train/val se chevauchent"
    assert X_val.index.max()   < X_test.index.min(), "LEAKAGE : val/test se chevauchent"

    # Distribution de la cible
    for name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        rate = y_split.mean()
        print(f"         {name} hausse J+1 : {rate:.2%}")

    return X_train, X_val, X_test, y_train, y_val, y_test

# ---------------------------------------------------------------------------
# 5. Normalisation
# ---------------------------------------------------------------------------

def normalize(X_train, X_val, X_test):
    scaler = RobustScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index, columns=X_train.columns
    )
    X_val_s = pd.DataFrame(
        scaler.transform(X_val),
        index=X_val.index, columns=X_val.columns
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index, columns=X_test.columns
    )

    scaler_path = MODELS_DIR / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\n[scaler] RobustScaler sauvegardé → {scaler_path}")
    return X_train_s, X_val_s, X_test_s, scaler

# ---------------------------------------------------------------------------
# 6. Export
# ---------------------------------------------------------------------------

def export(X_train, X_val, X_test, y_train, y_val, y_test):
    splits = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }
    for name, obj in splits.items():
        path = DATA_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"[export] {path}  shape={obj.shape}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 2 — Preprocessing & Feature Engineering")
    print("=" * 60)

    df    = load_merged()
    df    = clean(df)
    feat  = build_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split(feat)
    X_train, X_val, X_test, scaler = normalize(X_train, X_val, X_test)
    export(X_train, X_val, X_test, y_train, y_val, y_test)

    print("\n" + "=" * 60)
    print("  Preprocessing terminé.")
    print(f"  Features : {X_train.shape[1]}")
    print(f"  Train    : {X_train.shape[0]} lignes")
    print(f"  Val      : {X_val.shape[0]} lignes")
    print(f"  Test     : {X_test.shape[0]} lignes")
    print("=" * 60)
