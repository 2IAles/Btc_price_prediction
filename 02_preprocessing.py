"""
Phase 2 — Preprocessing & Feature Engineering
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


# Dossiers de données et de sauvegarde des modèles
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Dates de coupure pour les splits temporels (pas de mélange aléatoire !)
TRAIN_END = "2022-12-31"
VAL_END = "2023-12-31"


def load_merged() -> pd.DataFrame:
    # Charge le fichier fusionné produit par la phase 1
    path = DATA_DIR / "merged_daily.csv"
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df.index = pd.to_datetime(df.index).normalize()
    print(f"[load] {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"       {df.index.min().date()} → {df.index.max().date()}")
    return df


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
    # (les week-ends et jours fériés n'ont pas de cotation, on propage la dernière valeur)
    ffill_cols = [
        c
        for c in df.columns
        if any(
            c.endswith(s)
            for s in ["_gold", "_sp500", "_dxy", "_vix", "_us10y", "_oil", "_silver"]
        )
    ] + ["fedfunds", "google_trends", "hashrate", "mvrv", "nupl"]
    df[ffill_cols] = df[ffill_cols].ffill()

    print(f"[clean] shape après nettoyage : {df.shape}")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # DataFrame vide qui va accueillir toutes les features construites
    feat = pd.DataFrame(index=df.index)

    # Prix de clôture de chaque actif, utilisés pour calculer les rendements
    close_assets = {
        "btc": df["close_btc"],
        "gold": df["close_gold"],
        "eth": df["close_eth"],
        "sp500": df["close_sp500"],
        "dxy": df["close_dxy"],
        "vix": df["close_vix"],
        "us10y": df["close_us10y"],
        "oil": df["close_oil"],
        "silver": df["close_silver"],
    }

    for name, close in close_assets.items():
        # Rendement log journalier : mesure la variation relative du prix en %
        lr = np.log(close / close.shift(1))
        feat[f"ret_1d_{name}"] = lr

        # Rendements multi-fenêtres (rolling sum of log returns)
        for w in [3, 7, 14, 30]:
            feat[f"ret_{w}d_{name}"] = lr.rolling(w).sum()

        # Volatilité roulante : écart-type des rendements sur la fenêtre
        for w in [7, 30]:
            feat[f"vol_{w}d_{name}"] = lr.rolling(w).std()

    # Momentum BTC : close / SMA (ratio prix actuel / moyenne mobile)
    for w in [7, 30]:
        sma = df["close_btc"].rolling(w).mean()
        feat[f"momentum_{w}d_btc"] = df["close_btc"] / sma

    # Ratio volume BTC / moyenne 7j : détecte les pics d'activité
    feat["vol_ratio_7d_btc"] = df["volume_btc"] / df["volume_btc"].rolling(7).mean()

    # Encodage cyclique du jour de la semaine (sin/cos pour éviter la discontinuité 6→0)
    dow = df.index.dayofweek.astype(float)
    feat["dow_sin"] = np.sin(dow * 2 * np.pi / 7)
    feat["dow_cos"] = np.cos(dow * 2 * np.pi / 7)

    # Indicateurs on-chain et sentiment (déjà daily ou forward-fillé)
    for col in [
        "fedfunds",
        "funding_rate",
        "hashrate",
        "mvrv",
        "nupl",
        "google_trends",
    ]:
        feat[col] = df[col]

    # Décalage d'un jour : les features de J-1 prédisent la direction de J (anti-leakage)
    feat = feat.shift(1)

    # Récupérer la cible (non décalée) : 1 = hausse le lendemain, 0 = baisse
    feat["label_dir_1d"] = df["label_dir_1d"]

    # Supprimer les NaN introduits par shift(1) et les rolling windows
    n_before = len(feat)
    feat = feat.dropna()
    print(
        f"[features] lignes supprimées (NaN post-shift/rolling) : {n_before - len(feat)}"
    )
    print(f"[features] shape finale : {feat.shape}")
    print(f"           {feat.index.min().date()} → {feat.index.max().date()}")
    print(f"           {feat.shape[1] - 1} features + 1 cible")

    return feat


def split(feat: pd.DataFrame):
    # Séparer features (X) et cible (y)
    X = feat.drop(columns=["label_dir_1d"])
    y = feat["label_dir_1d"]

    # Masques temporels : on ne mélange JAMAIS les données, on coupe dans le temps
    train_mask = feat.index <= TRAIN_END
    val_mask = (feat.index > TRAIN_END) & (feat.index <= VAL_END)
    test_mask = feat.index > VAL_END

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(
        f"\n[split] Train : {len(X_train):>5} lignes  "
        f"({X_train.index.min().date()} → {X_train.index.max().date()})"
    )
    print(
        f"[split] Val   : {len(X_val):>5} lignes  "
        f"({X_val.index.min().date()} → {X_val.index.max().date()})"
    )
    print(
        f"[split] Test  : {len(X_test):>5} lignes  "
        f"({X_test.index.min().date()} → {X_test.index.max().date()})"
    )

    # Vérification anti-leakage : les splits ne doivent pas se chevaucher
    assert X_train.index.max() < X_val.index.min(), "LEAKAGE : train/val se chevauchent"
    assert X_val.index.max() < X_test.index.min(), "LEAKAGE : val/test se chevauchent"

    # Distribution de la cible : vérifie que les classes sont équilibrées
    for name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        rate = y_split.mean()
        print(f"         {name} hausse J+1 : {rate:.2%}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize(X_train, X_val, X_test):
    # RobustScaler est robuste aux outliers (utilise médiane et IQR plutôt que moyenne/std)
    scaler = RobustScaler()
    # fit_transform sur le train uniquement pour éviter toute fuite d'information
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns
    )
    # transform (sans fit) sur val et test : on applique les paramètres du train
    X_val_s = pd.DataFrame(
        scaler.transform(X_val), index=X_val.index, columns=X_val.columns
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test), index=X_test.index, columns=X_test.columns
    )

    # Sauvegarde du scaler pour pouvoir normaliser de nouvelles données en production
    scaler_path = MODELS_DIR / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\n[scaler] RobustScaler sauvegardé → {scaler_path}")
    return X_train_s, X_val_s, X_test_s, scaler


def export(X_train, X_val, X_test, y_train, y_val, y_test):
    # Sauvegarde tous les splits en fichiers .pkl pour les réutiliser dans les notebooks
    splits = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }
    for name, obj in splits.items():
        path = DATA_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"[export] {path}  shape={obj.shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 2 — Preprocessing & Feature Engineering")
    print("=" * 60)

    # Pipeline complet : chargement → nettoyage → features → split → normalisation → export
    df = load_merged()
    df = clean(df)
    feat = build_features(df)
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
