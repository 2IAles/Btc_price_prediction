import sys
import pickle
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Charger le prédicteur
from predictor import BTCPredictor

print("=" * 60)
print("  BTCPredictor — exemple d'utilisation")
print("=" * 60)

# ── 1. Charger le prédicteur ─────────────────────────────────────────────────
predictor = BTCPredictor.load(models_dir="models/", data_dir="data/")
print(f"\n{predictor}\n")

# ── 2. Construire un DataFrame de test (données réelles) ─────────────────────
# On recharge les données brutes journalières pour simuler un vrai scénario.
DATA_DIR = Path("data")


def load_ohlcv(fname, prefix):
    path = DATA_DIR / fname
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [f"{c.lower()}_{prefix}" for c in df.columns]
    return df


def load_single(fname, colname):
    path = DATA_DIR / fname
    if not path.exists():
        return pd.Series(dtype=float, name=colname)
    s = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
    s.name = colname
    return s


# Chargement des fichiers daily
frames = [
    load_ohlcv("btc_daily.csv", "btc"),
    load_ohlcv("xau_daily.csv", "xau"),
    load_ohlcv("eth_daily.csv", "eth"),
    load_ohlcv("snp500_daily.csv", "snp500"),
    load_ohlcv("dxy_daily.csv", "dxy"),
    load_ohlcv("vix_daily.csv", "vix"),
    load_ohlcv("us10y_daily.csv", "us10y"),
    load_ohlcv("oil_daily.csv", "oil"),
    load_ohlcv("silver_daily.csv", "silver"),
]
scalars = [
    load_single("fedfunds_daily.csv", "fedfunds"),
    load_single("btc_funding_rate_daily.csv", "funding_rate"),
    load_single("btc_hashrate_daily.csv", "hashrate"),
    load_single("btc_mvrv_daily.csv", "mvrv"),
    load_single("btc_nupl_daily.csv", "nupl"),
    load_single("google_trends_bitcoin.csv", "google_trends"),
]

non_empty_frames = [f for f in frames if not f.empty]
non_empty_scalars = [s for s in scalars if not s.empty]

if not non_empty_frames:
    raise RuntimeError(
        "Aucun fichier daily trouvé dans data/. " "Exécutez collect_daily.py d'abord."
    )

merged = non_empty_frames[0].copy()
for f in non_empty_frames[1:]:
    merged = merged.join(f, how="left")
for s in non_empty_scalars:
    merged = merged.join(s, how="left")

merged = merged.ffill()
print(
    f"Données disponibles : {merged.index.min().date()} → {merged.index.max().date()}"
)
print(f"Shape : {merged.shape}")

# ── 3. Prédiction sur les 90 derniers jours de données ──────────────────────
# On garde les 90 derniers jours pour s'assurer d'avoir SEQ_LEN lignes de
# features valides après le feature engineering (qui consomme ~30 jours pour
# les rendements roulants 30j).
input_df = merged.iloc[-90:].copy()

print(
    f"\nFenêtre d'inférence : {input_df.index.min().date()} → {input_df.index.max().date()}"
)

# ── 4. Prédiction ────────────────────────────────────────────────────────────
result = predictor.predict(input_df)

print("\n" + "=" * 60)
print("  RÉSULTAT DE LA PRÉDICTION")
print("=" * 60)
print(
    f"  Direction   : {'HAUSSE' if result['direction'] == 1 else 'BAISSE'} ({result['direction']})"
)
print(
    f"  Probabilité : {result['probability']:.4f}  ({result['probability']*100:.1f}%)"
)
print(f"  Horizon     : {result['horizon']}")
print(f"  Modèle      : {result['model']}")
print(f"  Seuil       : {result['threshold']}")
print("=" * 60)

# ── 5. Prédictions glissantes sur le test set (validation visuelle) ──────────
print("\nPrédictions glissantes (30 derniers jours disponibles) :")
print(f"  {'Date':<12}  {'Direction':>10}  {'Probabilité':>12}  {'Confiance':>10}")
print("  " + "-" * 50)

# On utilise une fenêtre glissante sur les 30 derniers jours du test set
with open("data/X_test.pkl", "rb") as f:
    X_test = pickle.load(f)
with open("data/y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

SEQ_LEN = BTCPredictor.SEQUENCE_LENGTH
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

X_scaled = scaler.transform(X_test)
n_days = min(30, len(X_test) - SEQ_LEN)
start = len(X_test) - SEQ_LEN - n_days

correct = 0
for i in range(n_days):
    idx = start + i
    seq = X_scaled[idx : idx + SEQ_LEN][np.newaxis, :, :].astype(np.float32)
    prob = predictor._model.predict(seq)[0]
    pred = int(prob >= predictor._threshold)
    true = int(y_test.iloc[idx + SEQ_LEN])
    ok = pred == true
    correct += ok
    date = X_test.index[idx + SEQ_LEN].date()
    label = "HAUSSE" if pred == 1 else "BAISSE"
    conf = abs(prob - 0.5) * 2  # 0=incertain, 1=très confiant
    mark = "" if ok else " X"
    print(f"  {str(date):<12}  {label:>10}  {prob:>12.4f}  {conf:>9.1%}{mark}")

print(
    f"\n  Accuracy sur ces {n_days} jours : {correct}/{n_days} ({correct/n_days:.1%})"
)
print("\nDone. Le prédicteur fonctionne correctement.")
