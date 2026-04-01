# Plan — Prédiction de Tendance Bitcoin (label_dir_1d)

## Contexte

Le pipeline de collecte de données journalières est terminé (16 CSV dans `data/`).
L'objectif est de construire le pipeline ML complet : EDA → Preprocessing → Modèles → Analyse → Prédicteur → Rapport.

Aucune infrastructure ML n'existe encore (pas de `models/`, aucun notebook d'entraînement).

---

## État actuel du projet

**Fichiers de données existants** (tous dans `data/`) :
- OHLCV journalier : `btc_daily.csv`, `xau_daily.csv`, `eth_daily.csv`, `snp500_daily.csv`, `dxy_daily.csv`, `vix_daily.csv`, `us10y_daily.csv`, `oil_daily.csv`, `silver_daily.csv`
- Format Date,Value : `fedfunds_daily.csv`, `btc_funding_rate_daily.csv`, `btc_hashrate_daily.csv`, `btc_mvrv_daily.csv`, `btc_nupl_daily.csv`, `google_trends_bitcoin.csv`

**Attention — noms réels des fichiers** (différents de la structure cible du plan) :
| Plan utilisateur | Fichier réel |
|---|---|
| `btc_1d.csv` | `btc_daily.csv` |
| `gold_1d.csv` | `xau_daily.csv` |
| `silver_1d.csv` | `silver_daily.csv` |
| `oil_1d.csv` | `oil_daily.csv` |
| `sp500_1d.csv` | `snp500_daily.csv` |

**Couverture temporelle limitée** (certains actifs) :
- `eth_daily.csv` : depuis 2017 seulement
- `btc_funding_rate_daily.csv` : depuis 2019-09 seulement
- Google Trends : mensuel (besoin de forward-fill vers le daily)

---

## Fichiers à créer

```
01_eda.ipynb
02_preprocessing.py
03_models.ipynb
04_results_analysis.ipynb
05_predictor.py
example_usage.py
rapport.ipynb
models/                    # à créer
data/X_train.pkl, X_val.pkl, X_test.pkl
data/y_train.pkl, y_val.pkl, y_test.pkl
models/scaler.pkl
models/best_model.pkl
```

---

## Phase 1 — EDA (`01_eda.ipynb`) [x]

1. Charger chaque source (utiliser les noms réels des fichiers)
2. Fusion sur `date` — left join sur `btc_daily.csv` comme référence
   - Renommer les colonnes par actif : `open_btc`, `close_gold`, `volume_sp500`, etc.
   - Les fichiers Date,Value → renommés en `fedfunds`, `funding_rate`, `hashrate`, `mvrv`, `nupl`, `google_trends`
3. Inspection : shape, types, plage temporelle, missing values
4. **Création de la cible** : `label_dir_1d = (close_btc.shift(-1) > close_btc).astype(int)` — supprimer la dernière ligne (target inconnue)
5. Analyse des classes (déséquilibre ?)
6. Missing values : forward-fill sur actifs marchés fermés (xau, silver, oil, snp500, fedfunds) + google_trends (mensuel → daily)
7. Outliers : IQR / zscore sur features numériques
8. Visualisations : prix BTC, heatmap corrélations rendements, distribution cible
9. Conclusions → décisions preprocessing

---

## Phase 2 — Preprocessing (`02_preprocessing.py`) [x]

**Nettoyage :**
- Supprimer lignes sans `close_btc`
- Forward-fill actifs marchés fermés
- Supprimer doublons de dates

**Feature engineering** (toutes les features sont `shift(1)` — données J-1 pour prédire J) :
- Rendements log par actif : `log(close[t] / close[t-1])`
- Rendements multi-fenêtres : 3j, 7j, 14j, 30j (rolling)
- Volatilité roulante : std des rendements sur 7j et 30j
- Ratio volume / moyenne 7j (BTC)
- Momentum : close / SMA(7j), close / SMA(30j)
- Encodage cyclique du jour de semaine : sin/cos(dow * 2π/7)
- Les indicateurs BTC (funding_rate, hashrate, mvrv, nupl, google_trends) : shift(1) obligatoire

**Séparation temporelle** (pas de shuffle) :
- Train : ≤ 2022-12-31
- Val : 2023-01-01 – 2023-12-31
- Test : ≥ 2024-01-01

**Normalisation :**
- `RobustScaler` fitté uniquement sur `X_train`
- Appliqué sur `X_val` et `X_test`
- Sauvegardé : `models/scaler.pkl`

**Export :**
- `data/X_train.pkl`, `data/X_val.pkl`, `data/X_test.pkl`
- `data/y_train.pkl`, `data/y_val.pkl`, `data/y_test.pkl`

---

## Phase 3 — Adaptation des modèles `modèle llm/` + Modélisation (`03_models.ipynb`) [x]

### Stratégie : adapter les modèles existants (pas réécrire)

Les corps des modèles DL (couches LSTM, attention, convolutions) sont identiques entre
régression et classification. Seuls **3 éléments** changent : la dernière couche, la loss,
et les métriques. Les modifications sont localisées dans 2-3 fichiers.

#### Étape 3.1 — `modèle llm/config.py` [x]

- Ajouter `CLASSIFICATION = True`
- Changer `TARGET_COLUMN = "label_dir_1d"` (était `"Close"`)
- Supprimer `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO` (remplacés par des dates fixes)
- Ajouter les dates de split :
  ```python
  TRAIN_END = "2022-12-31"
  VAL_END   = "2023-12-31"
  ```

#### Étape 3.2 — `modèle llm/metrics.py` [x]

Remplacer les métriques de régression par des métriques de classification :
- Supprimer : `compute_mae`, `compute_rmse`, `compute_mape`, `compute_r2`, `compute_direction_accuracy`
- Ajouter : `compute_f1`, `compute_auc_roc`, `compute_precision`, `compute_recall`, `compute_accuracy`
- Mettre à jour `evaluate_model()` et `print_evaluation()` en conséquence

#### Étape 3.3 — `modèle llm/trainer.py` [x]

- Changer la loss : `nn.MSELoss()` → `nn.BCEWithLogitsLoss()`
- Adapter `predict()` : appliquer `torch.sigmoid()` sur les logits de sortie
- Adapter l'early stopping : monitorer la val F1 (↑) plutôt que la val loss (↓)

#### Étape 3.4 — Modèles PyTorch (aucun changement dans les couches cachées) [x]

Dans chaque fichier modèle, la seule modification est la couche finale :
- `modèle llm/lstm_model.py` : `self.fc2 = nn.Linear(64, 1)` reste inchangé — c'est la loss qui change
- Idem pour `gru_model.py`, `cnn_lstm_model.py`, `tcn_model.py`, `transformer_model.py`, `tft_model.py`
- Aucune modification nécessaire dans les corps des modèles

#### Étape 3.5 — `modèle llm/xgboost_model.py` [x]

- Changer `objective='reg:squarederror'` → `objective='binary:logistic'`
- Ajouter `eval_metric='logloss'`

---

### Modèles à entraîner (après adaptation)

**Groupe 1 — Baselines sklearn** (implémentation directe dans `03_models.ipynb`) :
| Modèle | Justification |
|---|---|
| `DummyClassifier(most_frequent)` | Baseline absolue |
| `LogisticRegression` | Baseline linéaire |
| `RandomForestClassifier` | Robuste, feature importances |
| `GradientBoostingClassifier` / XGBoost | État de l'art tabulaire |

**Groupe 2 — Deep Learning PyTorch** (depuis `modèle llm/`, après étapes 3.1–3.5) :
| Modèle | Fichier source | Priorité |
|---|---|---|
| LSTM | `lstm_model.py` | Haute (référence littérature) |
| GRU | `gru_model.py` | Haute (plus léger que LSTM) |
| CNN-LSTM | `cnn_lstm_model.py` | Haute (patterns techniques) |
| TFT simplifié | `tft_model.py` | Haute (SOTA selon rapport) |
| Transformer | `transformer_model.py` | Moyenne |
| BiLSTM | `lstm_model.py` (bidirectional=True) | Optionnel |
| TCN | `tcn_model.py` | Optionnel |

> Le rapport `modèle llm/rapport_btc_prediction.md` recommande : TFT > LSTM/BiLSTM > CNN-LSTM > XGBoost. Confirmer ou infirmer sur les vraies données.

### Métriques cibles (classification)

**Principale :** F1-score (classe 1)
**Secondaires :** Accuracy, AUC-ROC, Precision, Recall, Confusion Matrix

### Protocole

- **Sklearn** : `TimeSeriesSplit(n_splits=5)` sur train+val, tuning des 2 meilleurs via GridSearchCV
- **PyTorch** : séquences de 60 jours (`SEQUENCE_LENGTH=60`), early stopping (patience=15), Adam, gradient clipping
- Évaluation finale sur test set : **une seule fois**
- Sauvegarde : `models/best_model.pkl`

---

## Phase 4 — Analyse des Résultats (`04_results_analysis.ipynb`) [x]

1. Courbes ROC comparatives
2. Matrice de confusion du meilleur modèle
3. Analyse des erreurs (conditions de mauvaise prédiction)
4. Feature importance globale (SHAP si disponible)
5. Comparaison avec la littérature
6. Discussion des limites
7. Recommandation finale

---

## Phase 5 — Prédicteur (`05_predictor.py` + `example_usage.py`) [x]

```python
from predictor import BTCPredictor
p = BTCPredictor.load("models/")
prediction = p.predict(input_df)
# → {"direction": 1, "probability": 0.67, "horizon": "1d"}
```

La classe `BTCPredictor` :
- Charge modèle + scaler depuis `models/`
- Applique le même preprocessing que Phase 2
- Expose `predict(df)` et `predict_proba(df)`

---

## Phase 6 — Rapport (`rapport.ipynb`) [x]

Structure (max 15 pages) :
1. Problématique
2. Jeu de données (sources, qualité)
3. Preprocessing (choix justifiés)
4. Méthodologie et modélisation
5. Protocole d'évaluation
6. Résultats et comparaison
7. Recommandation finale
8. État de l'art (réutiliser `modèle llm/rapport_btc_prediction.md`)
9. Conclusion

---

## Points critiques (data leakage)

- Jamais de shuffle sur données temporelles
- `label_dir_1d` construit depuis `close_btc.shift(-1)` — jamais comme feature
- Toutes les features doivent être décalées d'1 jour (`shift(1)`) — données J-1 pour prédire J
- RobustScaler fitté uniquement sur train
- `date` reste en index, jamais comme feature
- Évaluation test set : une seule fois en fin de projet

---

## Vérification

1. Exécuter `02_preprocessing.py` → vérifier que `X_train.pkl` etc. sont générés dans `data/`
2. Vérifier shape : `X_train`, `X_val`, `X_test` — colonnes identiques, aucun NaN résiduel
3. Vérifier absence de leakage : `assert X_train.index.max() < X_val.index.min()`
4. Dans `03_models.ipynb` : vérifier que `TimeSeriesSplit` respecte la chronologie
5. Tester le prédicteur : `example_usage.py` doit retourner `{"direction": 0|1, "probability": float, "horizon": "1d"}`
