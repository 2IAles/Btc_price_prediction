# Rapport de Recherche : Architectures ML/DL pour la Prédiction du Prix du Bitcoin

**Date :** Mars 2026  
**Auteur :** Claude (Anthropic)  
**Objectif :** Identifier, implémenter et comparer les architectures de Machine Learning et Deep Learning les plus adaptées à la prédiction du prix du Bitcoin.

---

## 1. Contexte et Problématique

La prédiction du prix du Bitcoin est un problème de régression sur série temporelle particulièrement complexe. Contrairement à des séries temporelles classiques (météo, consommation d'énergie), le Bitcoin présente des caractéristiques qui rendent la prédiction extrêmement difficile :

- **Volatilité extrême** : des variations journalières de 5 à 15% sont courantes, avec des krachs ou des rallyes soudains qui défient la plupart des modèles statistiques.
- **Non-stationnarité** : la distribution des rendements change au cours du temps (régimes bull/bear), ce qui invalide les hypothèses de nombreux modèles classiques.
- **Dépendances non-linéaires** : le prix est influencé par un mélange complexe de facteurs techniques, macroéconomiques, sentimentaux et réglementaires qui interagissent de manière non-linéaire.
- **Efficience partielle du marché** : l'hypothèse de marché efficient suggère que les prix reflètent déjà toute l'information disponible, rendant la prédiction théoriquement impossible. En pratique, des inefficiences existent mais sont difficiles à exploiter.

Ces défis expliquent pourquoi la littérature scientifique explore une grande variété d'architectures, des modèles statistiques classiques (ARIMA) aux architectures deep learning les plus récentes (Temporal Fusion Transformer).

---

## 2. Revue de la Littérature Scientifique

### 2.1 Modèles statistiques classiques : ARIMA

Le modèle ARIMA (AutoRegressive Integrated Moving Average) est la baseline statistique historique pour les séries temporelles. Il modélise la série comme une combinaison linéaire de ses valeurs passées (composante AR), de ses erreurs passées (composante MA), et d'une différenciation pour la stationnarité (composante I).

**Limitations pour le Bitcoin :** ARIMA suppose la linéarité et la stationnarité. Les études montrent systématiquement qu'il est surpassé par les méthodes non-linéaires sur les données crypto. Il reste néanmoins utile comme baseline minimale : tout modèle qui ne bat pas ARIMA n'apporte aucune valeur ajoutée. Certaines études ont cependant montré que dans des conditions spécifiques, ARIMA peut surpasser des modèles plus complexes comme le TFT ou le SVR sur des données normalisées.

### 2.2 Machine Learning classique : XGBoost, SVM, Random Forest

Les algorithmes d'ensemble (XGBoost, LightGBM, Random Forest) et les SVM sont fréquemment utilisés comme baselines ML. Ils ne modélisent pas nativement la temporalité mais peuvent être adaptés en "aplatissant" des fenêtres temporelles en vecteurs de features.

**Points forts :** robustesse, rapidité d'entraînement, bonne gestion des features bruitées. Une étude de 2025 montre qu'un SVM avec sélection de features Boruta atteint 83% de précision directionnelle et se révèle être le modèle le plus rentable en backtesting. XGBoost atteint régulièrement 65-67% de précision sur les données à haute fréquence (intervalles de 5 minutes).

**Limites :** perte de la structure séquentielle lors de l'aplatissement, difficulté à capturer les dépendances à très long terme.

### 2.3 Réseaux Récurrents : LSTM et GRU

#### LSTM (Long Short-Term Memory)

Le LSTM est l'architecture la plus étudiée et la plus citée pour la prédiction du Bitcoin. Sa cellule mémoire avec trois portes (forget, input, output) lui permet de retenir sélectivement l'information sur de longues périodes, résolvant le problème du gradient évanescent des RNN classiques.

**Résultats clés dans la littérature :**
- Dans une comparaison systématique sur cinq cryptomonnaies (BTC, ETH, XRP, LTC, XMR), le LSTM obtient les meilleures performances avec un RMSE moyen de 0.0222, soit 2.7% de mieux que le deuxième modèle.
- Les modèles récurrents occupent systématiquement les trois premières places du classement, devant les CNN et les Transformers.
- Un LSTM avec optimisation bayésienne atteint 52% de précision directionnelle et 8% de RMSE sur le Bitcoin Price Index, surpassant ARIMA et les RNN classiques.

#### GRU (Gated Recurrent Unit)

Le GRU simplifie le LSTM en fusionnant les portes forget et input en une seule porte de mise à jour. Il possède environ 25% de paramètres en moins, ce qui se traduit par un entraînement plus rapide et un risque de surapprentissage réduit sur des datasets de taille modeste.

**Résultats :** performances souvent comparables au LSTM, parfois supérieures sur des séquences courtes. Recommandé quand la rapidité d'entraînement est prioritaire.

#### BiLSTM (Bidirectional LSTM)

Le BiLSTM traite la séquence dans les deux sens temporels (passé → futur et futur → passé). Bien que le sens "futur → passé" puisse sembler contre-intuitif pour la prédiction, il permet au modèle de capturer des patterns symétriques et d'améliorer la représentation de chaque pas de temps.

**Résultats :** une approche Performer + BiLSTM avec indicateurs techniques montre des améliorations significatives du RMSE sur les données horaires et journalières du BTC.

### 2.4 Architectures Convolutionnelles : CNN-LSTM et TCN

#### CNN-LSTM (Hybride)

Cette architecture combine un CNN (pour l'extraction de features locales) avec un LSTM (pour la modélisation temporelle). Les filtres convolutionnels détectent des motifs courts (patterns de chandeliers, pics de volume, breakouts) que le LSTM assemble ensuite en une compréhension globale de la dynamique.

**Résultats clés :** la combinaison de la sélection de features Boruta avec le CNN-LSTM surpasse systématiquement les autres combinaisons, atteignant une précision directionnelle de 82.44%. Le CNN-LSTM est particulièrement efficace quand les données contiennent des patterns locaux forts (données intraday avec patterns techniques clairs).

#### TCN (Temporal Convolutional Network)

Le TCN est une alternative purement convolutionnelle aux RNN. Il utilise des convolutions causales dilatées dont le facteur de dilatation double à chaque couche (1, 2, 4, 8...), permettant d'atteindre un champ réceptif exponentiel avec un nombre linéaire de paramètres.

**Avantages :** parallélisable sur GPU (contrairement aux RNN qui sont séquentiels), entraînement plus stable, champ réceptif explicitement contrôlable. Le TCN a montré son efficacité pour la prédiction hebdomadaire de l'Ethereum et du Bitcoin.

### 2.5 Architectures à Attention : Transformer et TFT

#### Transformer Encoder

Le Transformer, introduit pour la traduction automatique, a été adapté aux séries temporelles. Son mécanisme d'attention multi-têtes permet de capturer des dépendances temporelles complexes sans récurrence : chaque position peut directement "regarder" n'importe quelle autre position dans la séquence.

**Résultats :** une comparaison de 2025 sur six architectures deep learning (LSTM, GPT-2, Informer, Autoformer, TFT, Vanilla Transformer) montre des performances variables selon la cryptomonnaie et l'horizon de prédiction. Le Transformer est compétitif mais ne domine pas systématiquement les RNN.

#### TFT (Temporal Fusion Transformer)

Le TFT, introduit par Lim et al. (2021), est considéré comme l'architecture la plus avancée pour les séries temporelles multi-variées. Il combine :

1. **Variable Selection Network** : apprend automatiquement quelles features sont importantes
2. **LSTM Encoder** : capture les dépendances temporelles
3. **Multi-Head Attention interprétable** : focalise sur les pas de temps pertinents
4. **Gated Residual Networks** : connexions résiduelle adaptatives

**Résultats clés :**
- Une étude de 2025 identifie le TFT comme l'approche prédictive la plus performante pour le Bitcoin, avec le prix de l'or comme facteur le plus déterminant.
- Un TFT adaptatif avec segmentation dynamique surpasse significativement le LSTM standard et le TFT standard sur les données à 10 minutes de l'ETH-USDT.
- Des TFT avec catégorisation de séries temporelles génèrent plus de 6% de profit supplémentaire sur deux semaines par rapport à une stratégie buy-and-hold.
- Une étude benchmarking le TFT contre LSTM, GRU, SVR et XGBoost sur cinq cryptomonnaies montre la supériorité du TFT en termes de RMSE, MAE et R².

---

## 3. Synthèse Comparative des Architectures

| Architecture | Forces | Faiblesses | Cas d'usage optimal |
|---|---|---|---|
| **ARIMA** | Simple, interprétable, baseline | Linéaire, ne capture pas la non-stationnarité | Baseline minimale, tendances lisses |
| **XGBoost** | Rapide, robuste au bruit, pas de GPU | Perd la structure séquentielle | Prédiction avec beaucoup de features tabulaires |
| **LSTM** | Référence, dépendances long-terme | Entraînement séquentiel, lent | Usage général, séries temporelles longues |
| **GRU** | Plus léger que LSTM, rapide | Légèrement moins expressif | Quand le dataset est petit ou la vitesse prime |
| **BiLSTM** | Patterns symétriques, meilleure représentation | Double les paramètres | Quand le contexte bidirectionnel aide |
| **CNN-LSTM** | Extraction automatique de features locales | Plus complexe à calibrer | Données avec patterns techniques forts |
| **TCN** | Parallélisable, stable, explicite | Champ réceptif fixe | Prédiction à horizon fixe, entraînement rapide |
| **Transformer** | Attention globale, parallélisable | Gourmand en données, peu interprétable | Séquences longues avec dépendances distantes |
| **TFT** | Interprétable, sélection de variables, SOTA | Complexe, beaucoup de paramètres | Prédiction multi-horizon avec features variées |

---

## 4. Architecture Python Implémentée

### 4.1 Structure du Projet

Le projet est organisé de manière modulaire pour faciliter l'expérimentation :

```
btc_prediction/
├── config.py                    # Hyperparamètres centralisés
├── main.py                      # Pipeline d'orchestration
├── requirements.txt             # Dépendances
├── models/
│   ├── lstm_model.py            # LSTM & BiLSTM (PyTorch)
│   ├── gru_model.py             # GRU (PyTorch)
│   ├── cnn_lstm_model.py        # Hybride CNN-LSTM (PyTorch)
│   ├── tcn_model.py             # TCN avec convolutions dilatées (PyTorch)
│   ├── transformer_model.py     # Transformer Encoder (PyTorch)
│   ├── tft_model.py             # TFT simplifié (PyTorch)
│   └── xgboost_model.py         # XGBoost wrapper
└── utils/
    ├── preprocessing.py         # Indicateurs techniques + séquences
    ├── metrics.py               # 5 métriques d'évaluation
    └── trainer.py               # Entraîneur unifié avec early stopping
```

### 4.2 Pipeline de Données

Le preprocessing suit un pipeline rigoureux pour éviter le data leakage :

1. **Données brutes OHLCV** → calcul de 19 features (indicateurs techniques)
2. **Split chronologique** (70% train / 15% val / 15% test) — jamais aléatoire
3. **Normalisation Min-Max** fittée uniquement sur le train
4. **Création de séquences** glissantes de 60 jours

Les indicateurs techniques calculés couvrent quatre catégories :
- **Tendance** : SMA(7), SMA(21), EMA(7), EMA(21)
- **Momentum** : RSI(14), MACD, Signal MACD
- **Volatilité** : Bandes de Bollinger, ATR(14), Volatilité sur 21 jours
- **Volume** : OBV (On-Balance Volume)
- **Rendements** : Returns, Log Returns

### 4.3 Détail des Implémentations

#### LSTM et BiLSTM (`models/lstm_model.py`)
Architecture : 2 couches LSTM empilées (hidden_size=128), dropout 20%, puis deux couches denses (64 → 1). Le BiLSTM utilise le même code avec `bidirectional=True`, ce qui double la dimension de sortie.

#### GRU (`models/gru_model.py`)
Identique au LSTM mais avec des cellules GRU qui ont ~25% de paramètres en moins (pas de cell state séparé).

#### CNN-LSTM (`models/cnn_lstm_model.py`)
Deux couches Conv1D (64 filtres, kernel 3) avec BatchNorm et MaxPooling, suivies d'un LSTM (128 hidden). Le CNN extrait des features locales que le LSTM contextualise temporellement.

#### TCN (`models/tcn_model.py`)
4 blocs de convolutions causales dilatées (dilatations 1, 2, 4, 8) avec connexions résiduelle et weight normalization. Champ réceptif total : 48 pas de temps.

#### Transformer (`models/transformer_model.py`)
Projection linéaire vers d_model=128, encodage positionnel sinusoïdal, 3 couches d'encodeur Transformer avec 8 têtes d'attention, activation GELU.

#### TFT simplifié (`models/tft_model.py`)
Implémentation des composants clés : Variable Selection Network (avec GRN par feature + pondération softmax), LSTM Encoder, Multi-Head Attention interprétable, GRN de sortie. Simplifié par rapport au TFT complet (pas de features statiques ni de quantile forecasts).

#### XGBoost (`models/xgboost_model.py`)
Wrapper qui aplatit les séquences 3D en vecteurs 2D. Configuration : 500 estimateurs, profondeur max 6, learning rate 0.05, avec régularisation L1/L2.

### 4.4 Entraînement Unifié

Le module `trainer.py` fournit un entraîneur générique avec :
- **Adam optimizer** avec learning rate adaptatif (ReduceLROnPlateau)
- **Early stopping** (patience=15 époques) avec restauration des meilleurs poids
- **Gradient clipping** (max_norm=1.0) pour la stabilité
- **Sélection automatique du device** (GPU/CPU)

### 4.5 Métriques d'Évaluation

Cinq métriques complémentaires sont implémentées dans `metrics.py` :

| Métrique | Formule | Interprétation |
|---|---|---|
| **MAE** | mean(\|y - ŷ\|) | Erreur moyenne en dollars |
| **RMSE** | √mean((y - ŷ)²) | Pénalise les grosses erreurs |
| **MAPE** | mean(\|y - ŷ\|/y) × 100 | Erreur relative en % |
| **R²** | 1 - SS_res/SS_tot | Variance expliquée (1 = parfait) |
| **Direction Accuracy** | % de directions correctes | La plus pertinente pour le trading |

---

## 5. Recommandations et Prochaines Étapes

### 5.1 Architecture Recommandée

Sur la base de la littérature, voici la recommandation par ordre de priorité :

1. **TFT (Temporal Fusion Transformer)** — meilleur rapport performance/interprétabilité quand on dispose de features variées (on-chain, sentiment, macroéconomie).
2. **LSTM / BiLSTM** — le choix le plus robuste et le mieux documenté, excellent point de départ.
3. **CNN-LSTM avec sélection de features** — particulièrement pertinent avec des indicateurs techniques riches.
4. **XGBoost** — toujours l'entraîner comme baseline. Si le deep learning ne le bat pas significativement, la complexité n'est pas justifiée.

### 5.2 Prochaines Étapes pour les Données Réelles

Quand les données réelles seront disponibles, voici les étapes recommandées :

1. **Sources de données à intégrer :**
   - Prix OHLCV (Binance, CoinGecko, Yahoo Finance)
   - Données on-chain (Glassnode, CryptoQuant) : SOPR, active addresses, exchange flows
   - Sentiment (Twitter/X, Reddit, Fear & Greed Index)
   - Facteurs macro (VIX, prix de l'or, DXY, taux Fed)

2. **Améliorations techniques :**
   - Implémenter le TFT complet avec features statiques et quantile forecasts
   - Ajouter un modèle d'ensemble (stacking) combinant les meilleurs modèles
   - Implémenter le backtesting avec coûts de transaction réalistes
   - Explorer les modèles de type Informer/Autoformer pour les horizons longs

3. **Validation rigoureuse :**
   - Walk-forward validation (pas de simple train/test split)
   - Tests de significativité statistique (Diebold-Mariano)
   - Analyse de robustesse aux changements de régime de marché

---

## 6. Références Bibliographiques

1. Köse, N., Gür, Y. & Ünal, E. (2025). Deep Learning and Machine Learning Insights Into the Global Economic Drivers of the Bitcoin Price. *Journal of Forecasting*, 44, 1666-1698.
2. Lim, B., Arık, S.O., Loeff, N. & Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *Int. J. Forecast.*, 37, 1748-1764.
3. Bai, S., Kolter, J.Z. & Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. *arXiv:1803.01271*.
4. Critien, J.V., Gatt, A. & Ellul, J. (2024). Deep learning for Bitcoin price direction prediction. *Financial Innovation*, 10.
5. Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.
6. Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.
7. Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS 2017*.
8. Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder. *EMNLP 2014*.
9. Frontiers in AI (2025). Predicting the Bitcoin's price using AI. *Frontiers in Artificial Intelligence*, 8.
10. ScienceDirect (2025). Using machine and deep learning models, on-chain data, and technical analysis for predicting bitcoin price direction and magnitude.

---

*Ce rapport a été généré dans le cadre d'une analyse comparative des architectures ML/DL pour la prédiction du prix du Bitcoin. Le code Python complet est disponible dans le répertoire `btc_prediction/`.*
