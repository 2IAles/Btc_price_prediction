"""
Métriques d'évaluation pour la classification binaire (label_dir_1d).

Métrique principale : F1-score (classe 1 = hausse)
Métriques secondaires : Accuracy, AUC-ROC, Precision, Recall
"""

import numpy as np
from typing import Dict
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score,
    recall_score, accuracy_score, confusion_matrix,
)


def evaluate_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    threshold: float = 0.5,
) -> Dict:
    """
    Calcule toutes les métriques de classification et retourne un dictionnaire.

    Args:
        y_true  : Labels réels (0 ou 1)
        y_prob  : Probabilités prédites pour la classe 1 (après sigmoid)
        model_name : Nom du modèle pour l'affichage
        threshold  : Seuil de décision (défaut 0.5)

    Returns:
        Dictionnaire avec toutes les métriques
    """
    y_pred = (y_prob >= threshold).astype(int)

    results = {
        "model":     model_name,
        "F1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "Accuracy":  float(accuracy_score(y_true, y_pred)),
        "AUC-ROC":   float(roc_auc_score(y_true, y_prob)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "ConfMatrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return results


def print_evaluation(results: Dict):
    """Affiche les résultats de manière lisible."""
    print(f"\n{'='*55}")
    print(f"  Résultats : {results['model']}")
    print(f"{'='*55}")
    print(f"  F1-score (classe 1)  : {results['F1']:.4f}  ← métrique principale")
    print(f"  Accuracy             : {results['Accuracy']:.4f}")
    print(f"  AUC-ROC              : {results['AUC-ROC']:.4f}")
    print(f"  Precision            : {results['Precision']:.4f}")
    print(f"  Recall               : {results['Recall']:.4f}")
    cm = np.array(results["ConfMatrix"])
    print(f"  Confusion Matrix     :")
    print(f"    TN={cm[0,0]:>5}  FP={cm[0,1]:>5}")
    print(f"    FN={cm[1,0]:>5}  TP={cm[1,1]:>5}")
    print(f"{'='*55}\n")
