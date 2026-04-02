import numpy as np
from typing import Dict
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)


def evaluate_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,   # probabilités brutes entre 0 et 1 (pas des classes)
    model_name: str = "Model",
    threshold: float = 0.5,
) -> Dict:
    # Convertit les probabilités en classes binaires selon le seuil
    y_pred = (y_prob >= threshold).astype(int)

    results = {
        "model": model_name,
        # F1 : équilibre entre précision et rappel (métrique principale pour classes déséquilibrées)
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        # Accuracy : taux de bonnes prédictions (peut être trompeur si classes déséquilibrées)
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        # AUC-ROC : capacité à séparer les deux classes, indépendante du seuil
        "AUC-ROC": float(roc_auc_score(y_true, y_prob)),
        # Précision : parmi les prédictions "hausse", combien étaient correctes
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        # Rappel : parmi les vraies hausses, combien ont été détectées
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        # Matrice de confusion : TN, FP, FN, TP
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
    # TN = vrai négatif (baisse prédite, baisse réelle), TP = vrai positif (hausse prédite, hausse réelle)
    print(f"  Confusion Matrix     :")
    print(f"    TN={cm[0,0]:>5}  FP={cm[0,1]:>5}")
    print(f"    FN={cm[1,0]:>5}  TP={cm[1,1]:>5}")
    print(f"{'='*55}\n")
