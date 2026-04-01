"""
Métriques d'évaluation pour la prédiction de séries temporelles financières.

On utilise plusieurs métriques complémentaires car chacune capture un aspect
différent de la qualité des prédictions :
- MAE : erreur moyenne absolue, facile à interpréter en dollars
- RMSE : pénalise davantage les grosses erreurs
- MAPE : erreur relative en pourcentage
- R² : variance expliquée par le modèle
- Direction Accuracy : capacité à prédire si le prix monte ou descend
"""

import numpy as np
from typing import Dict


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error — erreur moyenne en valeur absolue."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error — pénalise les grosses déviations."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error — erreur relative moyenne."""
    # Filtre les zéros pour éviter la division par zéro
    mask = y_true != 0
    if mask.sum() == 0:
        return float("inf")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient de détermination R².
    
    R² = 1 signifie que le modèle explique 100% de la variance.
    R² = 0 signifie que le modèle fait aussi bien qu'une prédiction constante (la moyenne).
    R² < 0 signifie que le modèle fait pire que la moyenne.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def compute_direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Précision directionnelle — pourcentage de fois où le modèle prédit
    correctement la direction du mouvement (hausse vs baisse).
    
    C'est la métrique la plus importante pour le trading :
    même si l'amplitude est fausse, si la direction est bonne,
    la stratégie peut être profitable.
    """
    if len(y_true) < 2:
        return 0.0
    # Direction réelle et prédite (variation par rapport au pas précédent)
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    # Pourcentage de directions correctes
    correct = np.sum(true_direction == pred_direction)
    return float(correct / len(true_direction) * 100)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Calcule toutes les métriques d'évaluation et retourne un dictionnaire structuré.
    
    Args:
        y_true: Valeurs réelles (dé-normalisées, en dollars)
        y_pred: Valeurs prédites (dé-normalisées, en dollars)
        model_name: Nom du modèle pour l'affichage
    
    Returns:
        Dictionnaire avec toutes les métriques
    """
    results = {
        "model": model_name,
        "MAE": compute_mae(y_true, y_pred),
        "RMSE": compute_rmse(y_true, y_pred),
        "MAPE (%)": compute_mape(y_true, y_pred),
        "R²": compute_r2(y_true, y_pred),
        "Direction Accuracy (%)": compute_direction_accuracy(y_true, y_pred),
    }
    return results


def print_evaluation(results: Dict[str, float]):
    """Affiche les résultats de manière lisible."""
    print(f"\n{'='*60}")
    print(f"  Résultats : {results['model']}")
    print(f"{'='*60}")
    print(f"  MAE                  : {results['MAE']:,.2f} $")
    print(f"  RMSE                 : {results['RMSE']:,.2f} $")
    print(f"  MAPE                 : {results['MAPE (%)']:.2f} %")
    print(f"  R²                   : {results['R²']:.4f}")
    print(f"  Direction Accuracy   : {results['Direction Accuracy (%)']:.2f} %")
    print(f"{'='*60}\n")
