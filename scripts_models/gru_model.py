"""
Modèle GRU (Gated Recurrent Unit) pour la prédiction du prix du Bitcoin.

Le GRU est une variante simplifiée du LSTM qui fusionne les portes forget et input
en une seule "porte de mise à jour" (update gate), et utilise une "porte de reset"
pour contrôler combien d'information passée est oubliée.

Avantages par rapport au LSTM :
    - Moins de paramètres → entraînement plus rapide
    - Performance souvent comparable ou supérieure sur des séquences courtes/moyennes
    - Moins sujet au surapprentissage sur de petits datasets

Architecture :
    Input → GRU(128, 2 couches) → Dropout → Dense(64) → Dense(1)

Références :
    - Cho et al. (2014) — article fondateur du GRU
    - Hong, Yan & Chen (2022) — GRU pour crypto avec moins de coût computationnel
"""

import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """
    Réseau GRU empilé pour la régression de séries temporelles.
    
    Le GRU apprend les mêmes types de dépendances temporelles que le LSTM
    mais avec environ 25% de paramètres en moins, ce qui en fait un bon
    compromis entre expressivité et efficacité.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass — identique au LSTM sauf qu'il n'y a pas de cell state.
        Le GRU n'utilise qu'un seul état caché (h_n) au lieu de (h_n, c_n).
        """
        gru_out, h_n = self.gru(x)
        
        # Dernier pas de temps
        last_output = gru_out[:, -1, :]
        
        out = self.dropout(last_output)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out
