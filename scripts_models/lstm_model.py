"""
Modèle LSTM (Long Short-Term Memory) pour la prédiction du prix du Bitcoin.

L'LSTM est l'architecture la plus citée dans la littérature pour la prédiction
de séries temporelles financières. Son avantage principal est sa capacité à
capturer les dépendances à long terme grâce à ses trois portes (forget, input, output)
qui contrôlent le flux d'information à travers la cellule mémoire.

Architecture :
    Input → LSTM(128, 2 couches) → Dropout → Dense(64) → Dense(1)

Références :
    - Hochreiter & Schmidhuber (1997) — article fondateur
    - Köse et al. (2025) — LSTM pour BTC avec facteurs macroéconomiques
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    Réseau LSTM empilé pour la régression de séries temporelles.
    
    L'empilement de plusieurs couches LSTM permet au réseau d'apprendre
    des représentations à différents niveaux d'abstraction :
    - Couche 1 : patterns locaux (tendances courtes, pics de volume)
    - Couche 2 : patterns globaux (cycles de marché, momentum)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Couches LSTM empilées avec dropout entre les couches
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,           # Format (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Couche de régularisation
        self.dropout = nn.Dropout(dropout)

        # Couches de sortie : projection vers une seule valeur (le prix)
        fc_input_size = hidden_size * self.num_directions
        self.fc1 = nn.Linear(fc_input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor de forme (batch_size, seq_length, n_features)
        
        Returns:
            Prédiction de forme (batch_size, 1)
        """
        # L'LSTM traite toute la séquence et produit un hidden state par pas de temps
        # On récupère seulement le dernier hidden state (résumé de toute la séquence)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Prend la sortie du dernier pas de temps
        if self.bidirectional:
            # Concatène les hidden states des deux directions
            last_output = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_output = lstm_out[:, -1, :]

        # Projection vers la prédiction
        out = self.dropout(last_output)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out
