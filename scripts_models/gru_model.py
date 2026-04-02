import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """GRU pour la classification binaire de séries temporelles.
    Le GRU est une version simplifiée du LSTM avec moins de paramètres.
    """

    def __init__(
        self,
        input_size: int,       # nombre de features par pas de temps
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
            batch_first=True,  # format (batch, séquence, features)
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        # Tête de classification identique au LSTM
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Le GRU retourne la sortie complète et l'état caché final h_n
        gru_out, h_n = self.gru(x)

        # Prend la sortie du dernier pas de temps de la séquence
        last_output = gru_out[:, -1, :]

        out = self.dropout(last_output)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)  # logit brut
        return out
