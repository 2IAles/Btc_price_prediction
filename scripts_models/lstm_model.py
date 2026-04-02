import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """LSTM pour la classification binaire de séries temporelles.
    Peut fonctionner en mode unidirectionnel ou bidirectionnel (BiLSTM).
    """

    def __init__(
        self,
        input_size: int,       # nombre de features par pas de temps
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        # Si bidirectionnel, la sortie est 2x plus grande (avant + arrière)
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # format (batch, séquence, features)
            dropout=dropout if num_layers > 1 else 0,  # dropout entre couches seulement
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)

        # Tête de classification : projection vers un scalaire (logit)
        fc_input_size = hidden_size * self.num_directions
        self.fc1 = nn.Linear(fc_input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)  # sortie = logit (pas encore sigmoid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Passe la séquence dans le LSTM, h_n = état caché du dernier pas
        lstm_out, (h_n, c_n) = self.lstm(x)

        if self.bidirectional:
            # Concatène les états cachés des deux directions (avant et arrière)
            last_output = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            # Prend uniquement la sortie du dernier pas de temps
            last_output = lstm_out[:, -1, :]

        out = self.dropout(last_output)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)  # logit brut (la sigmoid est dans la loss BCEWithLogitsLoss)
        return out
