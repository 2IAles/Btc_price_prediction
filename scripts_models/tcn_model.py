import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from typing import List


class CausalConv1dBlock(nn.Module):
    """Bloc de convolution causale dilatée avec connexion résiduelle.
    Causale = ne regarde pas dans le futur. Dilatée = champ récepteur exponentiel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,       # puissance de 2 croissante : 1, 2, 4, 8...
        dropout: float = 0.2,
    ):
        super().__init__()
        # Padding à gauche uniquement pour garantir la causalité (ne voit pas le futur)
        self.padding = (kernel_size - 1) * dilation

        # weight_norm : stabilise l'entraînement en normalisant les poids des convolutions
        self.conv1 = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                padding=self.padding,
            )
        )
        self.conv2 = weight_norm(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                padding=self.padding,
            )
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Connexion résiduelle : projection 1x1 si les canaux d'entrée/sortie diffèrent
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Première convolution causale dilatée
        out = self.conv1(x)
        if self.padding > 0:
            # Supprime les pas de temps "futurs" générés par le padding
            out = out[:, :, : -self.padding]
        out = self.relu(out)
        out = self.dropout(out)

        # Deuxième convolution causale dilatée
        out = self.conv2(out)
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        out = self.relu(out)
        out = self.dropout(out)

        # Connexion résiduelle : adapte les dimensions si nécessaire
        if self.downsample is not None:
            residual = self.downsample(residual)

        # Somme résiduelle + activation
        return self.relu(out + residual)


class TCNModel(nn.Module):
    """Temporal Convolutional Network : empilement de convolutions causales dilatées.
    Avantage : parallélisable (pas de récurrence), champ récepteur exponentiel.
    """

    def __init__(
        self,
        input_size: int,
        num_channels: List[int] = None,   # canaux pour chaque niveau de dilatation
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64, 64, 64]

        layers = []
        num_levels = len(num_channels)

        # Construction des blocs : la dilatation double à chaque niveau (2^i)
        for i in range(num_levels):
            dilation = 2**i  # 1, 2, 4, 8... → champ récepteur exponentiel
            in_ch = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(
                CausalConv1dBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )

        self.tcn = nn.Sequential(*layers)
        # Tête de classification
        self.fc1 = nn.Linear(num_channels[-1], 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Le TCN attend (batch, canaux, longueur), on permute depuis (batch, longueur, features)
        x = x.permute(0, 2, 1)

        # Passage dans les couches de convolution causale dilatée
        x = self.tcn(x)

        # Prend les features du dernier pas de temps (le plus récent)
        x = x[:, :, -1]

        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # logit brut
        return x
