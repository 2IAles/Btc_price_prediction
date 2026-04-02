"""
Modèle TCN (Temporal Convolutional Network) pour la prédiction du prix du Bitcoin.

Le TCN est une alternative purement convolutionnelle aux architectures récurrentes (LSTM, GRU).
Il utilise des convolutions causales dilatées pour capturer les dépendances temporelles
à différentes échelles sans utiliser de récurrence.

Principe clé — les convolutions dilatées :
    - Dilatation 1 : le filtre voit des pas de temps consécutifs (t-2, t-1, t)
    - Dilatation 2 : le filtre saute un pas (t-4, t-2, t) 
    - Dilatation 4 : le filtre saute 3 pas (t-8, t-4, t)
    - ... et ainsi de suite, doublant le champ réceptif à chaque couche
    
    Cela permet au réseau de "voir" très loin dans le passé avec relativement
    peu de couches, tout en gardant un coût computationnel linéaire.

Avantages :
    - Parallélisable (pas de dépendance séquentielle comme les RNN)
    - Entraînement plus stable (pas de problème de gradient évanescent)
    - Champ réceptif explicitement contrôlable
    - Souvent plus rapide que les LSTM sur GPU

Architecture :
    Input → [CausalConv1D(dilation=1) → CausalConv1D(dilation=2) → ... ] → Dense(1)

Références :
    - Bai, Kolter & Koltun (2018) — article fondateur du TCN
    - Résultats montrant TCN compétitif avec LSTM pour la prédiction financière
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from typing import List


class CausalConv1dBlock(nn.Module):
    """
    Bloc de convolution causale avec connexion résiduelle.
    
    "Causal" signifie que la sortie au temps t ne dépend que des entrées
    aux temps ≤ t. C'est essentiel pour éviter le data leakage : on ne veut
    pas que le modèle "triche" en regardant dans le futur.
    
    La connexion résiduelle (skip connection) permet à l'information de
    traverser directement le bloc si celui-ci n'est pas utile, ce qui facilite
    l'entraînement de réseaux profonds.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Le padding "causal" : on ajoute du padding uniquement à gauche
        # pour que la sortie n'accède pas aux valeurs futures
        self.padding = (kernel_size - 1) * dilation

        # Deux convolutions dilatées empilées (comme dans un ResNet)
        self.conv1 = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding
        ))
        self.conv2 = weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding
        ))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Connexion résiduelle : si les dimensions changent, on ajuste avec 1x1 conv
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass avec padding causal et connexion résiduelle.
        
        Le troncage [:, :, :-self.padding] supprime les valeurs "futures"
        qui ont été calculées à cause du padding symétrique par défaut de PyTorch.
        """
        residual = x

        # Première convolution causale
        out = self.conv1(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]  # Troncature causale
        out = self.relu(out)
        out = self.dropout(out)

        # Deuxième convolution causale
        out = self.conv2(out)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.relu(out)
        out = self.dropout(out)

        # Connexion résiduelle
        if self.downsample is not None:
            residual = self.downsample(residual)

        return self.relu(out + residual)


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network complet avec couches dilatées empilées.
    
    Le champ réceptif total est : kernel_size * 2^(num_layers) 
    Avec kernel_size=3 et 4 couches : 3 * 16 = 48 pas de temps visibles
    """
    
    def __init__(
        self,
        input_size: int,
        num_channels: List[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64, 64, 64]

        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            # La dilatation double à chaque couche : 1, 2, 4, 8, ...
            dilation = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(CausalConv1dBlock(
                in_ch, out_ch, kernel_size, dilation, dropout
            ))

        self.tcn = nn.Sequential(*layers)
        self.fc1 = nn.Linear(num_channels[-1], 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        (batch, seq, features) → transpose → TCN → dernier pas de temps → Dense
        """
        # Conv1D attend (batch, channels, seq_length)
        x = x.permute(0, 2, 1)
        
        # Passage dans les blocs TCN
        x = self.tcn(x)
        
        # Prend la sortie du dernier pas de temps
        x = x[:, :, -1]
        
        # Couches denses de sortie
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
