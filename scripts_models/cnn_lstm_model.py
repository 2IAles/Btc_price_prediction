"""
Modèle hybride CNN-LSTM pour la prédiction du prix du Bitcoin.

Cette architecture combine les forces de deux familles de réseaux :
    
    1. CNN (Convolutional Neural Network) : 
       Excellent pour détecter des patterns locaux dans les données.
       Ici, les convolutions 1D glissent le long de la dimension temporelle
       pour extraire des motifs courts (pics de volume, breakouts, patterns de chandeliers).
    
    2. LSTM :
       Capture les dépendances à long terme entre ces patterns locaux.
       Comprend comment les motifs courts s'enchaînent pour former des tendances.

L'idée est que le CNN sert de "feature extractor" automatique qui transforme
les données brutes en représentations de haut niveau, que le LSTM utilise
ensuite pour modéliser la dynamique temporelle.

Architecture :
    Input → Conv1D(64) → BatchNorm → ReLU → MaxPool → LSTM(128) → Dense(1)

Références :
    - Ortu et al. (2022) — CNN-LSTM pour crypto
    - Résultats CNN-LSTM surpassant LSTM seul dans plusieurs benchmarks (82.44% accuracy)
"""

import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):
    """
    Réseau hybride où les couches convolutionnelles extraient des features locales
    puis un LSTM modélise les dépendances temporelles entre ces features.
    """
    
    def __init__(
        self,
        input_size: int,
        cnn_filters: int = 64,
        cnn_kernel_size: int = 3,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        # --- Bloc CNN ---
        # Conv1D : opère sur la dimension temporelle
        # kernel_size=3 signifie que chaque filtre regarde 3 pas de temps à la fois
        self.conv1 = nn.Conv1d(
            in_channels=input_size,    # Nombre de features en entrée
            out_channels=cnn_filters,  # Nombre de filtres (features extraites)
            kernel_size=cnn_kernel_size,
            padding=cnn_kernel_size // 2,  # Padding pour conserver la longueur
        )
        self.bn1 = nn.BatchNorm1d(cnn_filters)  # Stabilise l'entraînement

        self.conv2 = nn.Conv1d(
            in_channels=cnn_filters,
            out_channels=cnn_filters,
            kernel_size=cnn_kernel_size,
            padding=cnn_kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm1d(cnn_filters)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # Réduit la résolution temporelle

        # --- Bloc LSTM ---
        self.lstm = nn.LSTM(
            input_size=cnn_filters,       # Entrée = sorties du CNN
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
        )

        # --- Bloc de sortie ---
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Flux de données :
            (batch, seq, features) → Conv1D attend (batch, channels, seq)
            → On transpose, applique les convolutions, retranspose
            → LSTM traite la séquence de features extraites
            → Couches denses produisent la prédiction
        """
        # Conv1D attend (batch, channels, seq_length), on transpose
        x = x.permute(0, 2, 1)  # (batch, features, seq) 

        # Bloc convolutionnel 1
        x = self.relu(self.bn1(self.conv1(x)))
        # Bloc convolutionnel 2
        x = self.relu(self.bn2(self.conv2(x)))
        # Pooling pour réduire la dimension temporelle
        x = self.pool(x)

        # Retranspose pour le LSTM : (batch, seq, features)
        x = x.permute(0, 2, 1)

        # LSTM sur les features extraites
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]

        # Prédiction
        out = self.dropout(last_output)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out
