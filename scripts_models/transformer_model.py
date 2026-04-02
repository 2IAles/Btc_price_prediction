import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Ajoute un encodage de position à chaque pas de temps.
    Nécessaire car le Transformer n'a pas de notion d'ordre intrinsèque (pas de récurrence).
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Matrice d'encodage de position : shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Terme de division pour les fréquences sin/cos (formule originale du papier "Attention is All You Need")
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Indices pairs = sinus, indices impairs = cosinus
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # ajoute la dimension batch

        # register_buffer : pas un paramètre entraînable, mais sauvegardé avec le modèle
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ajoute l'encodage de position aux embeddings d'entrée
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Transformer encoder pour la classification binaire de séries temporelles.
    L'attention multi-têtes permet de capturer des dépendances à longue portée.
    """

    def __init__(
        self,
        input_size: int,           # nombre de features par pas de temps
        d_model: int = 128,        # dimension interne du transformer
        nhead: int = 8,            # nombre de têtes d'attention (d_model doit être divisible)
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Projection linéaire : aligne la dimension des features sur d_model
        self.input_projection = nn.Linear(input_size, d_model)

        # Encodage positionnel : indique au modèle l'ordre des pas de temps
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Couche d'encodage Transformer avec activation GELU (plus douce que ReLU)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        # Empilement de plusieurs couches d'encodeur
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        # Tête de classification
        self.fc1 = nn.Linear(d_model, 64)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Projette les features dans l'espace d_model
        x = self.input_projection(x)

        # Injecte l'information de position dans les embeddings
        x = self.pos_encoder(x)

        # Traitement par les couches d'attention (chaque pas de temps "voit" tous les autres)
        x = self.transformer_encoder(x)

        # Prend la représentation du dernier pas de temps pour la prédiction
        x = x[:, -1, :]

        x = self.dropout_layer(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # logit brut
        return x
