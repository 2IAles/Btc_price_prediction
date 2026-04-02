"""
Modèle Transformer pour la prédiction du prix du Bitcoin.

Le Transformer, introduit par Vaswani et al. (2017) pour la traduction automatique,
a été adapté aux séries temporelles. Son mécanisme d'attention multi-têtes permet
de capturer des dépendances temporelles complexes SANS récurrence.

Différence fondamentale avec les RNN :
    - Un LSTM lit la séquence pas à pas (t1 → t2 → t3 → ...)
    - Un Transformer voit toute la séquence simultanément et apprend quels
      pas de temps sont importants pour la prédiction (via l'attention)

Cela signifie que le Transformer peut directement relier un événement à t=5
à un effet à t=55, sans devoir propager l'information à travers 50 cellules.

Encodage positionnel :
    Puisque le Transformer n'a pas de notion native de l'ordre temporel,
    on ajoute un encodage sinusoïdal qui encode la position de chaque pas
    de temps dans la séquence.

Architecture :
    Input → Positional Encoding → Transformer Encoder (3 couches) → Dense(1)

Références :
    - Vaswani et al. (2017) — "Attention Is All You Need"
    - Études récentes montrant les Transformers compétitifs sur BTC/ETH
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Encodage positionnel sinusoïdal — donne au Transformer la notion d'ordre temporel.
    
    Chaque position reçoit un vecteur unique basé sur des fonctions sin/cos
    à différentes fréquences. Les positions proches ont des encodages similaires,
    et les positions éloignées ont des encodages différents.
    
    Formule : PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
              PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Pré-calcul de la matrice d'encodage positionnel
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) pour le broadcasting

        # Enregistré comme buffer (pas un paramètre, mais sauvegardé avec le modèle)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Ajoute l'encodage positionnel à l'input."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer Encoder adapté aux séries temporelles.
    
    On utilise uniquement l'Encoder (pas le Decoder) car on fait de la régression
    séquence-vers-valeur, pas de la génération séquence-vers-séquence.
    
    Le masque causal est optionnel : pour la prédiction, on peut laisser chaque
    position voir toute la séquence passée (pas de masque), car contrairement
    à la génération de texte, on ne décode pas de manière auto-régressive.
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Projection linéaire pour adapter la dimension des features au d_model
        self.input_projection = nn.Linear(input_size, d_model)

        # Encodage positionnel
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Couches d'encodage Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (batch, seq, features) au lieu de (seq, batch, features)
            activation="gelu",  # GELU souvent meilleur que ReLU pour les Transformers
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        # Couches de sortie
        self.fc1 = nn.Linear(d_model, 64)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Flux : Input → Projection → Pos Encoding → Transformer → Dernière position → Dense
        """
        # Projection vers la dimension du modèle
        x = self.input_projection(x)  # (batch, seq, d_model)

        # Ajout de l'encodage positionnel
        x = self.pos_encoder(x)

        # Passage dans le Transformer Encoder
        x = self.transformer_encoder(x)

        # On prend la sortie de la dernière position temporelle
        # (résume l'attention sur toute la séquence)
        x = x[:, -1, :]

        # Couches de sortie
        x = self.dropout_layer(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
