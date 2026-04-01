"""
Modèle TFT simplifié (Temporal Fusion Transformer) pour la prédiction du prix du Bitcoin.

Le TFT, introduit par Lim et al. (2021), est l'une des architectures les plus avancées
pour la prédiction de séries temporelles. Il combine plusieurs mécanismes :

    1. Variable Selection Network (VSN) :
       Apprend automatiquement quelles features sont importantes à chaque pas de temps.
       Par exemple, le modèle pourrait apprendre que le RSI est crucial quand le marché
       est en surachat, mais que le volume est plus important en période de breakout.

    2. LSTM Encoder-Decoder :
       Capture les dépendances temporelles, comme dans un LSTM classique,
       mais enrichi par les features sélectionnées dynamiquement.

    3. Multi-Head Attention (Interpretable) :
       Permet au modèle de focaliser sur les pas de temps les plus pertinents
       pour la prédiction. Les poids d'attention sont interprétables.

    4. Gated Residual Network (GRN) :
       Des connexions résiduelle avec portes qui permettent au réseau de
       "sauter" des transformations non nécessaires, s'adaptant à la complexité
       réelle des données.

Cette implémentation est simplifiée par rapport au TFT complet mais conserve
les composants essentiels. Un TFT complet utiliserait aussi des features statiques,
des covariables futures connues, et des quantile forecasts.

Références :
    - Lim et al. (2021) — "Temporal Fusion Transformers for Interpretable Multi-horizon
      Time Series Forecasting"
    - Köse et al. (2025) — TFT identifié comme meilleur modèle pour BTC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) — brique de base du TFT.
    
    Le GRN applique une transformation non-linéaire avec un mécanisme de porte
    (gate) qui contrôle combien de la transformation est utilisée. Si les données
    sont simples, la porte peut "fermer" et laisser passer directement l'entrée
    via la connexion résiduelle.
    
    Schéma : input → Dense → ELU → Dense → GLU × (input + skip connection)
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Gated Linear Unit : produit une porte sigmoïde × une valeur linéaire
        self.gate = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)
        
        # Skip connection (ajuste les dimensions si nécessaire)
        self.skip_connection = nn.Linear(input_size, output_size) \
            if input_size != output_size else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass avec GLU et connexion résiduelle.
        """
        # Connexion résiduelle
        residual = self.skip_connection(x) if self.skip_connection else x
        
        # Transformation non-linéaire
        hidden = self.elu(self.fc1(x))
        hidden = self.dropout(hidden)
        
        # Gated Linear Unit : gate contrôle le flux d'information
        output = self.fc2(hidden)
        gate = self.sigmoid(self.gate(hidden))
        gated_output = gate * output
        
        # Résiduel + normalisation
        return self.layer_norm(gated_output + residual)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) — sélectionne dynamiquement les features pertinentes.
    
    Pour chaque pas de temps, le VSN calcule des "poids d'importance" softmax
    pour chaque feature, puis pondère les features par ces poids.
    
    C'est ce qui rend le TFT interprétable : on peut regarder les poids
    pour comprendre quelles features le modèle considère importantes.
    """
    
    def __init__(self, input_size: int, n_features: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        
        # Un GRN par feature pour transformer chaque feature individuellement
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_size, hidden_size, dropout)
            for _ in range(n_features)
        ])
        
        # GRN pour calculer les poids de sélection (attention sur les features)
        self.selection_grn = GatedResidualNetwork(
            input_size, hidden_size, n_features, dropout
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            (batch, seq_len, hidden_size) — features pondérées et transformées
        """
        batch_size, seq_len, _ = x.shape
        
        # Calcul des poids de sélection
        # Flatten pour passer dans le GRN : (batch * seq, n_features)
        flat_x = x.reshape(-1, self.n_features)
        weights = self.selection_grn(flat_x)
        weights = self.softmax(weights)  # (batch * seq, n_features)
        weights = weights.reshape(batch_size, seq_len, self.n_features)
        
        # Transformation individuelle de chaque feature
        transformed_features = []
        for i, grn in enumerate(self.feature_grns):
            feat = x[:, :, i:i+1]  # (batch, seq, 1)
            feat_flat = feat.reshape(-1, 1)
            transformed = grn(feat_flat)  # (batch * seq, hidden_size)
            transformed = transformed.reshape(batch_size, seq_len, self.hidden_size)
            transformed_features.append(transformed)
        
        # Stack et pondération : chaque feature contribue selon son poids
        stacked = torch.stack(transformed_features, dim=-1)  # (batch, seq, hidden, n_features)
        weights_expanded = weights.unsqueeze(2).expand_as(stacked)
        selected = (stacked * weights_expanded).sum(dim=-1)  # (batch, seq, hidden)
        
        return selected


class SimplifiedTFT(nn.Module):
    """
    Temporal Fusion Transformer simplifié.
    
    Pipeline :
        1. Variable Selection → sélectionne les features importantes
        2. LSTM Encoding → capture les dépendances temporelles  
        3. Multi-Head Attention → focalise sur les pas de temps pertinents
        4. GRN + Dense → produit la prédiction finale
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        lstm_layers: int = 1,
        attention_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # 1. Variable Selection Network
        self.vsn = VariableSelectionNetwork(
            input_size=input_size,
            n_features=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        # 2. LSTM Encoder pour les dépendances temporelles
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # 3. Multi-Head Attention interprétable
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_size)

        # 4. GRN de sortie + couches denses
        self.output_grn = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout
        )
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass complet du TFT simplifié.
        """
        # 1. Sélection des variables
        selected = self.vsn(x)  # (batch, seq, hidden)

        # 2. Encodage temporel via LSTM
        lstm_out, _ = self.lstm(selected)  # (batch, seq, hidden)

        # 3. Self-attention sur la séquence encodée
        # L'attention permet au modèle de peser l'importance relative
        # de chaque pas de temps pour la prédiction finale
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        # Connexion résiduelle + normalisation
        attn_out = self.attention_norm(attn_out + lstm_out)

        # 4. Prédiction à partir du dernier pas de temps
        last_output = attn_out[:, -1, :]
        output = self.output_grn(last_output)
        output = self.relu(self.fc1(output))
        output = self.fc2(output)
        
        return output
