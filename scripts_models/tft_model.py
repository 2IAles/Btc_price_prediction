import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedResidualNetwork(nn.Module):

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.1
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.gate = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)

        self.skip_connection = (
            nn.Linear(input_size, output_size) if input_size != output_size else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        residual = self.skip_connection(x) if self.skip_connection else x

        hidden = self.elu(self.fc1(x))
        hidden = self.dropout(hidden)

        output = self.fc2(hidden)
        gate = self.sigmoid(self.gate(hidden))
        gated_output = gate * output

        return self.layer_norm(gated_output + residual)


class VariableSelectionNetwork(nn.Module):

    def __init__(
        self, input_size: int, n_features: int, hidden_size: int, dropout: float = 0.1
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.feature_grns = nn.ModuleList(
            [
                GatedResidualNetwork(1, hidden_size, hidden_size, dropout)
                for _ in range(n_features)
            ]
        )

        self.selection_grn = GatedResidualNetwork(
            input_size, hidden_size, n_features, dropout
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch_size, seq_len, _ = x.shape

        flat_x = x.reshape(-1, self.n_features)
        weights = self.selection_grn(flat_x)
        weights = self.softmax(weights)
        weights = weights.reshape(batch_size, seq_len, self.n_features)

        transformed_features = []
        for i, grn in enumerate(self.feature_grns):
            feat = x[:, :, i : i + 1]
            feat_flat = feat.reshape(-1, 1)
            transformed = grn(feat_flat)
            transformed = transformed.reshape(batch_size, seq_len, self.hidden_size)
            transformed_features.append(transformed)

        stacked = torch.stack(transformed_features, dim=-1)
        weights_expanded = weights.unsqueeze(2).expand_as(stacked)
        selected = (stacked * weights_expanded).sum(dim=-1)

        return selected


class SimplifiedTFT(nn.Module):

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

        self.vsn = VariableSelectionNetwork(
            input_size=input_size,
            n_features=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_size)

        self.output_grn = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout
        )
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        selected = self.vsn(x)

        lstm_out, _ = self.lstm(selected)

        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        attn_out = self.attention_norm(attn_out + lstm_out)

        last_output = attn_out[:, -1, :]
        output = self.output_grn(last_output)
        output = self.relu(self.fc1(output))
        output = self.fc2(output)

        return output
