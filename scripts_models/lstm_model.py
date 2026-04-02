import torch
import torch.nn as nn


class LSTMModel(nn.Module):

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

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)

        fc_input_size = hidden_size * self.num_directions
        self.fc1 = nn.Linear(fc_input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        lstm_out, (h_n, c_n) = self.lstm(x)

        if self.bidirectional:
            last_output = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_output = lstm_out[:, -1, :]

        out = self.dropout(last_output)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out
