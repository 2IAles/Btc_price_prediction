import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):

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
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=cnn_filters,
            kernel_size=cnn_kernel_size,
            padding=cnn_kernel_size // 2,
        )
        self.bn1 = nn.BatchNorm1d(cnn_filters)

        self.conv2 = nn.Conv1d(
            in_channels=cnn_filters,
            out_channels=cnn_filters,
            kernel_size=cnn_kernel_size,
            padding=cnn_kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm1d(cnn_filters)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.permute(0, 2, 1)

        x = self.relu(self.bn1(self.conv1(x)))

        x = self.relu(self.bn2(self.conv2(x)))

        x = self.pool(x)

        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]

        out = self.dropout(last_output)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out
