import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from typing import List


class CausalConv1dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation

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

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        out = self.relu(out)
        out = self.dropout(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return self.relu(out + residual)


class TCNModel(nn.Module):

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
            dilation = 2**i
            in_ch = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(
                CausalConv1dBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )

        self.tcn = nn.Sequential(*layers)
        self.fc1 = nn.Linear(num_channels[-1], 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.permute(0, 2, 1)

        x = self.tcn(x)

        x = x[:, :, -1]

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
