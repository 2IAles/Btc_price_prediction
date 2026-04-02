import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):
    """Architecture hybride CNN + LSTM.
    Le CNN extrait des patterns locaux dans la séquence,
    puis le LSTM modélise la dynamique temporelle sur ces représentations.
    """

    def __init__(
        self,
        input_size: int,            # nombre de features par pas de temps
        cnn_filters: int = 64,
        cnn_kernel_size: int = 3,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Première couche convolutive : détecte des motifs locaux sur 3 pas de temps
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=cnn_filters,
            kernel_size=cnn_kernel_size,
            padding=cnn_kernel_size // 2,  # padding pour conserver la longueur de séquence
        )
        self.bn1 = nn.BatchNorm1d(cnn_filters)  # normalisation pour stabiliser l'entraînement

        # Deuxième couche convolutive : raffine les représentations
        self.conv2 = nn.Conv1d(
            in_channels=cnn_filters,
            out_channels=cnn_filters,
            kernel_size=cnn_kernel_size,
            padding=cnn_kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm1d(cnn_filters)

        self.relu = nn.ReLU()
        # MaxPool divise la longueur de séquence par 2 (sous-échantillonnage)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # LSTM qui prend en entrée les features extraites par le CNN
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
        # Le CNN attend (batch, canaux, longueur), on permute depuis (batch, longueur, features)
        x = x.permute(0, 2, 1)

        # Première extraction de features locales
        x = self.relu(self.bn1(self.conv1(x)))

        # Deuxième extraction de features locales
        x = self.relu(self.bn2(self.conv2(x)))

        # Réduction de la longueur de séquence par pooling
        x = self.pool(x)

        # Revient au format (batch, longueur, features) pour le LSTM
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]  # état au dernier pas de temps

        out = self.dropout(last_output)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)  # logit brut
        return out
