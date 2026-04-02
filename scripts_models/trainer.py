import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional, Tuple
import time


class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.best_model_state = None
        self.should_stop = False

    def check(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = {
                k: v.clone() for k, v in model.state_dict().items()
            }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 15,
        device: str = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.epochs = epochs
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        self.early_stopping = EarlyStopping(patience=patience)
        self.history = {"train_loss": [], "val_loss": []}

    def _create_dataloader(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool = True
    ) -> DataLoader:
        """Crée un DataLoader PyTorch à partir de données numpy."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool = True,
    ) -> Dict:
        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
        val_loader = self._create_dataloader(X_val, y_val, shuffle=False)

        start_time = time.time()

        for epoch in range(self.epochs):
            self.model.train()
            train_losses = []

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                predictions = self.model(batch_X).squeeze()
                loss = self.criterion(predictions, batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_losses.append(loss.item())

            self.model.eval()
            val_losses = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    predictions = self.model(batch_X).squeeze()
                    loss = self.criterion(predictions, batch_y)
                    val_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)

            self.scheduler.step()

            if verbose and (epoch + 1) % 10 == 0:
                elapsed = time.time() - start_time
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch+1:3d}/{self.epochs} | "
                    f"Train Loss: {avg_train_loss:.6f} | "
                    f"Val Loss: {avg_val_loss:.6f} | "
                    f"LR: {current_lr:.1e} | "
                    f"Time: {elapsed:.1f}s"
                )

            if self.early_stopping.check(avg_val_loss, self.model):
                if verbose:
                    print(f"  → Early stopping à l'époch {epoch+1}")
                break

        if self.early_stopping.best_model_state is not None:
            self.model.load_state_dict(self.early_stopping.best_model_state)

        total_time = time.time() - start_time
        if verbose:
            print(f"  → Entraînement terminé en {total_time:.1f}s")
            print(f"  → Meilleure val loss: {self.early_stopping.best_loss:.6f}")

        self.history["training_time"] = total_time
        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            logits = self.model(X_tensor).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()

        return probs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
