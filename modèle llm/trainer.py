"""
Module d'entraînement unifié pour tous les modèles PyTorch.

Fournit un Trainer générique qui gère :
    - La boucle d'entraînement avec mini-batches
    - L'early stopping pour éviter le surapprentissage
    - Le suivi des métriques par époque
    - La sauvegarde du meilleur modèle
    - Le logging des performances
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional, Tuple
import time


class EarlyStopping:
    """
    Early Stopping — arrête l'entraînement quand la validation ne s'améliore plus.
    
    C'est un mécanisme crucial de régularisation : on surveille la loss de validation,
    et si elle ne diminue pas pendant `patience` époques consécutives, on arrête
    pour éviter que le modèle mémorise le bruit des données d'entraînement.
    
    On sauvegarde toujours les poids du meilleur modèle (celui avec la plus basse
    loss de validation), pas le dernier.
    """
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.best_model_state = None
        self.should_stop = False

    def check(self, val_loss: float, model: nn.Module) -> bool:
        """
        Vérifie si on doit arrêter l'entraînement.
        
        Returns:
            True si l'entraînement doit s'arrêter
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Sauvegarde des meilleurs poids
            self.best_model_state = {
                k: v.clone() for k, v in model.state_dict().items()
            }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """
    Entraîneur unifié pour tous les modèles PyTorch du projet.

    Classification binaire : BCEWithLogitsLoss (combine sigmoid + BCE pour
    plus de stabilité numérique). Le modèle sort des logits bruts ; sigmoid
    est appliqué uniquement à l'inférence dans predict().
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 15,
        device: str = None,
    ):
        # Sélection automatique du device (GPU si disponible)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.epochs = epochs

        # BCEWithLogitsLoss pour la classification binaire
        self.criterion = nn.BCEWithLogitsLoss()

        # Adam : optimizer adaptatif qui ajuste le LR par paramètre
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Scheduler : divise le LR par 2 si la val_loss ne baisse plus pendant 5 époques
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=False
        )

        self.early_stopping = EarlyStopping(patience=patience)
        self.history = {"train_loss": [], "val_loss": []}

    def _create_dataloader(self, X: np.ndarray, y: np.ndarray, shuffle: bool = True) -> DataLoader:
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
        """
        Boucle d'entraînement complète.
        
        Pour chaque époque :
            1. Phase d'entraînement : forward + backward + update des poids
            2. Phase de validation : forward seulement (pas de gradient)
            3. Vérification de l'early stopping
            4. Mise à jour du learning rate scheduler
        
        Returns:
            Historique des losses {train_loss: [...], val_loss: [...]}
        """
        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
        val_loader = self._create_dataloader(X_val, y_val, shuffle=False)

        start_time = time.time()

        for epoch in range(self.epochs):
            # --- Phase d'entraînement ---
            self.model.train()
            train_losses = []

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                predictions = self.model(batch_X).squeeze()
                loss = self.criterion(predictions, batch_y)

                # Backward pass + mise à jour
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping pour éviter l'explosion des gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_losses.append(loss.item())

            # --- Phase de validation ---
            self.model.eval()
            val_losses = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    predictions = self.model(batch_X).squeeze()
                    loss = self.criterion(predictions, batch_y)
                    val_losses.append(loss.item())

            # Moyennes des losses
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)

            # Scheduler : ajuste le LR si la validation stagne
            self.scheduler.step(avg_val_loss)

            # Logging
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

            # Early stopping
            if self.early_stopping.check(avg_val_loss, self.model):
                if verbose:
                    print(f"  → Early stopping à l'époque {epoch+1}")
                break

        # Restaure les meilleurs poids
        if self.early_stopping.best_model_state is not None:
            self.model.load_state_dict(self.early_stopping.best_model_state)

        total_time = time.time() - start_time
        if verbose:
            print(f"  → Entraînement terminé en {total_time:.1f}s")
            print(f"  → Meilleure val loss: {self.early_stopping.best_loss:.6f}")

        self.history["training_time"] = total_time
        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retourne les probabilités de la classe 1 (après sigmoid sur les logits)."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            logits = self.model(X_tensor).squeeze()
            probs  = torch.sigmoid(logits).cpu().numpy()

        return probs

    def count_parameters(self) -> int:
        """Compte le nombre total de paramètres entraînables du modèle."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
