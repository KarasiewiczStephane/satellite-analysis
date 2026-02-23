"""Training loop with mixed-precision (AMP), early stopping, and checkpointing."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class Trainer:
    """Manages the training lifecycle for a classification model.

    Args:
        model: The neural network to train.
        train_loader: DataLoader for training samples.
        val_loader: DataLoader for validation samples.
        criterion: Loss function. Defaults to :class:`nn.CrossEntropyLoss`.
        optimizer: Optimizer instance. Defaults to AdamW with config values.
        scheduler: Optional learning-rate scheduler.
        device: Target device (``"auto"`` selects CUDA when available).
        use_amp: Enable automatic mixed-precision training.
        checkpoint_dir: Directory for saving model checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        device: str = "auto",
        use_amp: bool = True,
        checkpoint_dir: str | Path = "checkpoints",
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=config.get("model.learning_rate", 0.001),
            weight_decay=config.get("model.weight_decay", 0.0001),
        )
        self.scheduler = scheduler

        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def train_epoch(self) -> tuple[float, float]:
        """Run one training epoch.

        Returns:
            Tuple of ``(average_loss, accuracy_percentage)``.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({"loss": loss.item(), "acc": 100.0 * correct / total})

        return total_loss / total, 100.0 * correct / total

    @torch.no_grad()
    def validate(self) -> tuple[float, float]:
        """Evaluate the model on the validation set.

        Returns:
            Tuple of ``(average_loss, accuracy_percentage)``.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return total_loss / total, 100.0 * correct / total

    def fit(
        self,
        epochs: int,
        early_stopping_patience: int | None = None,
    ) -> dict[str, list[float]]:
        """Train the model for multiple epochs with early stopping.

        Args:
            epochs: Maximum number of training epochs.
            early_stopping_patience: Stop after this many epochs without
                validation loss improvement. Defaults to config value.

        Returns:
            Training history dictionary.
        """
        patience = early_stopping_patience or config.get("model.early_stopping_patience", 5)

        for epoch in range(epochs):
            logger.info("Epoch %d/%d", epoch + 1, epochs)

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            if self.scheduler:
                self.scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            logger.info(
                "Train Loss: %.4f, Train Acc: %.2f%% | Val Loss: %.4f, Val Acc: %.2f%%",
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint("best_model.pt")
                logger.info("Saved best model checkpoint")
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= patience:
                logger.info("Early stopping after %d epochs", epoch + 1)
                break

        return self.history

    def save_checkpoint(self, filename: str) -> None:
        """Persist model and optimizer state to disk.

        Args:
            filename: Checkpoint file name inside *checkpoint_dir*.
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
                "history": self.history,
            },
            self.checkpoint_dir / filename,
        )

    def load_checkpoint(self, filename: str) -> None:
        """Restore model and optimizer state from a checkpoint.

        Args:
            filename: Checkpoint file name inside *checkpoint_dir*.
        """
        checkpoint = torch.load(
            self.checkpoint_dir / filename,
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint["history"]
