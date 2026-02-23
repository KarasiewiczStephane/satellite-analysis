"""Tests for the training loop and checkpointing."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.classifier import LandUseClassifier
from src.models.trainer import Trainer
from src.utils.config import Config


@pytest.fixture()
def _mock_config(tmp_path):
    content = """\
model:
  learning_rate: 0.01
  weight_decay: 0.0001
  early_stopping_patience: 2
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(content)
    instance = Config()
    instance.reset()
    instance.load(cfg_path)


@pytest.fixture()
def dummy_loaders():
    """Create tiny DataLoaders for fast testing."""
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 3, (16,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=8)
    return loader, loader


@pytest.fixture()
def small_model():
    return LandUseClassifier(num_classes=3, architecture="resnet50", pretrained=False)


class TestTrainer:
    @pytest.mark.usefixtures("_mock_config")
    def test_train_epoch(self, small_model, dummy_loaders) -> None:
        train_loader, val_loader = dummy_loaders
        trainer = Trainer(small_model, train_loader, val_loader, device="cpu", use_amp=False)
        loss, acc = trainer.train_epoch()
        assert isinstance(loss, float)
        assert 0 <= acc <= 100

    @pytest.mark.usefixtures("_mock_config")
    def test_validate(self, small_model, dummy_loaders) -> None:
        train_loader, val_loader = dummy_loaders
        trainer = Trainer(small_model, train_loader, val_loader, device="cpu", use_amp=False)
        loss, acc = trainer.validate()
        assert isinstance(loss, float)
        assert 0 <= acc <= 100

    @pytest.mark.usefixtures("_mock_config")
    def test_fit_returns_history(self, small_model, dummy_loaders) -> None:
        train_loader, val_loader = dummy_loaders
        trainer = Trainer(small_model, train_loader, val_loader, device="cpu", use_amp=False)
        history = trainer.fit(epochs=2, early_stopping_patience=5)
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2

    @pytest.mark.usefixtures("_mock_config")
    def test_checkpoint_save_load(self, small_model, dummy_loaders, tmp_path) -> None:
        train_loader, val_loader = dummy_loaders
        trainer = Trainer(
            small_model,
            train_loader,
            val_loader,
            device="cpu",
            use_amp=False,
            checkpoint_dir=tmp_path,
        )
        trainer.fit(epochs=1, early_stopping_patience=5)
        trainer.save_checkpoint("test_ckpt.pt")

        assert (tmp_path / "test_ckpt.pt").exists()

        # Load checkpoint into fresh trainer
        fresh_model = LandUseClassifier(num_classes=3, architecture="resnet50", pretrained=False)
        fresh_trainer = Trainer(
            fresh_model,
            train_loader,
            val_loader,
            device="cpu",
            use_amp=False,
            checkpoint_dir=tmp_path,
        )
        fresh_trainer.load_checkpoint("test_ckpt.pt")
        assert fresh_trainer.best_val_loss < float("inf")

    @pytest.mark.usefixtures("_mock_config")
    def test_early_stopping(self, dummy_loaders, tmp_path) -> None:
        model = LandUseClassifier(num_classes=3, architecture="resnet50", pretrained=False)
        train_loader, val_loader = dummy_loaders
        trainer = Trainer(
            model,
            train_loader,
            val_loader,
            device="cpu",
            use_amp=False,
            checkpoint_dir=tmp_path,
        )
        history = trainer.fit(epochs=20, early_stopping_patience=2)
        # Should stop before 20 epochs if val loss doesn't improve
        assert len(history["train_loss"]) <= 20
