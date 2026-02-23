"""Tests for model evaluation and metrics."""

import matplotlib
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.classifier import LandUseClassifier
from src.models.evaluator import (
    EvaluationResults,
    ModelEvaluator,
    compare_models,
)

matplotlib.use("Agg")

CLASS_NAMES = ["Forest", "Residential", "River"]


@pytest.fixture()
def dummy_test_loader():
    x = torch.randn(20, 3, 32, 32)
    y = torch.tensor([0] * 7 + [1] * 7 + [2] * 6)
    return DataLoader(TensorDataset(x, y), batch_size=10)


@pytest.fixture()
def evaluator(dummy_test_loader):
    model = LandUseClassifier(num_classes=3, architecture="resnet50", pretrained=False)
    return ModelEvaluator(model, dummy_test_loader, CLASS_NAMES, device="cpu")


@pytest.fixture()
def eval_results(evaluator):
    return evaluator.evaluate()


class TestModelEvaluator:
    def test_evaluate_returns_results(self, eval_results) -> None:
        assert isinstance(eval_results, EvaluationResults)

    def test_accuracy_range(self, eval_results) -> None:
        assert 0 <= eval_results.accuracy <= 1

    def test_f1_macro_range(self, eval_results) -> None:
        assert 0 <= eval_results.f1_macro <= 1

    def test_confusion_matrix_shape(self, eval_results) -> None:
        cm = eval_results.confusion_matrix
        assert cm.shape[0] == cm.shape[1]

    def test_per_class_f1_keys(self, eval_results) -> None:
        assert set(eval_results.f1_per_class.keys()) == set(CLASS_NAMES)

    def test_predictions_length(self, eval_results) -> None:
        assert len(eval_results.predictions) == 20
        assert len(eval_results.true_labels) == 20

    def test_probabilities_shape(self, eval_results) -> None:
        assert eval_results.probabilities.shape == (20, 3)

    def test_classification_report(self, eval_results) -> None:
        report = ModelEvaluator.get_classification_report(eval_results)
        assert isinstance(report, str)
        assert "Forest" in report


class TestPlots:
    def test_confusion_matrix_plot(self, eval_results) -> None:
        fig = ModelEvaluator.plot_confusion_matrix(eval_results)
        assert fig is not None

    def test_confusion_matrix_save(self, eval_results, tmp_path) -> None:
        path = tmp_path / "cm.png"
        ModelEvaluator.plot_confusion_matrix(eval_results, save_path=path)
        assert path.exists()

    def test_per_class_metrics_plot(self, eval_results) -> None:
        fig = ModelEvaluator.plot_per_class_metrics(eval_results)
        assert fig is not None


class TestCompareModels:
    def test_compare_two_models(self, eval_results) -> None:
        fig = compare_models({"ModelA": eval_results, "ModelB": eval_results})
        assert fig is not None

    def test_compare_save(self, eval_results, tmp_path) -> None:
        path = tmp_path / "compare.png"
        compare_models({"A": eval_results, "B": eval_results}, save_path=path)
        assert path.exists()
