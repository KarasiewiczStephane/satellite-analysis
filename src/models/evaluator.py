"""Model evaluation with per-class metrics, confusion matrix, and comparison."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.utils.logger import setup_logger

matplotlib.use("Agg")

logger = setup_logger(__name__)


@dataclass
class EvaluationResults:
    """Container for classification evaluation metrics.

    Attributes:
        accuracy: Overall accuracy.
        f1_macro: Macro-averaged F1 score.
        f1_weighted: Weighted-average F1 score.
        f1_per_class: Per-class F1 scores keyed by class name.
        precision_macro: Macro-averaged precision.
        recall_macro: Macro-averaged recall.
        confusion_matrix: Confusion matrix array of shape ``(C, C)``.
        predictions: Flat array of predicted labels.
        true_labels: Flat array of ground-truth labels.
        probabilities: Softmax probability matrix ``(N, C)``.
        class_names: Ordered list of class names.
    """

    accuracy: float
    f1_macro: float
    f1_weighted: float
    f1_per_class: dict[str, float]
    precision_macro: float
    recall_macro: float
    confusion_matrix: np.ndarray
    predictions: np.ndarray
    true_labels: np.ndarray
    probabilities: np.ndarray
    class_names: list[str]


class ModelEvaluator:
    """Evaluate a trained classification model on a test set.

    Args:
        model: Trained :class:`nn.Module`.
        test_loader: DataLoader yielding ``(images, labels)`` batches.
        class_names: Ordered class name list.
        device: Compute device (``"auto"`` selects CUDA when available).
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        class_names: list[str],
        device: str = "auto",
    ) -> None:
        self.model = model
        self.test_loader = test_loader
        self.class_names = class_names

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self) -> EvaluationResults:
        """Run full evaluation on the test set.

        Returns:
            Populated :class:`EvaluationResults`.
        """
        all_preds: list[int] = []
        all_labels: list[int] = []
        all_probs: list[list[float]] = []

        for images, labels in self.test_loader:
            images = images.to(self.device)
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_per_class_arr = f1_score(y_true, y_pred, average=None, zero_division=0)
        precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        f1_dict = {
            name: float(score)
            for name, score in zip(self.class_names, f1_per_class_arr, strict=False)
        }

        logger.info("Test Accuracy: %.4f", accuracy)
        logger.info("F1 Macro: %.4f", f1_macro)

        return EvaluationResults(
            accuracy=accuracy,
            f1_macro=f1_macro,
            f1_weighted=f1_weighted,
            f1_per_class=f1_dict,
            precision_macro=precision_macro,
            recall_macro=recall_macro,
            confusion_matrix=cm,
            predictions=y_pred,
            true_labels=y_true,
            probabilities=y_prob,
            class_names=self.class_names,
        )

    @staticmethod
    def plot_confusion_matrix(
        results: EvaluationResults,
        save_path: str | Path | None = None,
        figsize: tuple[int, int] = (12, 10),
    ) -> plt.Figure:
        """Plot a confusion matrix heatmap.

        Args:
            results: Evaluation results containing the confusion matrix.
            save_path: Optional file path to save the figure.
            figsize: Figure dimensions.

        Returns:
            Matplotlib :class:`Figure`.
        """
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            results.confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=results.class_names,
            yticklabels=results.class_names,
            ax=ax,
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix (Accuracy: {results.accuracy:.4f})")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Confusion matrix saved to %s", save_path)

        return fig

    @staticmethod
    def plot_per_class_metrics(
        results: EvaluationResults,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Plot horizontal bar chart of per-class F1 scores.

        Args:
            results: Evaluation results with per-class metrics.
            save_path: Optional file path to save the figure.

        Returns:
            Matplotlib :class:`Figure`.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        classes = list(results.f1_per_class.keys())
        scores = list(results.f1_per_class.values())

        bars = ax.barh(classes, scores, color="steelblue")
        ax.set_xlabel("F1 Score")
        ax.set_title("Per-Class F1 Scores")
        ax.set_xlim(0, 1)

        for bar, score in zip(bars, scores, strict=False):
            ax.text(
                score + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}",
                va="center",
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    @staticmethod
    def get_classification_report(results: EvaluationResults) -> str:
        """Generate a sklearn classification report string.

        Args:
            results: Evaluation results.

        Returns:
            Formatted classification report.
        """
        return classification_report(
            results.true_labels,
            results.predictions,
            target_names=results.class_names,
            zero_division=0,
        )


def compare_models(
    results_dict: dict[str, EvaluationResults],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Compare multiple models side-by-side on accuracy and F1.

    Args:
        results_dict: Mapping of model name to evaluation results.
        save_path: Optional path to save the comparison figure.

    Returns:
        Matplotlib :class:`Figure`.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    model_names = list(results_dict.keys())
    accuracies = [r.accuracy for r in results_dict.values()]
    f1_scores = [r.f1_macro for r in results_dict.values()]

    colors = ["steelblue", "coral", "seagreen", "mediumpurple"][: len(model_names)]

    axes[0].bar(model_names, accuracies, color=colors)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Model Accuracy Comparison")
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 0.02, f"{v:.4f}", ha="center")

    axes[1].bar(model_names, f1_scores, color=colors)
    axes[1].set_ylabel("F1 Score (Macro)")
    axes[1].set_title("Model F1 Score Comparison")
    axes[1].set_ylim(0, 1)
    for i, v in enumerate(f1_scores):
        axes[1].text(i, v + 0.02, f"{v:.4f}", ha="center")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
