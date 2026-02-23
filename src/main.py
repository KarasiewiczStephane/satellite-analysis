"""Entry point for the satellite image analysis platform.

Supports training, evaluation, and prediction workflows through CLI arguments.
"""

import argparse

from src.utils.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        description="Satellite Image Analysis Platform",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train sub-command
    train_parser = subparsers.add_parser("train", help="Train a classification model")
    train_parser.add_argument(
        "--architecture",
        type=str,
        default=None,
        help="Model architecture (resnet50 or efficientnet_b0)",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )

    # Evaluate sub-command
    subparsers.add_parser("evaluate", help="Evaluate a trained model")

    # Predict sub-command
    predict_parser = subparsers.add_parser("predict", help="Run prediction on an image")
    predict_parser.add_argument(
        "--image",
        type=str,
        required=False,
        help="Path to input satellite image",
    )

    # Download sub-command
    subparsers.add_parser("download", help="Download the EuroSAT dataset")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the CLI.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    args = parse_args(argv)
    config.load(args.config)
    logger.info("Configuration loaded from %s", args.config)

    if args.command == "train":
        logger.info("Starting training pipeline")
    elif args.command == "evaluate":
        logger.info("Starting evaluation pipeline")
    elif args.command == "predict":
        logger.info("Starting prediction pipeline")
    elif args.command == "download":
        logger.info("Starting dataset download")
    else:
        logger.info("No command specified. Use --help for available commands.")


if __name__ == "__main__":
    main()
