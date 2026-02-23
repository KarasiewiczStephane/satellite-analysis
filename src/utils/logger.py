"""Structured logging setup for the satellite analysis platform."""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: str | None = None,
) -> logging.Logger:
    """Create and configure a logger with console and optional file output.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to a log file. Parent directories are created
            automatically.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
