"""Configuration management with singleton pattern and dot-notation access."""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class Config:
    """Singleton configuration loader for YAML-based project settings.

    Provides dot-notation key access (e.g. ``config.get("model.learning_rate")``)
    and lazy loading from a YAML file.
    """

    _instance: "Config | None" = None
    _config: dict[str, Any] | None = None

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
        """Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            Parsed configuration dictionary.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path) as f:
            self._config = yaml.safe_load(f)
        logger.info("Loaded configuration from %s", config_path)
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value using dot-separated key notation.

        Args:
            key: Dot-separated path into the config (e.g. ``"model.learning_rate"``).
            default: Fallback value when the key is missing.

        Returns:
            The configuration value or *default*.
        """
        if self._config is None:
            self.load()
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def reset(self) -> None:
        """Reset loaded configuration (useful for testing)."""
        self._config = None


config = Config()
