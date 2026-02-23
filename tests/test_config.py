"""Tests for configuration management."""

import pytest

from src.utils.config import Config


@pytest.fixture()
def config_file(tmp_path):
    """Create a temporary config YAML file."""
    content = """\
data:
  num_classes: 10
  classes:
    - Forest
    - Residential
    - River
model:
  learning_rate: 0.001
  architecture: resnet50
export:
  crs: "EPSG:4326"
logging:
  level: INFO
"""
    path = tmp_path / "config.yaml"
    path.write_text(content)
    return path


@pytest.fixture()
def cfg(config_file):
    """Return a fresh Config instance loaded from the temp file."""
    instance = Config()
    instance.reset()
    instance.load(config_file)
    return instance


class TestConfigLoad:
    def test_load_returns_dict(self, cfg: Config) -> None:
        assert isinstance(cfg._config, dict)

    def test_load_missing_file_raises(self) -> None:
        instance = Config()
        instance.reset()
        with pytest.raises(FileNotFoundError):
            instance.load("/nonexistent/path.yaml")

    def test_top_level_keys(self, cfg: Config) -> None:
        for key in ("data", "model", "export", "logging"):
            assert key in cfg._config


class TestConfigGet:
    def test_nested_key(self, cfg: Config) -> None:
        assert cfg.get("model.learning_rate") == 0.001

    def test_nested_list(self, cfg: Config) -> None:
        classes = cfg.get("data.classes")
        assert isinstance(classes, list)
        assert "Forest" in classes

    def test_missing_key_returns_default(self, cfg: Config) -> None:
        assert cfg.get("nonexistent.key", "fallback") == "fallback"

    def test_deep_nested_key(self, cfg: Config) -> None:
        assert cfg.get("export.crs") == "EPSG:4326"


class TestConfigReset:
    def test_reset_clears_config(self, cfg: Config) -> None:
        cfg.reset()
        assert cfg._config is None
