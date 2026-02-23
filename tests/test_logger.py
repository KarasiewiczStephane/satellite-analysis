"""Tests for the logging utility."""

import logging

from src.utils.logger import setup_logger


class TestSetupLogger:
    def test_returns_logger(self) -> None:
        logger = setup_logger("test_basic")
        assert isinstance(logger, logging.Logger)

    def test_logger_level(self) -> None:
        logger = setup_logger("test_level", level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_console_handler_attached(self) -> None:
        logger = setup_logger("test_handler")
        handler_types = [type(h) for h in logger.handlers]
        assert logging.StreamHandler in handler_types

    def test_file_handler_created(self, tmp_path) -> None:
        log_file = tmp_path / "logs" / "test.log"
        logger = setup_logger("test_file", log_file=str(log_file))
        logger.info("hello")
        assert log_file.exists()

    def test_no_duplicate_handlers(self) -> None:
        name = "test_no_dup"
        logger1 = setup_logger(name)
        count = len(logger1.handlers)
        logger2 = setup_logger(name)
        assert len(logger2.handlers) == count


class TestMainEntryPoint:
    def test_parse_args_default(self) -> None:
        from src.main import parse_args

        args = parse_args([])
        assert args.config == "configs/config.yaml"
        assert args.command is None

    def test_parse_args_train(self) -> None:
        from src.main import parse_args

        args = parse_args(["train", "--epochs", "10"])
        assert args.command == "train"
        assert args.epochs == 10

    def test_main_runs(self, tmp_path) -> None:
        from src.main import main

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("data:\n  num_classes: 10\n")
        main(["--config", str(cfg_path)])
