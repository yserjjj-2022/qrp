"""Тесты каркаса репозитория и CLI."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from qa.cli import main
from qa.log import setup_logging
from qa.settings import Settings


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture()
def config_path() -> Path:
    return _project_root() / "config.yaml"


@pytest.fixture()
def settings(config_path: Path) -> Settings:
    return Settings.load(config_path)


class TestCLI:
    """Проверка CLI-каркаса."""

    def test_help(self, capsys: pytest.CaptureFixture[Any]) -> None:
        """qa-pipeline --help завершается без ошибки и показывает usage."""
        with patch.object(sys, "argv", ["qa-pipeline", "--help"]), pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "qa-pipeline" in captured.out or "Анализатор" in captured.out

    def test_load_codebook_help(self, capsys: pytest.CaptureFixture[Any]) -> None:
        """qa-pipeline load-codebook --help показывает аргументы."""
        with (
            patch.object(sys, "argv", ["qa-pipeline", "load-codebook", "--help"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "file" in captured.out


class TestSettings:
    """Проверка загрузки и валидации настроек."""

    def test_default_provider(self, settings: Settings) -> None:
        assert settings.llm.provider == "stub"

    def test_db_path(self, settings: Settings) -> None:
        assert settings.db_path.endswith("data/interviews.db")

    def test_llm_model_none(self, settings: Settings) -> None:
        assert settings.llm.model is None

    def test_llm_endpoint_none(self, settings: Settings) -> None:
        assert settings.llm.endpoint is None

    def test_data_dir(self, settings: Settings) -> None:
        assert settings.data_dir == "data"

    def test_log_level(self, settings: Settings) -> None:
        assert settings.log_level == "INFO"

    def test_llm_max_calls(self, settings: Settings) -> None:
        assert settings.llm.max_calls_per_run == 100

    def test_api_key_not_required_when_stub(self) -> None:
        """При provider='stub' LLM_API_KEY не обязателен."""
        with patch.dict(os.environ, {}, clear=True):
            s = Settings.load()
        assert s.llm.provider == "stub"

    def test_api_key_required_when_not_stub(self) -> None:
        """При provider != 'stub' LLM_API_KEY обязателен."""
        from pydantic import ValidationError

        with patch.dict(os.environ, {}, clear=True), pytest.raises(ValidationError):
            Settings(db_path="data/test.db", llm={"provider": "openai"})

    def test_invalid_log_level(self) -> None:
        """Недопустимый log_level вызывает ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings(db_path="data/test.db", log_level="INVALID")


class TestLogging:
    """Проверка настройки логирования."""

    def test_setup_logging_returns_run_id(self) -> None:
        run_id = setup_logging("INFO")
        assert isinstance(run_id, str)
        assert len(run_id) == 8

    def test_setup_logging_custom_run_id(self) -> None:
        run_id = setup_logging("DEBUG", run_id="test1234")
        assert run_id == "test1234"
