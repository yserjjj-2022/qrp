"""Тесты каркаса репозитория и CLI."""

from __future__ import annotations

import os
import sqlite3
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
        assert "workbook" in captured.out


class TestSettings:
    """Проверка загрузки и валидации настроек."""

    def test_default_provider(self, settings: Settings) -> None:
        assert settings.llm.provider == "stub"

    def test_db_path(self, settings: Settings) -> None:
        assert settings.db_path.endswith("data/db/interviews.db")

    def test_paths_defaults(self, settings: Settings) -> None:
        assert settings.paths.workbook == "data/inputs/Rabochaya_kniga_proekta_v4.xlsx"
        assert settings.paths.transcripts == "data/inputs/transcripts"
        assert settings.paths.golden == "data/golden"

    def test_workbook_sheet_classes(self, settings: Settings) -> None:
        workbook = settings.workbook
        assert workbook.sheet_class("Вопросы_и_коды") == "required"
        assert workbook.sheet_class("Лексикон") == "optional"
        assert workbook.sheet_class("Факты_Финам") == "optional"
        assert workbook.sheet_class("Справка") == "ignored"
        assert settings.workbook.sheet_class("_Служебный") == "ignored"
        assert workbook.sheet_class("Что_то_новое") == "unknown"

    def test_workbook_column_types(self, settings: Settings) -> None:
        spec = settings.workbook.sheets["Коды"]
        assert spec.columns["code"].type == "str"
        assert spec.columns["questions"].type == "list"
        assert spec.columns["questions"].list_kind == "question"
        assert settings.workbook.sheets["Вопросы_и_коды"].columns["codes"].list_kind == "code"

    def test_llm_model_none(self, settings: Settings) -> None:
        assert settings.llm.model is None

    def test_llm_endpoint_none(self, settings: Settings) -> None:
        assert settings.llm.endpoint is None

    def test_data_dir(self, settings: Settings) -> None:
        assert settings.data_dir == "data"

    def test_log_level(self, settings: Settings) -> None:
        assert settings.log_level == "INFO"

    def test_llm_max_calls(self, settings: Settings) -> None:
        assert settings.llm.max_calls_per_run == 200

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


class TestDomainHint:
    """Строка предметной области задаётся вне публичного конфига."""

    def test_public_config_has_no_domain_hint(self, config_path: Path) -> None:
        """В репозитории строка пустая: названия брокеров не публикуются.

        Проверяется сам файл, а не результат слияния: локальный
        config.local.yaml рядом намеренно содержит рабочую строку.
        """
        import yaml

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert payload["domain_hint"] == ""

    def test_env_overrides(self) -> None:
        with patch.dict(os.environ, {"QA_DOMAIN_HINT": "интервью про роботов"}, clear=False):
            assert Settings.load().domain_hint == "интервью про роботов"

    def test_empty_env_keeps_config(self, settings: Settings) -> None:
        """Пустая переменная не затирает значение из конфига."""
        with patch.dict(os.environ, {"QA_DOMAIN_HINT": "   "}, clear=False):
            loaded = Settings.load(_project_root() / "config.yaml")
        assert loaded.domain_hint == settings.domain_hint

    def test_local_config_overrides(self, tmp_path: Path) -> None:
        """config.local.yaml перекрывает config.yaml и не коммитится."""
        config = tmp_path / "config.yaml"
        config.write_text('domain_hint: ""\nlog_level: INFO\n', encoding="utf-8")
        (tmp_path / "config.local.yaml").write_text(
            'domain_hint: "локальная строка"\n',
            encoding="utf-8",
        )
        assert Settings.load(config).domain_hint == "локальная строка"

    def test_env_beats_local_config(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yaml"
        config.write_text('domain_hint: ""\n', encoding="utf-8")
        (tmp_path / "config.local.yaml").write_text('domain_hint: "файл"\n', encoding="utf-8")
        with patch.dict(os.environ, {"QA_DOMAIN_HINT": "окружение"}, clear=False):
            assert Settings.load(config).domain_hint == "окружение"

    def test_missing_local_config_is_fine(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yaml"
        config.write_text("log_level: INFO\n", encoding="utf-8")
        assert Settings.load(config).domain_hint == ""


class TestDbPathOverride:
    """Путь к БД задаётся опцией --db или переменной QA_DB_PATH."""

    def test_argument_overrides_config(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yaml"
        config.write_text("paths:\n  db: data/db/interviews.db\n", encoding="utf-8")
        loaded = Settings.load(config, db_path=str(tmp_path / "copy.db"))
        assert loaded.db_path == str(tmp_path / "copy.db")

    def test_env_overrides_config(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yaml"
        config.write_text("paths:\n  db: data/db/interviews.db\n", encoding="utf-8")
        with patch.dict(os.environ, {"QA_DB_PATH": str(tmp_path / "copy.db")}, clear=False):
            assert Settings.load(config).db_path == str(tmp_path / "copy.db")

    def test_argument_beats_env(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yaml"
        config.write_text("paths:\n  db: data/db/interviews.db\n", encoding="utf-8")
        with patch.dict(os.environ, {"QA_DB_PATH": str(tmp_path / "env.db")}, clear=False):
            loaded = Settings.load(config, db_path=str(tmp_path / "cli.db"))
        assert loaded.db_path == str(tmp_path / "cli.db")

    def test_empty_env_keeps_config(self, config_path: Path) -> None:
        """Пустая переменная не затирает путь из конфига."""
        with patch.dict(os.environ, {"QA_DB_PATH": "   "}, clear=False):
            loaded = Settings.load(config_path)
        assert loaded.db_path == Settings.load(config_path).paths.db

    def test_db_override_creates_copy_not_working_db(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Команда с --db создаёт копию и не создаёт рабочую БД.

        `review-terms` выбран как команда без модели: на пустой базе очередь
        пуста, ввода с stdin нет. Рабочая база в тесте — путь, которого нет:
        если бы опция игнорировалась, файл появился бы.
        """
        work_db = tmp_path / "work.db"
        copy_db = tmp_path / "copy.db"
        (tmp_path / "config.yaml").write_text(
            f"paths:\n  data: {tmp_path / 'data'}\n  db: {work_db}\n"
            f"  backups: {tmp_path / 'backups'}\n  cache: {tmp_path / 'cache'}\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        with patch.object(sys, "argv", ["qa-pipeline", "review-terms", "--db", str(copy_db)]):
            assert main() == 0

        assert copy_db.exists()
        assert not work_db.exists()
        check = sqlite3.connect(copy_db)
        try:
            assert check.execute("SELECT COUNT(*) FROM applied_migrations").fetchone()[0] > 0
            assert check.execute("SELECT COUNT(*) FROM new_term").fetchone()[0] == 0
        finally:
            check.close()

    def test_env_path_is_used_by_command(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """QA_DB_PATH без --db направляет команду в указанную базу."""
        work_db = tmp_path / "work.db"
        copy_db = tmp_path / "copy.db"
        (tmp_path / "config.yaml").write_text(
            f"paths:\n  data: {tmp_path / 'data'}\n  db: {work_db}\n"
            f"  backups: {tmp_path / 'backups'}\n  cache: {tmp_path / 'cache'}\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("QA_DB_PATH", str(copy_db))
        with patch.object(sys, "argv", ["qa-pipeline", "review-terms"]):
            assert main() == 0

        assert copy_db.exists()
        assert not work_db.exists()


class TestLogging:
    """Проверка настройки логирования."""

    def test_setup_logging_returns_run_id(self) -> None:
        run_id = setup_logging("INFO")
        assert isinstance(run_id, str)
        assert len(run_id) == 8

    def test_setup_logging_custom_run_id(self) -> None:
        run_id = setup_logging("DEBUG", run_id="test1234")
        assert run_id == "test1234"
