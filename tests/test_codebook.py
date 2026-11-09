"""Тесты загрузки кодбука из рабочей книги."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from qa.db.connection import get_connection
from qa.db.migrations import migrate
from qa.pipeline.load_codebook import load_codebook
from qa.pipeline.workbook import WorkbookError
from qa.settings import Settings
from tests.fixtures.workbook_factory import build_workbook

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations"
_REAL_CODEBOOK_ENV = "QA_REAL_CODEBOOK_PATH"


@pytest.fixture()
def settings() -> Settings:
    return Settings.load(_PROJECT_ROOT / "config.yaml")


@pytest.fixture()
def db(tmp_path: Path) -> sqlite3.Connection:
    """Подключение к временной БД с применёнными миграциями."""
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    migrate(conn, db_path, _MIGRATIONS_DIR, tmp_path / "backups")
    yield conn
    conn.close()


class TestLoadCodebook:
    """Проверка загрузки обязательных листов и тем."""

    def test_synthetic(self, db: sqlite3.Connection, tmp_path: Path, settings: Settings) -> None:
        path = build_workbook(tmp_path / "book.xlsx")
        result = load_codebook(db, path, settings.workbook)
        assert result.created is True
        assert result.counts["codes"] == 10
        assert result.counts["questions"] == 8
        assert result.counts["glossary"] == 2
        assert result.counts["attributes"] == 3

    def test_question_code_links(self, db: sqlite3.Connection, tmp_path: Path, settings: Settings) -> None:
        path = build_workbook(tmp_path / "book.xlsx")
        load_codebook(db, path, settings.workbook)
        pairs = {
            (question, code)
            for question, code in db.execute(
                "SELECT q.identifier, c.name FROM question_code qc"
                " JOIN question q ON qc.question_id = q.id JOIN code c ON qc.code_id = c.id"
            )
        }
        # Коды вопроса 8 пришли и из «коды», и из колонки «вопросы» листа «Коды».
        assert ("8", "R1") in pairs
        assert ("8", "R2") in pairs
        assert ("9", "V1") in pairs
        # Диапазон E0–E1 развёрнут, но E0 в книге нет — остаётся только E1.
        assert ("9Т", "E1") in pairs
        # «все» у OTH — не идентификатор вопроса, связи нет.
        assert not any(code == "OTH" and question == "все" for question, code in pairs)

    def test_theme_tree_from_sheet(self, db: sqlite3.Connection, tmp_path: Path, settings: Settings) -> None:
        path = build_workbook(tmp_path / "book.xlsx")
        result = load_codebook(db, path, settings.workbook)
        tree = dict((node, themes) for node, themes in result.tree)
        assert set(tree) == {"Риски", "Ответственность", "Предупреждения", "Индуктивные"}
        risk_themes = dict(tree["Риски"])
        assert set(risk_codes := set(risk_themes["Риски"])) == {"R1", "R2"}
        assert "V1" in risk_themes["Проверка"]
        assert risk_codes

    def test_tree_from_families_when_themes_absent(
        self, db: sqlite3.Connection, tmp_path: Path, settings: Settings
    ) -> None:
        path = build_workbook(tmp_path / "book.xlsx", with_themes=False)
        result = load_codebook(db, path, settings.workbook)
        tree = dict((node, dict(themes)) for node, themes in result.tree)
        assert set(tree) == {"Риски", "Ответственность", "Предупреждения", "Индуктивные"}
        assert set(tree["Риски"]["Риски"]) == {"R1", "R2"}
        assert "E1" in tree["Риски"]["Ошибки исполнения"]
        assert "C1" in tree["Ответственность"]["Контроль"]

    def test_tree_from_families_when_themes_empty(
        self, db: sqlite3.Connection, tmp_path: Path, settings: Settings
    ) -> None:
        path = build_workbook(tmp_path / "book.xlsx", with_themes=True, themes=[])
        result = load_codebook(db, path, settings.workbook)
        nodes = {node for node, _themes in result.tree}
        assert "Риски" in nodes
        assert "Ответственность" in nodes

    def test_no_duplicates_on_reload(self, db: sqlite3.Connection, tmp_path: Path, settings: Settings) -> None:
        path = build_workbook(tmp_path / "book.xlsx")
        load_codebook(db, path, settings.workbook)
        second = load_codebook(db, path, settings.workbook)
        assert second.created is False
        assert db.execute("SELECT COUNT(*) FROM code").fetchone()[0] == 10
        assert db.execute("SELECT COUNT(*) FROM question").fetchone()[0] == 8
        assert db.execute("SELECT COUNT(*) FROM theme").fetchone()[0] == len(second.tree) + sum(
            len(themes) for _node, themes in second.tree
        )

    def test_duplicate_code_is_error(self, db: sqlite3.Connection, tmp_path: Path, settings: Settings) -> None:
        from tests.fixtures.workbook_factory import CODES, QUESTIONS

        codes = [CODES[0], CODES[0], *CODES[2:]]
        path = build_workbook(tmp_path / "book.xlsx", codes=codes, questions=QUESTIONS)
        with pytest.raises(WorkbookError, match="повторяющиеся коды"):
            load_codebook(db, path, settings.workbook)

    def test_missing_family_is_error(self, db: sqlite3.Connection, tmp_path: Path, settings: Settings) -> None:
        codes = [["R1", "содержимое", "", "Ошибка", "—", "—", "—", "—", "8"]]
        path = build_workbook(tmp_path / "book.xlsx", codes=codes)
        with pytest.raises(WorkbookError, match="не указана семья"):
            load_codebook(db, path, settings.workbook)

    def test_unknown_code_in_questions_is_error(
        self, db: sqlite3.Connection, tmp_path: Path, settings: Settings
    ) -> None:
        from tests.fixtures.workbook_factory import QUESTIONS

        questions = [[*QUESTIONS[0][:5], "R1, Z9", "—"], *QUESTIONS[1:]]
        path = build_workbook(tmp_path / "book.xlsx", questions=questions)
        with pytest.raises(WorkbookError, match="нет таких кодов"):
            load_codebook(db, path, settings.workbook)

    def test_unknown_question_in_codes_is_error(
        self, db: sqlite3.Connection, tmp_path: Path, settings: Settings
    ) -> None:
        from tests.fixtures.workbook_factory import CODES, QUESTIONS

        codes = [["R1", "содержание", "Риски", "Ошибка", "—", "—", "—", "—", "8, 99"], *CODES[1:]]
        path = build_workbook(tmp_path / "book.xlsx", codes=codes, questions=QUESTIONS)
        with pytest.raises(WorkbookError, match="нет таких вопросов"):
            load_codebook(db, path, settings.workbook)

    def test_unknown_code_in_themes_is_error(self, db: sqlite3.Connection, tmp_path: Path, settings: Settings) -> None:
        path = build_workbook(tmp_path / "book.xlsx", themes=[["Риски", "Риски", "R1; Z9", "черновик", "—"]])
        with pytest.raises(WorkbookError, match="коды не найдены"):
            load_codebook(db, path, settings.workbook)

    def test_glossary_terms(self, db: sqlite3.Connection, tmp_path: Path, settings: Settings) -> None:
        path = build_workbook(tmp_path / "book.xlsx")
        load_codebook(db, path, settings.workbook)
        terms = {row[0] for row in db.execute("SELECT term FROM glossary_term")}
        assert terms == {"MCP", "токен"}

    def test_file_not_found(self, db: sqlite3.Connection, settings: Settings) -> None:
        with pytest.raises(FileNotFoundError):
            load_codebook(db, Path("/nonexistent/file.xlsx"), settings.workbook)


@pytest.mark.real_codebook
class TestRealCodebook:
    """Тест на реальной книге (42 кода). Пропускается без env-переменной."""

    def test_42_codes_and_links(self, db: sqlite3.Connection, settings: Settings) -> None:
        real_path = os.environ.get(_REAL_CODEBOOK_ENV)
        if not real_path:
            pytest.skip(f"Переменная {_REAL_CODEBOOK_ENV} не задана")
        path = Path(real_path)
        if not path.exists():
            pytest.skip(f"Файл не найден: {path}")
        result = load_codebook(db, path, settings.workbook)
        assert result.counts["codes"] == 42, f"Ожидалось 42 кода, получено {result.counts['codes']}"
        assert result.counts["questions"] == 29
        assert result.counts["question_code"] > 0
        orphan_codes = db.execute(
            "SELECT c.name FROM code c LEFT JOIN question_code qc ON qc.code_id = c.id WHERE qc.id IS NULL"
        ).fetchall()
        # OTH — индуктивный код: в книге вместо вопросов стоит пометка «все»,
        # связей с вопросами у него нет по замыслу. Остальные коды обязаны иметь связь.
        assert [row[0] for row in orphan_codes] == ["OTH"]
        orphan_links = db.execute(
            "SELECT qc.id FROM question_code qc"
            " LEFT JOIN code c ON qc.code_id = c.id LEFT JOIN question q ON qc.question_id = q.id"
            " WHERE c.id IS NULL OR q.id IS NULL"
        ).fetchall()
        assert orphan_links == []
        nodes = {node for node, _themes in result.tree}
        assert {"Риски", "Ответственность", "Предупреждения", "Индуктивные"} <= nodes
