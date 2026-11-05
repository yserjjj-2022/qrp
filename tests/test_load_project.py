"""Тесты загрузки необязательных листов проекта и приоритета review-terms."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from qa.db.connection import get_connection
from qa.db.migrations import migrate
from qa.db.workbook_repo import LexiconRepository
from qa.pipeline.load_codebook import load_codebook
from qa.pipeline.load_project import load_project
from qa.settings import Settings
from tests.fixtures.workbook_factory import build_workbook

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations"


@pytest.fixture()
def settings() -> Settings:
    return Settings.load(_PROJECT_ROOT / "config.yaml")


@pytest.fixture()
def db(tmp_path: Path) -> sqlite3.Connection:
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    migrate(conn, db_path, _MIGRATIONS_DIR, tmp_path / "backups")
    yield conn
    conn.close()


def _loaded(db: sqlite3.Connection, path: Path, settings: Settings) -> None:
    load_codebook(db, path, settings.workbook)


class TestLoadProject:
    """Необязательные листы и умолчания."""

    def test_loads_lexicon_markers_interviews(self, db: sqlite3.Connection, tmp_path: Path, settings: Settings) -> None:
        path = build_workbook(tmp_path / "book.xlsx")
        _loaded(db, path, settings)
        result = load_project(db, path, settings.workbook)
        assert result.counts["lexicon"] == 2
        assert result.counts["uncertainty_markers"] == 2
        assert result.counts["interviews"] == 1
        terms = {row[0] for row in db.execute("SELECT canonical FROM lexicon")}
        assert terms == {"MCP", "Финам"}
        variants = db.execute("SELECT variants_json FROM lexicon WHERE canonical = 'MCP'").fetchone()[0]
        assert "эмсипи" in variants

    def test_defaults_when_optional_sheets_absent(
        self, db: sqlite3.Connection, tmp_path: Path, settings: Settings
    ) -> None:
        path = build_workbook(tmp_path / "book.xlsx", with_optional=False, with_themes=False)
        _loaded(db, path, settings)
        result = load_project(db, path, settings.workbook)
        assert "Маркеры_неуверенности" in result.missing_sheets
        markers = {row[0] for row in db.execute("SELECT expression FROM uncertainty_marker")}
        assert "не знаю" in markers
        assert db.execute("SELECT COUNT(*) FROM uncertainty_marker WHERE source = 'default'").fetchone()[0] > 0

    def test_review_decision_survives_reload(self, db: sqlite3.Connection, tmp_path: Path, settings: Settings) -> None:
        path = build_workbook(tmp_path / "book.xlsx")
        _loaded(db, path, settings)
        load_project(db, path, settings.workbook)

        repo = LexiconRepository(db)
        assert LexiconRepository(db).apply_review("MCP", "rejected") is True
        db.commit()

        second = load_project(db, path, settings.workbook)
        row = db.execute("SELECT status, source, variants_json FROM lexicon WHERE canonical = 'MCP'").fetchone()
        assert row[0] == "rejected"
        assert row[1] == "review"
        assert row[2] == '["эмсипи", "эсипи"]'
        assert any("MCP" in item and "review-terms" in item for item in second.discrepancies)
        assert repo.all_terms()["MCP"]["source"] == "review"

    def test_review_approved_status_not_overwritten(
        self, db: sqlite3.Connection, tmp_path: Path, settings: Settings
    ) -> None:
        path = build_workbook(tmp_path / "book.xlsx")
        _loaded(db, path, settings)
        load_project(db, path, settings.workbook)
        LexiconRepository(db).apply_review("Финам", "approved")
        db.commit()
        load_project(db, path, settings.workbook)
        status = db.execute("SELECT status FROM lexicon WHERE canonical = 'Финам'").fetchone()[0]
        assert status == "approved"

    def test_facts_loaded(self, db: sqlite3.Connection, tmp_path: Path, settings: Settings) -> None:
        path = build_workbook(
            tmp_path / "book.xlsx",
            extra_sheets={"Факты_Финам": [["F1", "Права", "только чтение", None, "черновик"]]},
        )
        _loaded(db, path, settings)
        result = load_project(db, path, settings.workbook)
        assert result.counts["facts"] == 1
        row = db.execute("SELECT sheet_name, external_id, answer FROM reference_fact").fetchone()
        assert row == ("Факты_Финам", "F1", "только чтение")

    def test_requires_codebook_loaded_first(self, db: sqlite3.Connection, tmp_path: Path, settings: Settings) -> None:
        path = build_workbook(tmp_path / "book.xlsx")
        with pytest.raises(RuntimeError, match="load-codebook"):
            load_project(db, path, settings.workbook)
