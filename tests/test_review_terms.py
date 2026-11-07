"""Тесты review-terms: решения оператора и добавление вариантов в лексикон.

Все данные синтетические, ввод подменяется функцией.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from qa.db.connection import get_connection
from qa.db.migrations import migrate
from qa.db.workbook_repo import LexiconRepository, NewTermRepository
from qa.pipeline.review_terms import _parse_answer, review_terms

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations"


@pytest.fixture()
def db(tmp_path: Path) -> sqlite3.Connection:
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    migrate(conn, db_path, _MIGRATIONS_DIR, tmp_path / "backups")
    yield conn
    conn.close()


def _seed_terms(conn: sqlite3.Connection, forms: list[str]) -> None:
    repo = NewTermRepository(conn)
    for form in forms:
        repo.add(form, None, None)
    conn.commit()


def _answers(values: list[str]):  # noqa: ANN202
    """Подменить ввод: отдать значения по очереди."""
    queue = list(values)

    def ask(_prompt: str) -> str:
        return queue.pop(0)

    return ask


class TestParseAnswer:
    """Разбор ответа оператора одной строкой."""

    @pytest.mark.parametrize("value", ["y", "Y", "н", "да", "yes"])
    def test_approve(self, value: str) -> None:
        assert _parse_answer(value) == ("approved", "")

    def test_approve_with_canonical(self) -> None:
        assert _parse_answer("y MCP") == ("approved", "MCP")

    def test_approve_with_spaces_in_canonical(self) -> None:
        assert _parse_answer("y Yahoo Finance") == ("approved", "Yahoo Finance")

    @pytest.mark.parametrize("value", ["n", "N", "т", "нет", "no"])
    def test_reject(self, value: str) -> None:
        assert _parse_answer(value) == ("rejected", "")

    @pytest.mark.parametrize("value", ["", "skip", "s", "непонятно"])
    def test_skip(self, value: str) -> None:
        assert _parse_answer(value) == ("skip", "")


class TestReviewTerms:
    """Запись решений и вариантов."""

    def test_approve_as_variant(self, db: sqlite3.Connection) -> None:
        _seed_terms(db, ["эсипи"])
        result = review_terms(db, ask=_answers(["y MCP"]))

        assert result.variants == {"эсипи": "MCP"}
        variants = LexiconRepository(db).all_terms()["MCP"]["variants"]
        assert "эсипи" in variants

    def test_variant_has_review_source(self, db: sqlite3.Connection) -> None:
        _seed_terms(db, ["эсипи"])
        review_terms(db, ask=_answers(["y MCP"]))

        row = db.execute("SELECT source, status FROM lexicon WHERE canonical = 'MCP'").fetchone()
        assert row == ("review", "approved")

    def test_variant_keeps_existing_variants(self, db: sqlite3.Connection) -> None:
        LexiconRepository(db).upsert_from_workbook("MCP", ["мсп"], None, "approved")
        db.commit()
        _seed_terms(db, ["эсипи"])
        review_terms(db, ask=_answers(["y MCP"]))

        variants = LexiconRepository(db).all_terms()["MCP"]["variants"]
        assert "мсп" in variants
        assert "эсипи" in variants

    def test_duplicate_variant_not_added(self, db: sqlite3.Connection) -> None:
        LexiconRepository(db).upsert_from_workbook("MCP", ["эсипи"], None, "approved")
        db.commit()
        _seed_terms(db, ["эсипи"])
        result = review_terms(db, ask=_answers(["y MCP"]))

        assert result.variants == {}
        variants = LexiconRepository(db).all_terms()["MCP"]["variants"]
        assert variants.count("эсипи") == 1

    def test_fallback_second_question(self, db: sqlite3.Connection) -> None:
        _seed_terms(db, ["эсипи"])
        result = review_terms(db, ask=_answers(["y"]), ask_canonical=_answers(["MCP"]))

        assert result.variants == {"эсипи": "MCP"}

    def test_second_question_not_asked_with_canonical(self, db: sqlite3.Connection) -> None:
        _seed_terms(db, ["эсипи"])

        def fail(_prompt: str) -> str:
            msg = "второй вопрос не должен задаваться"
            raise AssertionError(msg)

        review_terms(db, ask=_answers(["y MCP"]), ask_canonical=fail)

    def test_approve_as_new_term(self, db: sqlite3.Connection) -> None:
        _seed_terms(db, ["вертулятор"])
        result = review_terms(db, ask=_answers(["y"]), ask_canonical=_answers([""]))

        assert result.variants == {}
        terms = LexiconRepository(db).all_terms()
        assert "вертулятор" in terms
        assert terms["вертулятор"]["status"] == "approved"

    def test_reject_marks_candidate(self, db: sqlite3.Connection) -> None:
        _seed_terms(db, ["годаговор"])
        result = review_terms(db, ask=_answers(["n"]))

        assert result.rejected == ["годаговор"]
        row = db.execute("SELECT status FROM new_term WHERE surface_form = 'годаговор'").fetchone()
        assert row[0] == "rejected"

    def test_reject_does_not_touch_lexicon(self, db: sqlite3.Connection) -> None:
        _seed_terms(db, ["годаговор"])
        review_terms(db, ask=_answers(["n"]))

        assert LexiconRepository(db).all_terms() == {}

    def test_skip_keeps_candidate(self, db: sqlite3.Connection) -> None:
        _seed_terms(db, ["Снджи"])
        result = review_terms(db, ask=_answers(["skip"]))

        assert result.skipped == ["Снджи"]
        row = db.execute("SELECT status FROM new_term WHERE surface_form = 'Снджи'").fetchone()
        assert row[0] == "proposed"

    def test_only_filters_candidates(self, db: sqlite3.Connection) -> None:
        _seed_terms(db, ["эсипи", "годаговор"])
        result = review_terms(db, ask=_answers(["y MCP"]), only=["эсипи"])

        assert result.approved == ["эсипи"]
        statuses = dict(db.execute("SELECT surface_form, status FROM new_term").fetchall())
        assert statuses["годаговор"] == "proposed"

    def test_variant_survives_reload(self, db: sqlite3.Connection) -> None:
        """load-project не затирает вариант, добавленный в review-terms."""
        _seed_terms(db, ["эсипи"])
        review_terms(db, ask=_answers(["y MCP"]))

        LexiconRepository(db).upsert_from_workbook("MCP", ["мсп"], None, "approved")
        db.commit()

        variants = LexiconRepository(db).all_terms()["MCP"]["variants"]
        assert "эсипи" in variants
