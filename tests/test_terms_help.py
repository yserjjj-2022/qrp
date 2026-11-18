"""Таблица терминов проекта и команда apply-terms (M7b)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from qa.db.workbook_repo import LexiconRepository
from qa.pipeline.normalize import load_lexicon
from qa.pipeline.terms_help import (
    HelpRow,
    apply_terms,
    collect_open_forms,
    merge_rows,
    read_help_rows,
    terms_path,
    upsert_terms_table,
    write_terms_table,
)
from qa.pipeline.terms_memory import UNREADABLE, TermHint
from qa.settings import Settings
from tests.fixtures.m8_db import AnnotationSpec, M8Spec, UtteranceSpec, build_m8_db

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TEXT = "Я подключил сенжин к счёту и всё работает"


def _settings(tmp_path: Path) -> Settings:
    loaded = Settings.load(_PROJECT_ROOT / "config.yaml")
    loaded.paths.exports = str(tmp_path / "exports")
    loaded.paths.reports = str(tmp_path / "reports")
    loaded.paths.transcripts = str(tmp_path / "transcripts")
    return loaded


def _spec(*, annotations: list[AnnotationSpec] | None = None) -> M8Spec:
    return M8Spec(
        utterances=[UtteranceSpec("RESPONDENT", _TEXT, _TEXT + ".", 40, "5")],
        annotations=annotations or [],
        lexicon=("MCP",),
    )


def _table(settings: Settings, choice: str, confidence: float = 0.9) -> Path:
    path = terms_path(settings)
    write_terms_table(
        path,
        [
            HelpRow(
                form="сенжин",
                frequency=1,
                places="R01: 00:40",
                hints=(TermHint("OsEngine", confidence, "model"),),
                choice=choice,
            )
        ],
    )
    return path


def _text_l2(conn: sqlite3.Connection) -> str:
    return str(conn.execute("SELECT text FROM text_unit ORDER BY id LIMIT 1").fetchone()[0])


class TestCollect:
    def test_open_form_found_in_l2(self, tmp_path: Path) -> None:
        conn = build_m8_db(tmp_path, _spec())
        opened = collect_open_forms(conn, "R01", load_lexicon(conn))
        assert [item.form for item in opened] == ["сенжин"]
        assert "R01: 00:40" in opened[0].places
        conn.close()

    def test_known_word_not_reported(self, tmp_path: Path) -> None:
        conn = build_m8_db(
            tmp_path, M8Spec(utterances=[UtteranceSpec("RESPONDENT", "Подключил счёт", "Подключил счёт.", 5)])
        )
        assert collect_open_forms(conn, "R01", load_lexicon(conn)) == []
        conn.close()


class TestTable:
    def test_roundtrip_keeps_choice_and_hints(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        _table(settings, "OsEngine")
        rows = read_help_rows(terms_path(settings))
        assert rows[0].choice == "OsEngine"
        assert rows[0].hints[0].to == "OsEngine"
        assert rows[0].hints[0].confidence == pytest.approx(0.9)

    def test_merge_does_not_overwrite_operator_choice(self) -> None:
        existing = [HelpRow(form="сенжин", frequency=1, choice="OsEngine")]
        opened = [HelpRow(form="сенжин", frequency=5), HelpRow(form="всенж", frequency=2)]
        merged = merge_rows(existing, opened)
        assert {row.form for row in merged} == {"сенжин", "всенж"}
        assert merged[0].form == "всенж"  # сортировка по частоте
        assert next(row for row in merged if row.form == "сенжин").choice == "OsEngine"
        assert next(row for row in merged if row.form == "сенжин").frequency == 1

    def test_upsert_appends_new_forms(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        _table(settings, "=")
        upsert_terms_table(terms_path(settings), [HelpRow(form="всенж", frequency=2)])
        rows = read_help_rows(terms_path(settings))
        assert {row.form for row in rows} == {"сенжин", "всенж"}
        assert next(row for row in rows if row.form == "сенжин").choice == "="


class TestApply:
    def test_operator_replacement_updates_lexicon_and_l2(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        conn = build_m8_db(tmp_path, _spec())
        _table(settings, "OsEngine")
        result = apply_terms(conn, settings)
        assert result.applied == 1
        assert "OsEngine" in _text_l2(conn)
        entry = next(item for item in load_lexicon(conn) if item.canonical == "OsEngine")
        assert entry.status == "review"
        assert "сенжин" in entry.variants
        assert read_help_rows(terms_path(settings)) == []
        conn.close()

    def test_empty_cell_applies_first_hint_as_auto(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        conn = build_m8_db(tmp_path, _spec())
        _table(settings, "", confidence=0.9)
        result = apply_terms(conn, settings)
        assert result.auto == 1
        entry = next(item for item in load_lexicon(conn) if item.canonical == "OsEngine")
        assert entry.status == "auto"
        conn.close()

    def test_empty_cell_below_threshold_keeps_word(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        conn = build_m8_db(tmp_path, _spec())
        _table(settings, "", confidence=0.3)
        apply_terms(conn, settings)
        entry = next(item for item in load_lexicon(conn) if item.canonical == "сенжин")
        assert entry.status == "keep"
        assert "сенжин" in _text_l2(conn)
        conn.close()

    def test_equals_keeps_word(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        conn = build_m8_db(tmp_path, _spec())
        _table(settings, "=")
        apply_terms(conn, settings)
        assert "сенжин" in _text_l2(conn)
        conn.close()

    def test_question_marks_unreadable(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        conn = build_m8_db(tmp_path, _spec())
        _table(settings, "?")
        apply_terms(conn, settings)
        assert UNREADABLE in _text_l2(conn)
        conn.close()

    def test_second_run_is_idempotent(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        conn = build_m8_db(tmp_path, _spec())
        _table(settings, "OsEngine")
        apply_terms(conn, settings)
        text = _text_l2(conn)
        second = apply_terms(conn, settings)
        assert second.applied == 0
        assert _text_l2(conn) == text
        conn.close()

    def test_annotations_freeze_rebuild(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        spec = _spec(annotations=[AnnotationSpec(0, "C1", "подключил")])
        conn = build_m8_db(tmp_path, spec)
        _table(settings, "OsEngine")
        before = _text_l2(conn)
        result = apply_terms(conn, settings)
        assert result.skipped == ("R01",)
        assert _text_l2(conn) == before
        conn.close()

    def test_conflict_keeps_row_and_text(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        conn = build_m8_db(tmp_path, _spec())
        LexiconRepository(conn).set_form_decision("сенжин", "Другой", "review")
        conn.commit()
        _table(settings, "OsEngine")
        result = apply_terms(conn, settings)
        assert result.conflicts == 1
        # Решение оператора не перезаписано: в L2 стоит «Другой», а не «OsEngine».
        assert "Другой" in _text_l2(conn)
        assert "OsEngine" not in _text_l2(conn)
        assert [row.form for row in read_help_rows(terms_path(settings))] == ["сенжин"]
        conn.close()

    def test_missing_table_is_reported(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        conn = build_m8_db(tmp_path, _spec())
        with pytest.raises(Exception, match="Таблица терминов не найдена"):
            apply_terms(conn, settings)
        conn.close()
