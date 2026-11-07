"""Тесты выгрузки единиц L2 для ручной проверки.

Все тексты синтетические, файл пишется во временный каталог.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from openpyxl import load_workbook

from qa.db.connection import get_connection
from qa.db.migrations import migrate
from qa.pipeline.review_export import COLUMNS, export_manual_review

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations"


@pytest.fixture()
def db(tmp_path: Path) -> sqlite3.Connection:
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    migrate(conn, db_path, _MIGRATIONS_DIR, tmp_path / "backups")
    yield conn
    conn.close()


def _seed(conn: sqlite3.Connection, units: list[tuple[str, str, str, list[dict[str, object]]]]) -> None:
    """Интервью R01 и единицы L2 версии 1: (роль, L1, L2, changes)."""
    cursor = conn.execute("INSERT INTO interview (id_external, name) VALUES ('R01', 'тест')")
    interview_id = int(cursor.lastrowid or 0)
    for role, text_l1, text_l2, changes in units:
        cursor = conn.execute(
            "INSERT INTO utterance (interview_id, speaker_role, text, turn_ids, is_backchannel)"
            " VALUES (?, ?, ?, '[]', 0)",
            (interview_id, role, text_l1),
        )
        utterance_id = int(cursor.lastrowid or 0)
        conn.execute(
            "INSERT INTO text_unit (utterance_id, version, text, status, asr_quality, changes_json)"
            " VALUES (?, 1, ?, 'proposed', 'ok', ?)",
            (utterance_id, text_l2, json.dumps(changes, ensure_ascii=False)),
        )
    conn.commit()


def _rows(path: Path) -> list[tuple[object, ...]]:
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb[wb.sheetnames[0]]
        return [tuple(row) for row in ws.iter_rows(min_row=2, values_only=True)]
    finally:
        wb.close()


def test_flagged_units_included(db: sqlite3.Connection, tmp_path: Path) -> None:
    _seed(
        db,
        [
            ("RESPONDENT", "там нечечно", "Там нечечно.", [{"type": "asr_fix", "from": "нечечно", "to": "нечто"}]),
            ("RESPONDENT", "проверка", "Проверка.", []),
        ],
    )
    result = export_manual_review(db, "R01", 1, tmp_path / "review.xlsx", n_random=0)
    assert result.flagged == 1
    assert result.sampled == 0
    rows = _rows(result.path)
    assert len(rows) == 1
    assert rows[0][2] == "asr_fix"


def test_names_review_and_doubtful_flags(db: sqlite3.Connection, tmp_path: Path) -> None:
    _seed(
        db,
        [
            ("RESPONDENT", "одна реплика", "Одна реплика.", [{"type": "names_review", "detail": "x"}]),
        ],
    )
    db.execute("UPDATE text_unit SET asr_quality = 'doubtful'")
    db.commit()
    result = export_manual_review(db, "R01", 1, tmp_path / "review.xlsx", n_random=0)
    rows = _rows(result.path)
    assert rows[0][2] == "doubtful, names_review"


def test_random_sample_is_reproducible(db: sqlite3.Connection, tmp_path: Path) -> None:
    _seed(db, [("RESPONDENT", f"реплика номер {index}", f"Реплика номер {index}.", []) for index in range(20)])
    first = export_manual_review(db, "R01", 1, tmp_path / "a.xlsx", n_random=5, seed=7)
    second = export_manual_review(db, "R01", 1, tmp_path / "b.xlsx", n_random=5, seed=7)
    assert first.sampled == 5
    assert [row[0] for row in _rows(first.path)] == [row[0] for row in _rows(second.path)]


def test_columns_and_changes_text(db: sqlite3.Connection, tmp_path: Path) -> None:
    _seed(db, [("RESPONDENT", "эмсипи", "MCP.", [{"type": "term", "from": "эмсипи", "to": "MCP"}])])
    result = export_manual_review(db, "R01", 1, tmp_path / "review.xlsx", n_random=15)
    rows = _rows(result.path)
    assert len(rows[0]) == len(COLUMNS)
    assert rows[0][3] == "эмсипи"
    assert rows[0][4] == "MCP."
    assert "эмсипи" in str(rows[0][7])


def test_missing_version_raises(db: sqlite3.Connection, tmp_path: Path) -> None:
    _seed(db, [("RESPONDENT", "реплика", "Реплика.", [])])
    with pytest.raises(ValueError, match="Нет единиц"):
        export_manual_review(db, "R01", 99, tmp_path / "review.xlsx")
