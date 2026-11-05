"""Тесты золотого набора: экспорт L1 в xlsx и импорт заполненного файла."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from openpyxl import Workbook, load_workbook

from qa.db.connection import get_connection
from qa.db.migrations import migrate
from qa.pipeline.golden import COLUMNS, export_golden_sample, import_golden

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations"


@pytest.fixture()
def db(tmp_path: Path) -> sqlite3.Connection:
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    migrate(conn, db_path, _MIGRATIONS_DIR, tmp_path / "backups")
    yield conn
    conn.close()


def _add_interview(conn: sqlite3.Connection, interview_id: str, texts: list[tuple[str, int]]) -> None:
    """Добавить интервью и реплики: (текст, is_backchannel)."""
    cursor = conn.execute(
        "INSERT INTO interview (id_external, name, respondent_code) VALUES (?, NULL, ?)",
        (interview_id, interview_id),
    )
    db_id = cursor.lastrowid
    for text, backchannel in texts:
        conn.execute(
            "INSERT INTO utterance (interview_id, speaker_role, text, turn_ids, is_backchannel)"
            " VALUES (?, 'RESPONDENT', ?, '[]', ?)",
            (db_id, text, backchannel),
        )
    conn.commit()


class TestExport:
    """Выгрузка реплик L1."""

    def test_exports_respondent_only(self, db: sqlite3.Connection, tmp_path: Path) -> None:
        _add_interview(db, "R01", [("первая", 0), ("угу", 1), ("вторая", 0)])
        output = tmp_path / "golden.xlsx"
        result = export_golden_sample(db, output, n=10)
        assert result.n_rows == 2
        assert output.exists()

        wb = load_workbook(output)
        ws = wb[wb.sheetnames[0]]
        headers = [cell.value for cell in ws[1]]
        assert headers == COLUMNS
        texts = [row[2] for row in ws.iter_rows(min_row=2, values_only=True)]
        assert texts == ["первая", "вторая"]
        # Колонка ручного L2 пустая
        assert all(row[3] is None for row in ws.iter_rows(min_row=2, values_only=True))

    def test_sample_size_and_seed(self, db: sqlite3.Connection, tmp_path: Path) -> None:
        _add_interview(db, "R01", [(f"текст {i}", 0) for i in range(20)])
        first = export_golden_sample(db, tmp_path / "a.xlsx", n=5, seed=7)
        second = export_golden_sample(db, tmp_path / "b.xlsx", n=5, seed=7)
        assert first.n_rows == second.n_rows == 5

    def test_interview_filter(self, db: sqlite3.Connection, tmp_path: Path) -> None:
        _add_interview(db, "R01", [("один", 0)])
        _add_interview(db, "R02", [("два", 0), ("три", 0)])
        result = export_golden_sample(db, tmp_path / "g.xlsx", n=10, interview_id="R02")
        assert result.n_rows == 2

    def test_empty_db_raises(self, db: sqlite3.Connection, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="нет реплик"):
            export_golden_sample(db, tmp_path / "g.xlsx")


class TestImport:
    """Импорт заполненного файла."""

    def test_roundtrip(self, db: sqlite3.Connection, tmp_path: Path) -> None:
        _add_interview(db, "R01", [("первая", 0), ("вторая", 0)])
        output = tmp_path / "golden.xlsx"
        export_golden_sample(db, output, n=10)

        wb = load_workbook(output)
        ws = wb[wb.sheetnames[0]]
        ws.cell(row=2, column=4, value="первая L2")
        wb.save(output)

        result = import_golden(db, output)
        assert result.n_rows == 2
        assert result.n_filled == 1
        rows = dict(db.execute("SELECT text_l1, text_l2_manual FROM golden_sample").fetchall())
        assert rows["первая"] == "первая L2"
        assert rows["вторая"] is None

    def test_import_is_idempotent(self, db: sqlite3.Connection, tmp_path: Path) -> None:
        path = tmp_path / "manual.xlsx"
        wb = Workbook()
        ws = wb.active
        ws.append(COLUMNS)
        ws.append(["R01", "utterance:1", "текст", "нормализованный"])
        wb.save(path)

        import_golden(db, path)
        import_golden(db, path)
        assert db.execute("SELECT COUNT(*) FROM golden_sample").fetchone()[0] == 1
        assert db.execute("SELECT text_l2_manual FROM golden_sample").fetchone()[0] == "нормализованный"

    def test_missing_file_raises(self, db: sqlite3.Connection, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            import_golden(db, tmp_path / "нет.xlsx")

    def test_missing_column_raises(self, db: sqlite3.Connection, tmp_path: Path) -> None:
        path = tmp_path / "bad.xlsx"
        wb = Workbook()
        ws = wb.active
        ws.append(["id_интервью", "что-то"])
        ws.append(["R01", "текст"])
        wb.save(path)
        with pytest.raises(ValueError, match="текст_L1"):
            import_golden(db, path)
