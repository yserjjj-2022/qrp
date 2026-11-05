"""Тесты системы миграций."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from qa.db.connection import get_connection
from qa.db.migrations import migrate

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations"
_TABLES = [
    "interview",
    "turn",
    "fragment",
    "codebook_version",
    "code",
    "attribute",
    "question",
    "annotation",
    "review",
    "run",
    "theme",
    "theme_code",
    "embedding",
    "code_proposal",
    "theme_proposal",
]
_WORKBOOK_TABLES = [
    "workbook_version",
    "workbook_sheet",
    "question_code",
    "attribute_definition",
    "glossary_term",
    "lexicon",
    "uncertainty_marker",
    "golden_sample",
    "reference_fact",
    "interview_speaker",
    "new_term",
]


@pytest.fixture()
def tmp_db(tmp_path: Path) -> tuple[sqlite3.Connection, Path, Path]:
    """Временная БД, путь к ней и каталог для бэкапов."""
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    yield conn, db_path, tmp_path
    conn.close()


class TestMigrations:
    """Проверка системы миграций."""

    def test_empty_db(self, tmp_db: tuple[sqlite3.Connection, Path, Path]) -> None:
        """Миграции создают все таблицы каркаса, M1a и рабочей книги."""
        conn, db_path, tmp_path = tmp_db
        backup_dir = tmp_path / "backups"
        applied = migrate(conn, db_path, _MIGRATIONS_DIR, backup_dir)
        expected = sorted(path.stem for path in _MIGRATIONS_DIR.glob("*.sql"))
        assert applied == expected
        assert len(applied) >= 3
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        for table in _TABLES + _WORKBOOK_TABLES:
            assert table in tables, f"Таблица {table} не найдена"
        assert "_guard_empty" not in tables
        assert "_guard_fk" not in tables
        cols = {row[1] for row in conn.execute("PRAGMA table_info(interview)").fetchall()}
        assert {"respondent_code", "group", "transcript_source", "interview_date"} <= cols
        turn_cols = {row[1] for row in conn.execute("PRAGMA table_info(turn)").fetchall()}
        assert {"line_no", "start_sec", "speaker_role", "pii_flag"} <= turn_cols
        code_cols = {row[1] for row in conn.execute("PRAGMA table_info(code)").fetchall()}
        assert {"kind", "title", "tiebreak_rule"} <= code_cols

    def test_idempotent(self, tmp_db: tuple[sqlite3.Connection, Path, Path]) -> None:
        """Повторное применение не создаёт ошибок."""
        conn, db_path, tmp_path = tmp_db
        backup_dir = tmp_path / "backups"
        migrate(conn, db_path, _MIGRATIONS_DIR, backup_dir)
        applied_again = migrate(conn, db_path, _MIGRATIONS_DIR, backup_dir)
        assert applied_again == []

    def test_backup_created(self, tmp_db: tuple[sqlite3.Connection, Path, Path]) -> None:
        """Перед миграцией создаётся копия БД (если файл существует и не пуст)."""
        conn, db_path, tmp_path = tmp_db
        backup_dir = tmp_path / "backups"
        # Сначала создаём таблицы, чтобы файл был непустым
        migrate(conn, db_path, _MIGRATIONS_DIR, backup_dir)
        conn.close()
        # Повторное подключение и миграция
        conn2 = get_connection(db_path)
        migrate(conn2, db_path, _MIGRATIONS_DIR, backup_dir)
        backups = list(backup_dir.glob("*.db"))
        assert len(backups) >= 1
        conn2.close()

    def test_no_backup_when_no_pending(self, tmp_db: tuple[sqlite3.Connection, Path, Path]) -> None:
        """Нет бэкапа при повторном запуске без новых миграций."""
        conn, db_path, tmp_path = tmp_db
        backup_dir = tmp_path / "backups"
        # Первая миграция: создаёт таблицы и (поскольку файл уже не пуст) бэкап
        migrate(conn, db_path, _MIGRATIONS_DIR, backup_dir)
        conn.close()
        backups_after_first = len(list(backup_dir.glob("*.db")))
        # Повторное применение: нет неприменённых, бэкап НЕ создаётся
        conn2 = get_connection(db_path)
        migrate(conn2, db_path, _MIGRATIONS_DIR, backup_dir)
        conn2.close()
        backups_after_second = len(list(backup_dir.glob("*.db")))
        assert backups_after_second == backups_after_first

    def test_0002_aborts_on_nonempty_turn(self, tmp_path: Path) -> None:
        """Миграция 0002 останавливается, если turn не пуста."""
        db_path = tmp_path / "test.db"
        # Применяем только 0001: копируем её в отдельный каталог
        partial_dir = tmp_path / "migrations_partial"
        partial_dir.mkdir()
        shutil.copy(_MIGRATIONS_DIR / "0001_create_tables.sql", partial_dir)
        conn = get_connection(db_path)
        migrate(conn, db_path, partial_dir, tmp_path / "backups")
        # Заполняем turn данными по старой схеме
        conn.execute("INSERT INTO interview (id_external) VALUES ('IVX1')")
        conn.execute("INSERT INTO turn (interview_id, speaker, text) VALUES (1, 'A', 'текст')")
        conn.commit()
        # Полная миграция должна остановиться на гарде пустоты
        with pytest.raises(sqlite3.IntegrityError):
            migrate(conn, db_path, _MIGRATIONS_DIR, tmp_path / "backups")
        # 0002 не записана как применённая
        applied = {row[0] for row in conn.execute("SELECT name FROM applied_migrations").fetchall()}
        assert "0002_transcript_turns" not in applied
        # Режим FK восстановлен после сбоя
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        conn.close()

    def test_fk_enforced_after_migrations(self, tmp_db: tuple[sqlite3.Connection, Path, Path]) -> None:
        """После миграций внешние ключи включены и проверяются."""
        conn, db_path, tmp_path = tmp_db
        migrate(conn, db_path, _MIGRATIONS_DIR, tmp_path / "backups")
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO utterance (interview_id, speaker_role, text, turn_ids)"
                " VALUES (999, 'RESPONDENT', 'x', '[]')"
            )
