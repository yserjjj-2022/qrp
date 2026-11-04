"""Тесты системы миграций."""

from __future__ import annotations

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
        """Миграция создаёт все 15 таблиц на пустой БД."""
        conn, db_path, tmp_path = tmp_db
        backup_dir = tmp_path / "backups"
        applied = migrate(conn, db_path, _MIGRATIONS_DIR, backup_dir)
        assert len(applied) == 1
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        for table in _TABLES:
            assert table in tables, f"Таблица {table} не найдена"

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
