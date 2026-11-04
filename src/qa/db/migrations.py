"""Система миграций SQLite."""

from __future__ import annotations

import logging
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_MIGRATIONS_TABLE = "applied_migrations"


def _ensure_migrations_table(conn: sqlite3.Connection) -> None:
    """Создать таблицу applied_migrations, если её нет."""
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {_MIGRATIONS_TABLE} ("
        "    id INTEGER PRIMARY KEY,"
        "    name TEXT NOT NULL UNIQUE,"
        "    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        ")"
    )


def _get_applied(conn: sqlite3.Connection) -> set[str]:
    """Вернуть множество имён применённых миграций."""
    cursor = conn.execute(f"SELECT name FROM {_MIGRATIONS_TABLE}")  # noqa: S608
    return {row[0] for row in cursor.fetchall()}


def _pending(migrations_dir: Path, applied: set[str]) -> list[Path]:
    """Отсортированные файлы миграций, которые ещё не применены."""
    all_files = sorted(migrations_dir.glob("*.sql"), key=lambda p: p.name)
    return [p for p in all_files if p.stem not in applied]


def _backup_if_needed(db_path: Path, backup_dir: Path) -> None:
    """Сохранить копию БД, если файл существует и не пуст."""
    if not db_path.exists():
        return
    if db_path.stat().st_size == 0:
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{timestamp}.db"
    shutil.copy2(db_path, backup_path)
    logger.info("Резервная копия БД: %s", backup_path)


def migrate(conn: sqlite3.Connection, db_path: Path, migrations_dir: Path, backup_dir: Path) -> list[str]:
    """Применить неприменённые миграции.

    Args:
        conn: sqlite3.Connection
        db_path: путь к файлу БД
        migrations_dir: каталог с .sql файлами
        backup_dir: каталог для резервных копий

    Returns:
        Список имён применённых миграций.
    """
    # Проверяем состояние файла ДО любых изменений
    file_was_non_empty = db_path.exists() and db_path.stat().st_size > 0

    # Проверяем наличие неприменённых миграций
    _ensure_migrations_table(conn)
    applied = _get_applied(conn)
    pending = _pending(migrations_dir, applied)

    if not pending:
        logger.info("Все миграции уже применены")
        return []

    # Копия ДО любых изменений (если файл был не пустым)
    if file_was_non_empty:
        _backup_if_needed(db_path, backup_dir)

    applied_list: list[str] = []
    for migration_file in pending:
        sql = migration_file.read_text(encoding="utf-8")
        logger.info("Применяю миграцию: %s", migration_file.name)
        conn.executescript(sql)
        conn.execute(f"INSERT INTO {_MIGRATIONS_TABLE} (name) VALUES (?)", (migration_file.stem,))
        conn.commit()
        applied_list.append(migration_file.stem)

    return applied_list
