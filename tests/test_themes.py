"""Тесты дерева тем."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import yaml

from qa.db.connection import get_connection
from qa.db.migrations import migrate
from qa.pipeline.load_themes import load_themes

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations"


@pytest.fixture()
def db(tmp_path: Path) -> sqlite3.Connection:
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    migrate(conn, db_path, _MIGRATIONS_DIR, tmp_path / "backups")
    yield conn
    conn.close()


def _create_theme_yaml(tmp_path: Path, codes: list[str], unbound: list[str] | None = None) -> Path:
    """Создать временный YAML с привязкой кодов к темам."""
    data = {
        "status": "draft",
        "nodes": [
            {
                "name": "Риски",
                "key": "risks",
                "themes": [
                    {
                        "name": "Финансовые",
                        "key": "fin",
                        "codes": codes[:3],
                    },
                    {
                        "name": "Безопасность",
                        "key": "sec",
                        "codes": codes[3:],
                    },
                ],
            }
        ],
        "no_theme": unbound or [],
    }
    path = tmp_path / "themes.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)
    return path


def _insert_codes(conn: sqlite3.Connection, names: list[str]) -> None:
    """Вставить коды в БД для тестов."""
    conn.execute("INSERT INTO codebook_version (file_hash) VALUES ('test_hash')")
    version_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    for name in names:
        conn.execute(
            "INSERT INTO code (codebook_version_id, family, name, origin, status) VALUES (?, 'f', ?, 'base', 'active')",
            (version_id, name),
        )
    conn.commit()


class TestLoadThemes:
    """Проверка загрузки дерева тем."""

    def test_basic_load(self, db: sqlite3.Connection, tmp_path: Path) -> None:
        """Корректная загрузка узлов и тем."""
        codes = ["R1", "R2", "R3", "O1", "O2"]
        _insert_codes(db, codes)
        yaml_path = _create_theme_yaml(tmp_path, codes)
        n_themes, n_bindings = load_themes(db, yaml_path)
        # 1 узел + 2 темы = 3, 5 привязок
        assert n_themes == 3
        assert n_bindings == 5

    def test_coverage_all_bound(self, db: sqlite3.Connection, tmp_path: Path) -> None:
        """Все коды привязаны к темам."""
        codes = ["R1", "R2", "R3", "O1", "O2"]
        _insert_codes(db, codes)
        yaml_path = _create_theme_yaml(tmp_path, codes)
        load_themes(db, yaml_path)
        # Проверяем: все коды имеют запись в theme_code
        count = db.execute("SELECT COUNT(DISTINCT code_id) FROM theme_code").fetchone()[0]
        assert count == len(codes)

    def test_coverage_no_theme(self, db: sqlite3.Connection, tmp_path: Path) -> None:
        """Код помечен «без темы» — не привязан к theme_code."""
        codes = ["R1", "R2", "UNBOUND"]
        _insert_codes(db, codes)
        yaml_path = _create_theme_yaml(tmp_path, ["R1", "R2"], unbound=["UNBOUND"])
        load_themes(db, yaml_path)
        # UNBOUND не должен быть в theme_code
        count = db.execute("SELECT COUNT(*) FROM theme_code").fetchone()[0]
        assert count == 2
