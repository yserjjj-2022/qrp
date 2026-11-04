"""Загрузчик дерева тем из YAML в таблицы theme и theme_code."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_themes(conn: sqlite3.Connection, yaml_path: Path) -> tuple[int, int]:
    """Загрузить дерево тем из YAML.

    Args:
        conn: подключение к SQLite
        yaml_path: путь к themes.yaml

    Returns:
        Кортеж (n_themes, n_bindings).
    """
    if not yaml_path.exists():
        msg = f"Файл не найден: {yaml_path}"
        raise FileNotFoundError(msg)

    with open(yaml_path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    if data.get("status") == "draft":
        logger.info("Загружается черновик дерева тем")

    nodes = data.get("nodes", [])
    no_theme_codes = data.get("no_theme", [])

    # Строим карту code_name -> code_id для текущего кодбука
    code_map: dict[str, int] = {}
    for row in conn.execute("SELECT id, name FROM code WHERE status = 'active'"):
        code_map[row[1]] = row[0]

    n_themes = 0
    n_bindings = 0
    bound_codes: set[str] = set()

    for node in nodes:
        # Узел верхнего уровня
        node_name = node.get("name", "")
        cursor = conn.execute("INSERT INTO theme (name, parent_id, level) VALUES (?, NULL, 0)", (node_name,))
        node_id = cursor.lastrowid
        n_themes += 1

        # Вложенные темы
        for theme in node.get("themes", []):
            theme_name = theme.get("name", "")
            cursor2 = conn.execute("INSERT INTO theme (name, parent_id, level) VALUES (?, ?, 1)", (theme_name, node_id))
            theme_id = cursor2.lastrowid
            n_themes += 1

            # Привязка кодов к теме
            for code_name in theme.get("codes", []):
                code_id = code_map.get(code_name)
                if code_id is None:
                    logger.warning("Код '%s' не найден в БД, пропускается привязка", code_name)
                    continue
                conn.execute(
                    "INSERT OR IGNORE INTO theme_code (theme_id, code_id) VALUES (?, ?)",
                    (theme_id, code_id),
                )
                n_bindings += 1
                bound_codes.add(code_name)

    # Проверяем: каждый активный код привязан или в no_theme
    all_codes = set(code_map.keys())
    unbound = all_codes - bound_codes - set(no_theme_codes)
    if unbound:
        logger.warning(
            "Не привязаны к темам и не помечены «без темы»: %s",
            ", ".join(sorted(unbound)),
        )

    conn.commit()
    logger.info("Дерево тем загружено: themes=%s, bindings=%s", n_themes, n_bindings)
    return n_themes, n_bindings
