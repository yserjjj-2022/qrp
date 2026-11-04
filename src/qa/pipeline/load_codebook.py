"""Загрузчик кодбука из приложения Б (Excel)."""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from pathlib import Path
from typing import Any

from openpyxl import load_workbook
from openpyxl.workbook.workbook import Workbook
from openpyxl.worksheet.worksheet import Worksheet

logger = logging.getLogger(__name__)

_SHEET_CODES = "Коды"
_SHEET_QUESTIONS = "Вопросы_и_коды"
_SHEET_ATTRIBUTES = "Атрибуты"

_REQUIRED_COLUMNS: dict[str, list[str]] = {
    _SHEET_CODES: ["Код", "Семейство", "Название", "Определение", "Включаем", "Исключаем", "Привязка к вопросу"],
    _SHEET_QUESTIONS: ["Код", "Идентификатор вопроса"],
    _SHEET_ATTRIBUTES: ["Код", "Имя атрибута", "Значение"],
}


def _file_hash(path: Path) -> str:
    """SHA-256 хеш файла."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_headers(ws: Worksheet) -> list[str]:
    """Прочитать заголовки (первая строка) листа."""
    return [str(cell.value) if cell.value is not None else "" for cell in ws[1]]


def _validate_sheet(ws: Worksheet, sheet_name: str) -> dict[str, int]:
    """Проверить, что все обязательные колонки присутствуют. Вернуть имя->index."""
    headers = _read_headers(ws)
    required = _REQUIRED_COLUMNS[sheet_name]
    col_map: dict[str, int] = {}
    for col_name in required:
        found = None
        for i, h in enumerate(headers):
            if h.strip() == col_name:
                found = i
                break
        if found is None:
            msg = f"В листе '{sheet_name}' не найдена колонка '{col_name}'. Доступные колонки: {headers}"
            raise ValueError(msg)
        col_map[col_name] = found
    return col_map


def _row_value(row: tuple[Any, ...], col_map: dict[str, int], col_name: str) -> str | None:
    """Получить значение ячейки, приведя к строке."""
    idx = col_map[col_name]
    if idx >= len(row):
        return None
    val = row[idx].value
    if val is None:
        return None
    return str(val).strip()


def load_codebook(conn: sqlite3.Connection, excel_path: Path) -> tuple[int, int, int]:
    """Загрузить кодбук из Excel-файла в БД.

    Args:
        conn: подключение к SQLite
        excel_path: путь к .xlsx файлу

    Returns:
        Кортеж (n_codes, n_questions, n_attributes).

    Raises:
        FileNotFoundError: файл не найден.
        ValueError: отсутствует обязательный лист или колонка.
    """
    if not excel_path.exists():
        msg = f"Файл не найден: {excel_path}"
        raise FileNotFoundError(msg)

    file_hash = _file_hash(excel_path)

    # Проверка на повторную загрузку
    existing = conn.execute("SELECT id FROM codebook_version WHERE file_hash = ?", (file_hash,)).fetchone()
    if existing is not None:
        logger.info("Кодбук уже загружен (hash=%s, version_id=%s)", file_hash[:12], existing[0])
        return _count_records(conn, existing[0])

    wb = load_workbook(excel_path, read_only=True, data_only=True)

    # Проверяем наличие всех обязательных листов
    for sheet_name in (_SHEET_CODES, _SHEET_QUESTIONS, _SHEET_ATTRIBUTES):
        if sheet_name not in wb.sheetnames:
            msg = f"В файле отсутствует обязательный лист '{sheet_name}'. Доступные листы: {wb.sheetnames}"
            raise ValueError(msg)

    # Создаём версию
    conn.execute("INSERT INTO codebook_version (file_hash) VALUES (?)", (file_hash,))
    version_id: int = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    n_codes = _load_codes(conn, wb, version_id)
    n_questions = _load_questions(conn, wb, version_id)
    n_attributes = _load_attributes(conn, wb, version_id)

    conn.commit()
    wb.close()

    logger.info(
        "Кодбук загружен: version_id=%s, codes=%s, questions=%s, attributes=%s",
        version_id,
        n_codes,
        n_questions,
        n_attributes,
    )
    return n_codes, n_questions, n_attributes


def _load_codes(conn: sqlite3.Connection, wb: Workbook, version_id: int) -> int:
    """Загрузить лист Коды."""
    ws: Worksheet = wb[_SHEET_CODES]
    col_map = _validate_sheet(ws, _SHEET_CODES)

    n = 0
    for row in ws.iter_rows(min_row=2, values_only=False):
        code_id_str = _row_value(row, col_map, "Код")
        if not code_id_str:
            continue
        family = _row_value(row, col_map, "Семейство")
        definition = _row_value(row, col_map, "Определение")
        include = _row_value(row, col_map, "Включаем")
        exclude = _row_value(row, col_map, "Исключаем")
        # "Название" сохраняем в definition, если определение пустое
        name = _row_value(row, col_map, "Название")
        if not definition and name:
            definition = name

        conn.execute(
            "INSERT OR IGNORE INTO code "
            "(codebook_version_id, family, name, definition, include, exclude, origin, status) "
            "VALUES (?, ?, ?, ?, ?, ?, 'base', 'active')",
            (version_id, family, code_id_str, definition, include, exclude),
        )
        n += 1

    return n


def _load_questions(conn: sqlite3.Connection, wb: Workbook, version_id: int) -> int:
    """Загрузить лист Вопросы_и_коды."""
    ws: Worksheet = wb[_SHEET_QUESTIONS]
    col_map = _validate_sheet(ws, _SHEET_QUESTIONS)

    n = 0
    for row in ws.iter_rows(min_row=2, values_only=False):
        identifier = _row_value(row, col_map, "Идентификатор вопроса")
        if not identifier:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO question (codebook_version_id, identifier) VALUES (?, ?)",
            (version_id, identifier),
        )
        n += 1

    return n


def _load_attributes(conn: sqlite3.Connection, wb: Workbook, version_id: int) -> int:
    """Загрузить лист Атрибуты."""
    ws = wb[_SHEET_ATTRIBUTES]
    col_map = _validate_sheet(ws, _SHEET_ATTRIBUTES)

    # Строим карту code_name -> code_id
    code_map: dict[str, int] = {}
    for row in conn.execute("SELECT id, name FROM code WHERE codebook_version_id = ?", (version_id,)):
        code_map[row[1]] = row[0]

    n = 0
    for row in ws.iter_rows(min_row=2, values_only=False):
        code_name = _row_value(row, col_map, "Код")
        attr_name = _row_value(row, col_map, "Имя атрибута")
        attr_value = _row_value(row, col_map, "Значение")
        if not code_name or not attr_name:
            continue
        code_id = code_map.get(code_name)
        if code_id is None:
            logger.warning("Атрибут '%s' ссылается на неизвестный код '%s'", attr_name, code_name)
            continue
        conn.execute(
            "INSERT OR IGNORE INTO attribute (code_id, name, value) VALUES (?, ?, ?)",
            (code_id, attr_name, attr_value),
        )
        n += 1

    return n


def _count_records(conn: sqlite3.Connection, version_id: int) -> tuple[int, int, int]:
    """Посчитать записи для версии."""
    n_codes = conn.execute("SELECT COUNT(*) FROM code WHERE codebook_version_id = ?", (version_id,)).fetchone()[0]
    n_questions = conn.execute("SELECT COUNT(*) FROM question WHERE codebook_version_id = ?", (version_id,)).fetchone()[
        0
    ]
    n_attributes = conn.execute(
        "SELECT COUNT(*) FROM attribute a JOIN code c ON a.code_id = c.id WHERE c.codebook_version_id = ?",
        (version_id,),
    ).fetchone()[0]
    return int(n_codes), int(n_questions), int(n_attributes)
