"""Чтение рабочей книги проекта: листы, колонки, типы, списки, хеши.

Код только читает книгу. Любая ошибка данных указывает лист и номер строки.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from openpyxl import load_workbook
from openpyxl.workbook.workbook import Workbook
from openpyxl.worksheet.worksheet import Worksheet

from qa.settings import ColumnSpec, SheetSpec, WorkbookSettings

logger = logging.getLogger(__name__)

CellValue = str | int | list[str] | None

# Разделители элементов списка в ячейках: «8, 9, 12Т», «R1; R2», «R1–R8».
_LIST_SPLIT_RE = re.compile(r"[;,\n]+")
# Пометки «пусто» в ячейках списка: считаются отсутствием значений, а не ошибкой.
_EMPTY_MARKERS = frozenset({"—", "–", "-"})
# Диапазон: R1–R8, E0–E7, A1–A8, N1–N4, C1–C4 (дефис, минус, en dash, em dash).
_RANGE_RE = re.compile(r"^([A-Za-zА-Яа-яЁё]+)(\d+)\s*[-–—]\s*([A-Za-zА-Яа-яЁё]+)(\d+)$")
# Свободный текст, который не является идентификатором: «индуктивно», «все вопросы».
_PLAIN_NOTE_RE = re.compile(r"^(индуктивн|оценк|степень|рекоменд|все\b|все$)", re.IGNORECASE)
# Идентификатор кода: R1, 7C, MCP, A3/4, OTH.
_CODE_ID_RE = re.compile(r"^[A-Za-zА-ЯЁа-яё][A-Za-zА-ЯЁа-яё0-9]*(?:/\d+)?$")
# Идентификатор вопроса: 1, 7Т, 14Т-б, 19Т.
_QUESTION_ID_RE = re.compile(r"^\d+[A-Za-zА-ЯЁа-яё]?(?:-[A-Za-zА-ЯЁа-яё])?$")


class WorkbookError(Exception):
    """Ошибка структуры или данных книги (лист, строка)."""


@dataclass
class SheetData:
    """Прочитанный лист: имя, хеши, строки как словари логических полей."""

    name: str
    header_hash: str
    content_hash: str
    rows: list[dict[str, CellValue]] = field(default_factory=list)
    row_numbers: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """Лист без строк данных."""
        return not self.rows


@dataclass
class WorkbookData:
    """Прочитанная книга: версия, хеши, листы."""

    path: Path
    file_hash: str
    codebook_hash: str
    sheets: dict[str, SheetData]
    sheet_hashes: dict[str, str]
    warnings: list[str] = field(default_factory=list)

    def sheet(self, name: str) -> SheetData | None:
        """Лист по имени или None."""
        return self.sheets.get(name)


def file_hash(path: Path) -> str:
    """SHA-256 файла книги."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_text(text: str) -> str:
    """SHA-256 строки (для хешей листов)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_cell(value: Any) -> str | None:
    """Привести значение ячейки к строке: пустое -> None, числа без .0."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).replace("\u00a0", " ").strip()
    return text or None


def parse_list(value: Any) -> list[str]:
    """Разобрать список из ячейки: разделители, диапазоны.

    «8, 9, 12Т» -> ['8', '9', '12Т']; «R1–R8» -> R1..R8; «R1; R2» -> ['R1', 'R2'].
    Разделители внутри скобок не считаются («индуктивно (влияние, доверие)»).
    Элементы, которые не являются идентификатором («индуктивно», «атрибут: стаж»),
    возвращаются как есть: вызывающая сторона решает, ошибка это или заметка.
    """
    text = normalize_cell(value)
    if text is None:
        return []
    items: list[str] = []
    for chunk in _split_top_level(text):
        part = chunk.strip()
        if not part or part in _EMPTY_MARKERS:
            continue
        items.extend(_expand_item(part))
    return items


def _split_top_level(text: str) -> list[str]:
    """Разделить по «;» и «,» вне скобок."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(depth - 1, 0)
        if char in ";,\n" and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    parts.append("".join(current))
    return parts


def _convert_cell(sheet_name: str, column: ColumnSpec, raw: Any, row_number: int) -> CellValue:
    """Привести значение ячейки к типу колонки."""
    if column.type == "list":
        return parse_list(raw)
    if column.type == "int":
        text = normalize_cell(raw)
        if text is None:
            return None
        try:
            return int(text)
        except ValueError:
            msg = (
                f"Лист '{sheet_name}', строка {row_number}, колонка '{column.header}': "
                f"ожидалось целое число, получено {text!r}"
            )
            raise WorkbookError(msg) from None
    return normalize_cell(raw)


def _expand_item(part: str) -> list[str]:
    """Развернуть один элемент списка, если это диапазон вида R1–R8."""
    match = _RANGE_RE.match(part)
    if match is None:
        return [part]
    prefix_a, start_raw, prefix_b, end_raw = match.groups()
    if prefix_a.upper() != prefix_b.upper():
        return [part]
    start, end = int(start_raw), int(end_raw)
    if end < start or end - start > 999:
        return [part]
    width = len(start_part) if (start_part := start_raw).startswith("0") else 0
    return [f"{prefix_a.upper()}{n:0{width}d}" if width else f"{prefix_a.upper()}{n}" for n in range(start, end + 1)]


def is_identifier_like(item: str) -> bool:
    """Похож ли элемент на идентификатор кода или вопроса (R1, 12Т, 14Т-б).

    Служебные пометки («индуктивно», «спонт./подск.») не считаются кодами.
    """
    if _PLAIN_NOTE_RE.match(item):
        return False
    return bool(_CODE_ID_RE.match(item) or _QUESTION_ID_RE.match(item))


def _cell_text(value: object) -> str:
    """Текст ячейки для хеша содержимого листа."""
    if value is None:
        return ""
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return str(value)


class _SheetReader:
    """Чтение одного листа по схеме колонок."""

    def __init__(self, ws: Worksheet, spec: SheetSpec) -> None:
        self._ws = ws
        self._spec = spec
        self._headers = self._read_headers()
        self._columns = self._resolve_columns()
        self.row_numbers: list[int] = []

    def _read_headers(self) -> list[str]:
        """Заголовки первой строки листа."""
        return [str(cell.value).strip() if cell.value is not None else "" for cell in self._ws[1]]

    def _resolve_columns(self) -> dict[str, int]:
        """Индексы колонок по логическим полям; ошибка, если нет обязательной."""
        resolved: dict[str, int] = {}
        missing: list[str] = []
        for field_name, column in self._spec.columns.items():
            index = None
            for i, header in enumerate(self._headers):
                if header == column.header:
                    index = i
                    break
            if index is None:
                if column.required:
                    missing.append(column.header)
                continue
            resolved[field_name] = index
        if missing:
            msg = (
                f"Лист '{self._ws.title}': не найдены обязательные колонки {missing}. "
                f"Доступные колонки: {self._headers}"
            )
            raise WorkbookError(msg)
        return resolved

    def read_rows(self) -> tuple[list[dict[str, CellValue]], list[str]]:
        """Прочитать строки данных и заметки о нераспознанных значениях."""
        rows: list[dict[str, CellValue]] = []
        notes: list[str] = []
        for row_number, row in enumerate(self._ws.iter_rows(min_row=2), start=2):
            if all(cell.value is None for cell in row):
                continue
            record: dict[str, CellValue] = {}
            for field_name, column in self._spec.columns.items():
                index = self._columns.get(field_name)
                if index is None or index >= len(row):
                    record[field_name] = None
                else:
                    record[field_name] = self._convert(column, row[index].value, row_number=row_number)
            if self._is_blank_row(record):
                continue
            rows.append(record)
            self.row_numbers.append(row_number)
            note = self._collect_notes(row_number, record)
            if note:
                notes.append(note)
        return rows, notes

    def _is_blank_row(self, record: dict[str, CellValue]) -> bool:
        """Строка без значений в обязательных полях не является записью."""
        return all(record.get(field) is None for field, column in self._spec.columns.items() if column.required)

    def _convert(self, column: ColumnSpec, raw: Any, row_number: int) -> CellValue:
        """Привести значение ячейки к типу колонки."""
        return _convert_cell(self._ws.title, column, raw, row_number)

    def _collect_notes(self, row_number: int, record: dict[str, Any]) -> str | None:
        """Заметка о нераспознанных элементах списков.

        Raises:
            WorkbookError: в строгой колонке элемент не является идентификатором.
        """
        unknown_parts: list[str] = []
        for field_name, column in self._spec.columns.items():
            if column.type != "list":
                continue
            value = record.get(field_name)
            if not isinstance(value, list):
                continue
            unknown = [item for item in value if not is_identifier_like(item)]
            if not unknown:
                continue
            if column.strict:
                msg = (
                    f"Лист '{self._ws.title}', строка {row_number}, колонка '{column.header}': "
                    f"нераспознанные значения {unknown} (ожидались коды или вопросы)"
                )
                raise WorkbookError(msg)
            unknown_parts.append(f"{column.header}: {', '.join(unknown)}")
        if not unknown_parts:
            return None
        return f"Лист '{self._ws.title}', строка {row_number}: нераспознано — {'; '.join(unknown_parts)}"


def read_sheet(ws: Worksheet, spec: SheetSpec, sheet_name: str) -> SheetData:
    """Прочитать лист по схеме."""
    reader = _SheetReader(ws, spec)
    rows, notes = reader.read_rows()
    header_text = "\n".join(reader._headers)  # noqa: SLF001 - заголовки уже прочитаны
    content_text = "\n".join("	".join(_cell_text(record.get(field)) for field in spec.columns) for record in rows)
    return SheetData(
        name=sheet_name,
        header_hash=hash_text(header_text),
        content_hash=hash_text(content_text),
        rows=rows,
        row_numbers=reader.row_numbers,
        notes=notes,
    )


def read_workbook(path: Path, workbook_settings: WorkbookSettings) -> WorkbookData:
    """Прочитать книгу: проверить обязательные листы, классифицировать остальные.

    Raises:
        FileNotFoundError: файл не найден.
        WorkbookError: нет обязательного листа или обязательной колонки.
    """
    if not path.exists():
        msg = f"Файл книги не найден: {path}"
        raise FileNotFoundError(msg)

    wb: Workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        names = list(wb.sheetnames)
        missing = [name for name in workbook_settings.required_sheets if name not in names]
        if missing:
            msg = f"В книге отсутствуют обязательные листы {missing}. Доступные листы: {names}"
            raise WorkbookError(msg)

        sheets: dict[str, SheetData] = {}
        warnings: list[str] = []
        for name in names:
            sheet_class = workbook_settings.sheet_class(name)
            if sheet_class == "ignored":
                continue
            spec = workbook_settings.spec_for(name)
            if spec is None:
                if sheet_class == "unknown":
                    message = f"Неизвестный лист '{name}': предупреждение, лист пропущен"
                    logger.warning(message)
                    warnings.append(message)
                continue
            if sheet_class == "unknown":
                message = f"Неизвестный лист '{name}' имеет схему колонок и загружается как необязательный"
                logger.warning(message)
                warnings.append(message)
            sheet_data = read_sheet(wb[name], spec, name)
            sheets[name] = sheet_data
            warnings.extend(sheet_data.notes)

        codebook_hashes = [sheets[name].content_hash for name in workbook_settings.codebook_sheets if name in sheets]
        return WorkbookData(
            path=path,
            file_hash=file_hash(path),
            codebook_hash=hash_text("\n".join(codebook_hashes)),
            sheets=sheets,
            sheet_hashes={name: data.content_hash for name, data in sheets.items()},
            warnings=warnings,
        )
    finally:
        wb.close()


def sheet_kind(name: str, workbook_settings: WorkbookSettings) -> Literal["required", "optional", "ignored", "unknown"]:
    """Класс листа (публичная обёртка для отчётов)."""
    return workbook_settings.sheet_class(name)
