"""Золотой набор: экспорт реплик L1 в xlsx и импорт заполненного файла.

Экспорт выбирает реплики респондента из БД и пишет их текст L1 в
data/golden/golden_sample.xlsx с пустой колонкой «текст_L2_вручную».
Импорт читает заполненный файл или лист «Золотой_набор» книги.
"""

from __future__ import annotations

import logging
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from openpyxl import Workbook, load_workbook

from qa.db.workbook_repo import GoldenRepository

logger = logging.getLogger(__name__)

COLUMNS = ["id_интервью", "ссылка_на_реплику", "текст_L1", "текст_L2_вручную"]
GOLDEN_FILENAME = "golden_sample.xlsx"


@dataclass
class ExportResult:
    """Итог экспорта золотого набора."""

    path: Path
    n_rows: int


@dataclass
class ImportResult:
    """Итог импорта золотого набора."""

    n_rows: int
    n_filled: int


def _candidate_rows(
    conn: sqlite3.Connection,
    n: int,
    interview_id: str | None,
    seed: int,
) -> list[tuple[str, int, str]]:
    """Выбрать реплики респондента: (код интервью, id реплики, текст L1)."""
    sql = (
        "SELECT i.id_external, u.id, u.text FROM utterance u"
        " JOIN interview i ON u.interview_id = i.id"
        " WHERE u.speaker_role = 'RESPONDENT' AND u.is_backchannel = 0"
    )
    params: list[object] = []
    if interview_id is not None:
        sql += " AND i.id_external = ?"
        params.append(interview_id)
    sql += " ORDER BY i.id, u.id"
    rows = conn.execute(sql, params).fetchall()
    candidates = [(str(row[0]), int(row[1]), str(row[2])) for row in rows]
    if n >= len(candidates):
        return candidates
    rng = random.Random(seed)
    return sorted(rng.sample(candidates, n), key=lambda item: (item[0], item[1]))


def export_golden_sample(
    conn: sqlite3.Connection,
    output_path: Path,
    n: int = 40,
    interview_id: str | None = None,
    seed: int = 0,
) -> ExportResult:
    """Выбрать реплики и записать xlsx с пустой колонкой ручного L2.

    Raises:
        ValueError: нет реплик для выборки.
    """
    rows = _candidate_rows(conn, n, interview_id, seed)
    if not rows:
        msg = "В БД нет реплик респондента для золотого набора: сначала загрузите интервью"
        raise ValueError(msg)

    wb = Workbook()
    ws = wb.active
    if ws is None:
        msg = "Не удалось создать лист для золотого набора"
        raise RuntimeError(msg)
    ws.title = "Золотой_набор"
    ws.append(COLUMNS)
    for interview_code, utterance_id, text in rows:
        ws.append([interview_code, f"utterance:{utterance_id}", text, None])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    logger.info("Золотой набор выгружен: %s строк -> %s", len(rows), output_path)
    return ExportResult(path=output_path, n_rows=len(rows))


def _cell(row: tuple[object, ...], index: dict[str, int], name: str) -> str | None:
    """Значение ячейки строки по имени колонки."""
    idx = index.get(name)
    if idx is None or idx >= len(row) or row[idx] is None:
        return None
    return str(row[idx]).strip()


def _read_rows(path: Path) -> list[tuple[str | None, str | None, str, str | None]]:
    """Прочитать строки xlsx золотого набора."""
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb[wb.sheetnames[0]]
        headers = [str(cell.value).strip() if cell.value is not None else "" for cell in ws[1]]
        index = {name: headers.index(name) for name in COLUMNS if name in headers}
        if "текст_L1" not in index:
            msg = f"В файле золотого набора нет колонки 'текст_L1'. Заголовки: {headers}"
            raise ValueError(msg)
        rows: list[tuple[str | None, str | None, str, str | None]] = []
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row is None or all(value is None for value in row):
                continue
            text_l1 = _cell(row, index, "текст_L1")
            if text_l1 is None:
                continue
            rows.append(
                (
                    _cell(row, index, "id_интервью"),
                    _cell(row, index, "ссылка_на_реплику"),
                    text_l1,
                    _cell(row, index, "текст_L2_вручную"),
                )
            )
        return rows
    finally:
        wb.close()


def import_golden(conn: sqlite3.Connection, path: Path) -> ImportResult:
    """Загрузить заполненный файл золотого набора в БД.

    Raises:
        FileNotFoundError: файл не найден.
        ValueError: нет обязательной колонки.
    """
    if not path.exists():
        msg = f"Файл золотого набора не найден: {path}"
        raise FileNotFoundError(msg)
    rows = _read_rows(path)
    repo = GoldenRepository(conn)
    n_filled = 0
    for interview_code, utterance_ref, text_l1, text_l2 in rows:
        repo.upsert(interview_code, utterance_ref, text_l1, text_l2, "import")
        n_filled += int(text_l2 is not None)
    conn.commit()
    logger.info("Золотой набор импортирован: %s строк, заполнено L2: %s", len(rows), n_filled)
    return ImportResult(n_rows=len(rows), n_filled=n_filled)
