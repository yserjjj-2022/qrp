"""Ворота 2: выборочная проверка качества L2 (M8).

Выборка — 10–15 случайных реплик респондента с содержимым; L1 и L2 стоят
рядом, решение оператора одно из трёх: «принято», «исправлено», «отклонено».
Метрика — доля принятых без правок; она показывает читаемость L2, а не
точность относительно эталона: `golden_sample` пуст, и ручного эталона L2 нет
(SPEC-M8, решение 3).

Маркер «выборка просмотрена» — файл `.done` рядом с выгрузкой, содержащий хеш
самой выгрузки. После новой выгрузки хеш меняется, и старый маркер перестаёт
действовать: просмотренной считается именно эта выборка.
"""

from __future__ import annotations

import hashlib
import logging
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from openpyxl import Workbook, load_workbook

from qa.db.pipeline_repo import PipelineRepository, QualitySampleRow

logger = logging.getLogger(__name__)

COLUMNS = ["id_реплики", "тайм-код", "текст_L1", "текст_L2", "asr_quality", "решение", "комментарий"]
UTTERANCE_COLUMN = "id_реплики"
DECISION_COLUMN = "решение"
COMMENT_COLUMN = "комментарий"

DECISIONS: dict[str, str] = {
    "принято": "approved",
    "исправлено": "edited",
    "отклонено": "rejected",
}
SAMPLE_TITLE = "Выборка качества L2"
INSTRUCTION = (
    "Сверьте текст L2 с L1. Решение: принято / исправлено / отклонено. "
    "Пустая ячейка означает, что решения нет. Метрика — доля принятых без правок."
)
DONE_SUFFIX = ".done"


@dataclass(frozen=True)
class QualitySampleResult:
    """Итог выгрузки выборки качества."""

    path: Path
    n_rows: int


@dataclass(frozen=True)
class QualityImportResult:
    """Итог импорта решений выборки качества."""

    path: Path
    approved: int
    edited: int
    rejected: int
    skipped: int

    @property
    def decided(self) -> int:
        """Число решений, учтённых в метрике."""
        return self.approved + self.edited + self.rejected

    @property
    def approved_share(self) -> float | None:
        """Доля принятых без правок."""
        if self.decided == 0:
            return None
        return self.approved / self.decided


def quality_sample_path(exports_dir: Path, interview_code: str) -> Path:
    """Путь к файлу выборки качества интервью."""
    return Path(exports_dir) / f"quality_sample_{interview_code}.xlsx"


def marker_path(sample_path: Path) -> Path:
    """Путь к файлу-маркеру «выборка просмотрена»."""
    return sample_path.with_name(sample_path.name + DONE_SUFFIX)


def file_hash(path: Path) -> str:
    """SHA-256 файла."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_marker(sample_path: Path) -> Path:
    """Записать маркер с хешем выгрузки; вернуть путь маркера."""
    path = marker_path(sample_path)
    path.write_text(file_hash(sample_path) + "\n", encoding="utf-8")
    return path


def marker_matches(sample_path: Path) -> bool:
    """Действует ли маркер: файл есть и его хеш совпадает с выгрузкой.

    Маркер без совпадающего хеша не действует: выборку перевыгрузили, и
    просмотр относился к прошлому файлу.
    """
    path = marker_path(sample_path)
    if not path.is_file() or not sample_path.is_file():
        return False
    try:
        stored = path.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return bool(stored) and stored == file_hash(sample_path)


def sample_rows(
    conn: sqlite3.Connection,
    interview_code: str,
    *,
    min_rows: int = 10,
    max_rows: int = 15,
    seed: int = 0,
) -> list[QualitySampleRow]:
    """Выбрать случайные реплики для проверки.

    Raises:
        ValueError: интервью нет в БД или у него нет L2.
    """
    repo = PipelineRepository(conn)
    interview_db_id = repo.interview_db_id(interview_code)
    if interview_db_id is None:
        msg = f"Интервью {interview_code} не найдено в БД"
        raise ValueError(msg)
    version = repo.max_l2_version(interview_db_id)
    if version <= 0:
        msg = f"У интервью {interview_code} нет слоя L2: сначала выполните normalize"
        raise ValueError(msg)
    rows = repo.quality_sample_rows(interview_db_id, version)
    if not rows:
        msg = f"У интервью {interview_code} нет реплик респондента с L2"
        raise ValueError(msg)
    if len(rows) <= min_rows:
        return list(rows)
    take = min(max_rows, max(min_rows, len(rows)))
    return sorted(random.Random(seed).sample(rows, take), key=lambda item: item.utterance_id)


def export_quality_sample(
    conn: sqlite3.Connection,
    interview_code: str,
    output_path: Path,
    *,
    min_rows: int = 10,
    max_rows: int = 15,
    seed: int = 0,
) -> QualitySampleResult:
    """Записать xlsx с выборкой качества L2.

    Raises:
        ValueError: выборка невозможна (нет интервью или L2).
    """
    from qa.pipeline.terms_table import format_timecode

    rows = sample_rows(conn, interview_code, min_rows=min_rows, max_rows=max_rows, seed=seed)
    wb = Workbook()
    ws = wb.active
    if ws is None:
        msg = "Не удалось создать лист для выборки качества"
        raise RuntimeError(msg)
    ws.title = SAMPLE_TITLE
    ws.append([INSTRUCTION])
    ws.append([])
    ws.append(COLUMNS)
    for row in rows:
        ws.append(
            [
                row.utterance_id,
                format_timecode(row.timecode),
                row.text_l1,
                row.text_l2,
                row.asr_quality,
                None,
                None,
            ]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    logger.info("Выборка качества %s: строк=%d -> %s", interview_code, len(rows), output_path)
    return QualitySampleResult(path=output_path, n_rows=len(rows))


def _find_header_row(ws: object) -> tuple[int, dict[str, int]]:
    """Найти строку шапки и индексы колонок."""
    for row_index, row in enumerate(ws.iter_rows(values_only=True), start=1):  # type: ignore[attr-defined]
        headers = [str(value).strip() if value is not None else "" for value in row]
        if UTTERANCE_COLUMN in headers and DECISION_COLUMN in headers:
            return row_index, {name: headers.index(name) for name in headers if name}
        if row_index > 20:
            break
    msg = f"В файле выборки качества нет шапки с колонками {UTTERANCE_COLUMN!r} и {DECISION_COLUMN!r}"
    raise ValueError(msg)


def import_quality_sample(conn: sqlite3.Connection, path: Path) -> QualityImportResult:
    """Прочитать решения выборки и посчитать метрику.

    Решения не пишутся в БД: схема в M8 не меняется, а метрика нужна на один
    просмотр. После импорта ставится маркер, привязанный к хешу файла.

    Raises:
        FileNotFoundError: файл не найден.
        ValueError: нет шапки или неизвестное решение.
    """
    if not path.exists():
        msg = f"Файл выборки качества не найден: {path}"
        raise FileNotFoundError(msg)
    wb = load_workbook(path, read_only=True, data_only=True)
    approved = edited = rejected = skipped = 0
    try:
        ws = wb[wb.sheetnames[0]]
        header_row, index = _find_header_row(ws)
        for row in ws.iter_rows(min_row=header_row + 1, values_only=True):
            if row is None or all(value is None for value in row):
                continue
            position = index.get(UTTERANCE_COLUMN)
            if position is None or position >= len(row) or row[position] is None:
                continue
            raw = _cell(row, index, DECISION_COLUMN)
            if not raw:
                skipped += 1
                continue
            decision = DECISIONS.get(raw.casefold())
            if decision is None:
                allowed = ", ".join(DECISIONS)
                msg = f"Неизвестное решение {raw!r}: допустимы {allowed}"
                raise ValueError(msg)
            if decision == "approved":
                approved += 1
            elif decision == "edited":
                edited += 1
            else:
                rejected += 1
    finally:
        wb.close()
    write_marker(path)
    logger.info(
        "Решения выборки качества %s: принято=%d, исправлено=%d, отклонено=%d, без решения=%d",
        path.name,
        approved,
        edited,
        rejected,
        skipped,
    )
    return QualityImportResult(
        path=path,
        approved=approved,
        edited=edited,
        rejected=rejected,
        skipped=skipped,
    )


def _cell(row: tuple[object, ...], index: dict[str, int], name: str) -> str | None:
    """Значение ячейки по имени колонки."""
    position = index.get(name)
    if position is None or position >= len(row) or row[position] is None:
        return None
    value = str(row[position]).strip()
    return value or None


__all__ = [
    "DECISIONS",
    "QualityImportResult",
    "QualitySampleResult",
    "export_quality_sample",
    "file_hash",
    "import_quality_sample",
    "marker_matches",
    "marker_path",
    "quality_sample_path",
    "sample_rows",
    "write_marker",
]
