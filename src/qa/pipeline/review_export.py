"""Выгрузка единиц L2 для ручной проверки оператором.

В выборку попадают все единицы с флагами (`names_review`, `asr_fix`,
`doubtful`) и случайная доля остальных: оператор сравнивает L1 и L2 рядом.
Файл xlsx кладётся в `data/exports/` и в репозиторий не коммитится.
"""

from __future__ import annotations

import json
import logging
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from openpyxl import Workbook

logger = logging.getLogger(__name__)

COLUMNS = [
    "id_реплики",
    "роль",
    "флаг",
    "L1",
    "L2",
    "статус",
    "asr_quality",
    "changes",
]


@dataclass(frozen=True)
class ReviewExportResult:
    """Итог выгрузки для ручной проверки."""

    path: Path
    flagged: int
    sampled: int

    @property
    def n_rows(self) -> int:
        """Всего строк в файле."""
        return self.flagged + self.sampled


def _flags(changes: list[dict[str, object]], asr_quality: str) -> list[str]:
    """Почему единица попала в обязательную выборку."""
    found: list[str] = []
    for change in changes:
        kind = change.get("type")
        if kind == "names_review":
            found.append("names_review")
        elif kind == "asr_fix":
            found.append("asr_fix")
    if asr_quality == "doubtful":
        found.append("doubtful")
    if asr_quality == "garbled":
        found.append("garbled")
    return sorted(set(found))


def export_manual_review(
    conn: sqlite3.Connection,
    interview_code: str,
    version: int,
    output_path: Path,
    n_random: int = 15,
    seed: int = 0,
) -> ReviewExportResult:
    """Записать xlsx со всеми флагованными единицами и случайной выборкой.

    Raises:
        ValueError: интервью или версия L2 не найдены.
    """
    rows = conn.execute(
        """
        SELECT u.id, u.speaker_role, u.text, t.text, t.status, t.asr_quality, t.changes_json
        FROM text_unit t
        JOIN utterance u ON t.utterance_id = u.id
        JOIN interview i ON u.interview_id = i.id
        WHERE i.id_external = ? AND t.version = ?
        ORDER BY u.id
        """,
        (interview_code, version),
    ).fetchall()
    if not rows:
        msg = f"Нет единиц L2 версии {version} для интервью {interview_code}"
        raise ValueError(msg)

    flagged: list[tuple[int, str, str, str, str, str, str, str]] = []
    rest: list[tuple[int, str, str, str, str, str, str, str]] = []
    for utterance_id, role, text_l1, text_l2, status, quality, changes_json in rows:
        changes = json.loads(changes_json or "[]")
        flags = _flags(changes, str(quality))
        item = (
            int(utterance_id),
            str(role),
            str(flags and ", ".join(flags)),
            str(text_l1),
            str(text_l2),
            str(status),
            str(quality),
            str(changes_json or "[]"),
        )
        (flagged if flags else rest).append(item)

    sampled = rest
    if n_random < len(rest):
        sampled = sorted(random.Random(seed).sample(rest, n_random), key=lambda item: item[0])

    wb = Workbook()
    ws = wb.active
    if ws is None:
        msg = "Не удалось создать лист для ручной проверки"
        raise RuntimeError(msg)
    ws.title = "Проверка_L2"
    ws.append(COLUMNS)
    for item in [*flagged, *sampled]:
        ws.append(list(item))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    logger.info(
        "Ручная проверка L2 %s v%d: флагов=%d, случайных=%d -> %s",
        interview_code,
        version,
        len(flagged),
        len(sampled),
        output_path,
    )
    return ReviewExportResult(path=output_path, flagged=len(flagged), sampled=len(sampled))
