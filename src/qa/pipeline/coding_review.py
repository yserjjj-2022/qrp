"""Выгрузка очереди ручной проверки кодирования (M2).

Очередь = выборочная доля аннотаций + **все** аннотации с `OTH`, все аннотации
с `interpretation_only = 1` или `reliability = 1`, все аннотации `doubtful`
фрагментов с непустой интерпретацией и все аннотации фрагментов с `pii_flag`.
Обязательные случаи не проверяются выборочно: `OTH` — место, где кодбук не
покрывает смысл, ненадёжные — место возможной догадки, `pii_flag` — место
возможной утечки.

Файл xlsx кладётся в `data/exports/` и в репозиторий не коммитится
(CONSTITUTION §5).
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from openpyxl import Workbook

from qa.db.workbook_repo import CodingRepository, InterviewRepository

logger = logging.getLogger(__name__)

COLUMNS = [
    "код_респондента",
    "вопрос",
    "фрагмент_L2",
    "код",
    "цитата",
    "ПОРЯДОК",
    "НАДЁЖНОСТЬ_КОДА",
    "интерпретация",
    "только_на_интерпретации",
    "описание_OTH",
    "причина_проверки",
]


@dataclass(frozen=True)
class CodingReviewResult:
    """Итог выгрузки очереди ручной проверки."""

    path: Path
    required: int
    sampled: int

    @property
    def n_rows(self) -> int:
        """Всего строк в файле."""
        return self.required + self.sampled


def export_coding_review(
    conn: sqlite3.Connection,
    interview_code: str,
    run_id: int,
    output_path: Path,
    *,
    sample_share: float = 0.10,
    seed: int = 0,
) -> CodingReviewResult:
    """Записать xlsx с очередью ручной проверки кодирования.

    Raises:
        ValueError: интервью не найдено или у прогона нет аннотаций.
    """
    interview_db_id = InterviewRepository(conn).interview_db_id_by_code(interview_code)
    if interview_db_id is None:
        msg = f"Интервью {interview_code} не найдено в БД"
        raise ValueError(msg)

    repository = CodingRepository(conn)
    queue = repository.review_queue(interview_db_id, run_id, sample_share, seed=seed)
    rows = repository.review_export_rows(interview_db_id, run_id)
    if not rows:
        msg = f"У прогона {run_id} нет аннотаций: ручная проверка не сформирована"
        raise ValueError(msg)

    reasons = dict(queue)
    wb = Workbook()
    ws = wb.active
    if ws is None:
        msg = "Не удалось создать лист для ручной проверки"
        raise RuntimeError(msg)
    ws.title = "Проверка_кодов"
    ws.append(COLUMNS)
    for row in rows:
        reason = reasons.get(row.annotation_id)
        if reason is None:
            continue
        ws.append(
            [
                row.respondent_code,
                row.question_identifier,
                row.fragment_text,
                row.code_name,
                row.quote,
                row.order_kind,
                row.reliability,
                row.interpretation,
                int(row.interpretation_only),
                row.oth_description,
                reason,
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    required = sum(1 for _ann, reason in queue if reason != "sample")
    sampled = len(queue) - required
    logger.info(
        "Ручная проверка кодирования %s run=%d: обязательных=%d, выборочных=%d -> %s",
        interview_code,
        run_id,
        required,
        sampled,
        output_path,
    )
    return CodingReviewResult(path=output_path, required=required, sampled=sampled)
