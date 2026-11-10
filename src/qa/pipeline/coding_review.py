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

from openpyxl import Workbook, load_workbook

from qa.db.workbook_repo import CodingRepository, InterviewRepository

logger = logging.getLogger(__name__)

COLUMNS = [
    "id_аннотации",
    "код_респондента",
    "вопрос",
    "фрагмент_L2",
    "реплика_интервьюера_перед_ответом",
    "код",
    "определение_кода",
    "цитата",
    "ПОРЯДОК",
    "НАДЁЖНОСТЬ_КОДА",
    "интерпретация",
    "только_на_интерпретации",
    "описание_OTH",
    "причина_проверки",
    "решение",
    "комментарий",
]

# Значения колонки «решение» в заполненном файле и соответствующие им
# `review.decision` (SPEC-M2, «Ручная проверка»).
DECISIONS: dict[str, str] = {
    "принято": "approved",
    "исправлено": "needs_revision",
    "отклонено": "rejected",
}
DECISION_COLUMN = "решение"
COMMENT_COLUMN = "комментарий"
ANNOTATION_COLUMN = "id_аннотации"


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
                row.annotation_id,
                row.respondent_code,
                row.question_identifier,
                row.fragment_text,
                row.interviewer_context,
                row.code_name,
                row.code_definition,
                row.quote,
                row.order_kind,
                row.reliability,
                row.interpretation,
                int(row.interpretation_only),
                row.oth_description,
                reason,
                None,
                None,
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


@dataclass(frozen=True)
class CodingReviewImportResult:
    """Итог импорта решений ручной проверки."""

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
        """Доля принятых без правок: принято / (принято + исправлено + отклонено)."""
        if self.decided == 0:
            return None
        return self.approved / self.decided


def _read_decision_rows(path: Path) -> list[tuple[int, str, str | None]]:
    """Прочитать заполненный файл: (id аннотации, решение, комментарий)."""
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb[wb.sheetnames[0]]
        headers = [str(cell.value).strip() if cell.value is not None else "" for cell in ws[1]]
        missing = [name for name in (ANNOTATION_COLUMN, DECISION_COLUMN) if name not in headers]
        if missing:
            msg = f"В файле ручной проверки нет колонки {', '.join(missing)}. Заголовки: {headers}"
            raise ValueError(msg)
        index = {name: headers.index(name) for name in headers if name}
        rows: list[tuple[int, str, str | None]] = []
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row is None or all(value is None for value in row):
                continue
            raw_id = row[index[ANNOTATION_COLUMN]] if index[ANNOTATION_COLUMN] < len(row) else None
            raw_decision = row[index[DECISION_COLUMN]] if index[DECISION_COLUMN] < len(row) else None
            if raw_id is None:
                continue
            try:
                annotation_id = int(str(raw_id).strip())
            except ValueError as exc:
                msg = f"Некорректный id аннотации в файле ручной проверки: {raw_id!r}"
                raise ValueError(msg) from exc
            comment_index = index.get(COMMENT_COLUMN)
            raw_comment = row[comment_index] if comment_index is not None and comment_index < len(row) else None
            comment = None if raw_comment is None else str(raw_comment).strip() or None
            decision = "" if raw_decision is None else str(raw_decision).strip().casefold()
            rows.append((annotation_id, decision, comment))
        return rows
    finally:
        wb.close()


def import_coding_review(conn: sqlite3.Connection, path: Path) -> CodingReviewImportResult:
    """Записать решения ручной проверки в `review` и посчитать метрику пилота.

    Пустое решение пропускается: строка остаётся неразобранной, а не получает
    ложное «принято». Неизвестное значение — ошибка файла, а не тихий пропуск.

    Raises:
        FileNotFoundError: файл не найден.
        ValueError: нет обязательных колонок или неизвестное решение.
    """
    if not path.exists():
        msg = f"Файл ручной проверки не найден: {path}"
        raise FileNotFoundError(msg)
    rows = _read_decision_rows(path)
    repository = CodingRepository(conn)
    approved = edited = rejected = skipped = 0
    for annotation_id, raw_decision, comment in rows:
        if not raw_decision:
            skipped += 1
            continue
        decision = DECISIONS.get(raw_decision)
        if decision is None:
            allowed = ", ".join(DECISIONS)
            msg = f"Неизвестное решение {raw_decision!r}: допустимы {allowed}"
            raise ValueError(msg)
        repository.insert_review(annotation_id, decision, comment)
        if decision == "approved":
            approved += 1
        elif decision == "needs_revision":
            edited += 1
        else:
            rejected += 1
    conn.commit()
    result = CodingReviewImportResult(
        path=path,
        approved=approved,
        edited=edited,
        rejected=rejected,
        skipped=skipped,
    )
    logger.info(
        "Решения ручной проверки %s: принято=%d, исправлено=%d, отклонено=%d, без решения=%d, доля принятых=%s",
        path.name,
        approved,
        edited,
        rejected,
        skipped,
        "—" if result.approved_share is None else f"{result.approved_share:.3f}",
    )
    return result
