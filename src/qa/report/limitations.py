"""Ограничения отчёта, зависящие от состояния конвейера (M8).

Базовые ограничения M5 считаются по покрытию (`report/model.py::_limitations`).
Здесь — те, которые видны только по состоянию конвейера: неразобранные
кандидаты терминов, неприсмотренная выборка качества, применённые разовые
правки, пустая таблица имён. Текст по-прежнему детерминированный: числа
подставляются из репозитория, формулировки не пишет модель.

SQL в этом модуле нет: состояние приходит из `PipelineRepository`
(CONSTITUTION §4).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from qa.db.pipeline_repo import PipelineRepository
from qa.pipeline.orchestrate import warn_names_map
from qa.pipeline.quality_sample import marker_matches, quality_sample_path
from qa.report.model import Limitation
from qa.settings import Settings


def pipeline_limitations(
    conn: sqlite3.Connection,
    interview_code: str,
    settings: Settings,
) -> tuple[Limitation, ...]:
    """Ограничения отчёта по фактическому состоянию конвейера.

    Пустой список — не «ограничений нет», а «состояние конвейера их не
    добавило»: базовые ограничения M5 остаются в модели отчёта.

    Args:
        conn: соединение с БД (только чтение через репозиторий).
        interview_code: код интервью.
        settings: настройки проекта (пути выгрузок, секция `m8`).

    Returns:
        Ограничения в порядке значимости для читателя.
    """
    repo = PipelineRepository(conn)
    interview_db_id = repo.interview_db_id(interview_code)
    if interview_db_id is None:
        return ()

    items: list[Limitation] = []

    proposed = repo.proposed_term_count(interview_db_id)
    if proposed:
        items.append(
            Limitation(
                code="terms_pending",
                text=(
                    f"Кандидатов терминов без решения: {proposed}. "
                    "Правки глоссария в L2 не попали: нормализация прошла до ревью терминов."
                ),
            )
        )

    edits = repo.latest_manual_edits(interview_db_id)
    if edits is not None and edits.count:
        source = f", файл {edits.file_hash[:12]}" if edits.file_hash else ""
        items.append(
            Limitation(
                code="manual_edits",
                text=(
                    f"В L2 применено разовых правок: {edits.count}{source}. "
                    "Исходные реплики L1 не менялись, поэтому в отчёте текст может отличаться от записи."
                ),
            )
        )

    if settings.m8.quality_sample_enabled:
        sample = quality_sample_path(Path(settings.paths.exports), interview_code)
        if not marker_matches(sample):
            items.append(
                Limitation(
                    code="quality_sample",
                    text=(
                        "Выборка качества L2 не просмотрена: читаемость слоя L2 на этом интервью "
                        "не подтверждена выборочной проверкой."
                    ),
                )
            )

    names_map = warn_names_map()
    if names_map is not None:
        items.append(
            Limitation(
                code="names_map",
                text=(
                    "Таблица имён не задана: в цитатах остаются имена и названия организаций. "
                    "Перед передачей отчёта вне проекта текст нужно просмотреть вручную."
                ),
            )
        )

    return tuple(items)


__all__ = ["pipeline_limitations"]
