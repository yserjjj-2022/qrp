"""Стадия normalize: нормализация L1 -> L2 для интервью.

Собирает входные данные из БД (реплики, лексикон, маркеры), вызывает проход
модели по репликам респондента и записывает результат одной транзакцией.
Реплики интервьюера, наблюдателя и бэкканалы получают только детерминированный
результат.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field

from qa.db.workbook_repo import (
    InterviewRepository,
    LexiconRepository,
    MarkerRepository,
    NewTermRepository,
    RunRepository,
    TextUnitRepository,
)
from qa.llm.base import LlmProvider
from qa.pipeline.normalize_l2 import (
    NormalizeOutcome,
    NormalizeStats,
    estimate_dry_run,
    load_examples,
    normalize_utterance,
    prompt_hash,
    should_use_model,
    summarize,
    word_count,
)
from qa.pipeline.normalize_rules import LexiconEntry
from qa.settings import Settings

logger = logging.getLogger(__name__)


class NormalizeError(Exception):
    """Нормализация невозможна: нет интервью или реплик."""


@dataclass
class NormalizeResult:
    """Итог стадии нормализации."""

    interview_code: str
    interview_db_id: int
    version: int
    stats: NormalizeStats
    skipped: bool = False
    run_id: int | None = None
    model_used: str = ""
    outcomes: list[NormalizeOutcome] = field(default_factory=list)


@dataclass(frozen=True)
class DryRunReport:
    """Оценка объёма работы без вызовов модели."""

    interview_code: str
    utterances: int
    model_calls: int
    prompt_tokens: int
    already_done: bool


def load_lexicon(conn: sqlite3.Connection) -> list[LexiconEntry]:
    """Термины лексикона со статусом не rejected."""
    terms = LexiconRepository(conn).all_terms()
    entries: list[LexiconEntry] = []
    for canonical, data in terms.items():
        if data.get("status") == "rejected":
            continue
        variants = tuple(str(item) for item in data.get("variants") or [])
        entries.append(LexiconEntry(canonical=canonical, variants=variants))
    return entries


def load_markers(conn: sqlite3.Connection) -> list[str]:
    """Маркеры неуверенности из БД."""
    return MarkerRepository(conn).expressions()


def _previous_question(utterances: list[tuple[int, str, str, int]], index: int) -> str | None:
    """Ближайшая предыдущая реплика интервьюера."""
    for position in range(index - 1, -1, -1):
        role = utterances[position][1]
        if role == "INTERVIEWER":
            return utterances[position][2]
    return None


def plan_dry_run(
    conn: sqlite3.Connection,
    interview_code: str,
    settings: Settings,
    version: int = 1,
) -> DryRunReport:
    """Посчитать объём работы без обращения к модели."""
    interview_db_id = InterviewRepository(conn).interview_db_id_by_code(interview_code)
    if interview_db_id is None:
        msg = f"Интервью {interview_code} не найдено в БД: сначала загрузите транскрипт"
        raise NormalizeError(msg)
    utterances = InterviewRepository(conn).utterances_for_normalize(interview_db_id)
    if not utterances:
        msg = f"У интервью {interview_code} нет реплик: сначала загрузите транскрипт"
        raise NormalizeError(msg)

    min_words = settings.m1b.min_words_for_model
    texts = [text for _uid, role, text, flag in utterances if should_use_model(role, bool(flag), text, min_words)]
    estimate = estimate_dry_run(texts)
    done = TextUnitRepository(conn).count_for_interview(interview_db_id, version) > 0
    return DryRunReport(
        interview_code=interview_code,
        utterances=len(utterances),
        model_calls=estimate["calls"],
        prompt_tokens=estimate["prompt_tokens"],
        already_done=done,
    )


def normalize_interview(
    conn: sqlite3.Connection,
    interview_code: str,
    settings: Settings,
    provider: LlmProvider | None,
    *,
    version: int = 1,
    code_version: str = "",
) -> NormalizeResult:
    """Нормализовать все реплики интервью и записать L2.

    Args:
        conn: соединение с БД.
        interview_code: внешний код интервью (R01).
        settings: настройки проекта.
        provider: провайдер LLM; None — только детерминированный проход.
        version: версия L2.
        code_version: версия кода для записи в run.
    """
    interviews = InterviewRepository(conn)
    interview_db_id = interviews.interview_db_id_by_code(interview_code)
    if interview_db_id is None:
        msg = f"Интервью {interview_code} не найдено в БД: сначала загрузите транскрипт"
        raise NormalizeError(msg)

    utterances = interviews.utterances_for_normalize(interview_db_id)
    if not utterances:
        msg = f"У интервью {interview_code} нет реплик: сначала загрузите транскрипт"
        raise NormalizeError(msg)

    text_units = TextUnitRepository(conn)
    if text_units.count_for_interview(interview_db_id, version) > 0:
        logger.info("L2 версии %d для %s уже записан, пропуск", version, interview_code)
        return NormalizeResult(
            interview_code=interview_code,
            interview_db_id=interview_db_id,
            version=version,
            stats=NormalizeStats(),
            skipped=True,
        )

    lexicon = load_lexicon(conn)
    markers = load_markers(conn)
    examples = load_examples()
    min_words = settings.m1b.min_words_for_model

    outcomes: list[NormalizeOutcome] = []
    for index, (utterance_id, role, text, flag) in enumerate(utterances):
        outcome = normalize_utterance(
            utterance_id,
            role,
            bool(flag),
            text,
            previous_question=_previous_question(utterances, index),
            lexicon=lexicon,
            markers=markers,
            examples=examples,
            provider=provider,
            min_words=min_words,
        )
        outcomes.append(outcome)

    stats = summarize(outcomes)
    model_used = _model_label(provider, settings, outcomes)
    runs = RunRepository(conn)
    prompt_file = load_prompt_text()
    params = {
        "version": version,
        "min_words_for_model": min_words,
        "temperature": 0,
        "lexicon_terms": len(lexicon),
        "markers": len(markers),
        "model_calls": stats.model_calls,
    }

    conn.execute("SAVEPOINT normalize_l2")
    try:
        run_id = runs.start(
            model=model_used,
            prompt_hash=prompt_hash(prompt_file),
            codebook_version_id=runs.latest_codebook_version_id(),
            code_version=code_version,
            params_json=json.dumps(params, ensure_ascii=False),
        )
        new_terms = NewTermRepository(conn)
        for outcome in outcomes:
            text_units.insert(
                utterance_id=outcome.utterance_id,
                version=version,
                text=outcome.text,
                status=outcome.status,
                asr_quality=outcome.asr_quality,
                changes_json=json.dumps(outcome.changes, ensure_ascii=False),
                uncertain_spans_json=json.dumps(outcome.uncertain_spans, ensure_ascii=False),
                fillers_removed=outcome.fillers_removed,
                stutter_removed=outcome.stutter_removed,
                repeat_unclear=outcome.repeat_unclear,
                run_id=run_id,
            )
            for term in outcome.new_terms:
                new_terms.add(term, interview_db_id, outcome.utterance_id)
    except BaseException:
        conn.execute("ROLLBACK TO SAVEPOINT normalize_l2")
        conn.execute("RELEASE SAVEPOINT normalize_l2")
        raise
    conn.execute("RELEASE SAVEPOINT normalize_l2")
    conn.commit()

    logger.info(
        "L2 %s: единиц=%d, proposed=%d, rejected=%d, вызовов=%d",
        interview_code,
        stats.total,
        stats.proposed,
        stats.rejected,
        stats.model_calls,
    )
    return NormalizeResult(
        interview_code=interview_code,
        interview_db_id=interview_db_id,
        version=version,
        stats=stats,
        run_id=run_id,
        model_used=model_used,
        outcomes=outcomes,
    )


def load_prompt_text() -> str:
    """Текст основного промпта (для хеша в run)."""
    from qa.pipeline.normalize_l2 import load_prompt

    return load_prompt()


def _model_label(
    provider: LlmProvider | None,
    settings: Settings,
    outcomes: list[NormalizeOutcome],
) -> str:
    """Модель для записи в run: фактическая из ответа API, иначе запрошенная."""
    if provider is None:
        return "deterministic"
    for outcome in outcomes:
        if outcome.model_used:
            return outcome.model_used
    return settings.llm.model_normalize or settings.llm.model or ""


def count_words(text: str) -> int:
    """Число слов в реплике (для отчёта)."""
    return word_count(text)


__all__ = [
    "DryRunReport",
    "NormalizeError",
    "NormalizeResult",
    "count_words",
    "load_lexicon",
    "load_markers",
    "normalize_interview",
    "plan_dry_run",
]
