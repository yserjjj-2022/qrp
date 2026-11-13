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
from pathlib import Path

from qa.db.workbook_repo import (
    InterviewRepository,
    LexiconRepository,
    MarkerRepository,
    NewTermRepository,
    RunRepository,
    TextUnitRepository,
)
from qa.llm.base import LlmProvider
from qa.pipeline.manual_edits import ManualEdits, load_manual_edits, manual_edits_path
from qa.pipeline.normalize_l2 import (
    NormalizeOutcome,
    NormalizeStats,
    estimate_dry_run,
    load_examples,
    normalize_utterance,
    prompt_hash,
    should_use_model,
    summarize,
    surviving_new_terms,
    word_count,
)
from qa.pipeline.normalize_rules import (
    LexiconEntry,
    lexicon_snapshot_hash,
    normalize_deterministic,
    suspect_frequencies,
)
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
    # Снимок лексикона, на котором записана версия, отличается от текущего:
    # L2 не перезаписан, для применения нужен --new-version.
    lexicon_changed: bool = False
    lexicon_run_hash: str = ""
    lexicon_current_hash: str = ""


@dataclass(frozen=True)
class SuspectReport:
    """Итог предварительного сбора подозрительных слов без модели."""

    interview_code: str
    utterances: int
    forms: list[tuple[str, int]]
    total: int
    added: int


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
    manual_edits: ManualEdits | None = None,
) -> NormalizeResult:
    """Нормализовать все реплики интервью и записать L2.

    Args:
        conn: соединение с БД.
        interview_code: внешний код интервью (R01).
        settings: настройки проекта.
        provider: провайдер LLM; None — только детерминированный проход.
        version: версия L2.
        code_version: версия кода для записи в run.
        manual_edits: разовые правки реплик; None — прочитать файл рядом с
            выгрузками (отсутствие файла не ошибка).
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
        current_hash = lexicon_snapshot_hash(load_lexicon(conn))
        frozen_hash, changed = _lexicon_changed(conn, interview_db_id, version, current_hash)
        if changed:
            logger.warning(
                "L2 v%d для %s заморожен на снимке лексикона %s, текущий %s: изменения не применены",
                version,
                interview_code,
                frozen_hash[:12],
                current_hash[:12],
            )
        return NormalizeResult(
            interview_code=interview_code,
            interview_db_id=interview_db_id,
            version=version,
            stats=NormalizeStats(),
            skipped=True,
            lexicon_changed=changed,
            lexicon_run_hash=frozen_hash,
            lexicon_current_hash=current_hash,
        )

    lexicon = load_lexicon(conn)
    markers = load_markers(conn)
    examples = load_examples()
    min_words = settings.m1b.min_words_for_model
    fuzzy = settings.m1b.fuzzy_config()
    edits = (
        manual_edits
        if manual_edits is not None
        else load_manual_edits(manual_edits_path(Path(settings.paths.exports), interview_code))
    )

    outcomes: list[NormalizeOutcome] = []
    for index, (utterance_id, role, text, flag) in enumerate(utterances):
        # Разовая правка применяется до модели и до детерминированного прохода:
        # L1 остаётся неизменным, исправленный текст видит только L2.
        edited_text, applied = edits.apply(utterance_id, text)
        outcome = normalize_utterance(
            utterance_id,
            role,
            bool(flag),
            edited_text,
            previous_question=_previous_question(utterances, index),
            lexicon=lexicon,
            markers=markers,
            examples=examples,
            provider=provider,
            min_words=min_words,
            fuzzy=fuzzy,
            domain_hint=settings.domain_hint,
            prompt_lexicon_filter=settings.m1b.prompt_lexicon_filter,
            prompt_lexicon_threshold=settings.m1b.prompt_lexicon_threshold,
        )
        if applied:
            outcome.changes = [
                *outcome.changes,
                *[
                    {"type": "manual_edit", "from": edit.source, "to": edit.target, "utterance_id": utterance_id}
                    for edit in applied
                ],
            ]
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
        "lexicon_hash": lexicon_snapshot_hash(lexicon),
        "prompt_lexicon_filter": settings.m1b.prompt_lexicon_filter,
        "prompt_lexicon_threshold": settings.m1b.prompt_lexicon_threshold,
        "markers": len(markers),
        "model_calls": stats.model_calls,
        "cached_calls": stats.cached_calls,
    }
    if edits.path is not None:
        # Хеш файла правок виден в прогоне, как хеш лексикона: изменение файла
        # меняет результат и должно быть заметно (SPEC-M8, решение 4).
        params["manual_edits"] = edits.count
        params["manual_edits_hash"] = edits.file_hash

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
            for term in surviving_new_terms(outcome):
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


def _lexicon_changed(
    conn: sqlite3.Connection,
    interview_db_id: int,
    version: int,
    current_hash: str,
) -> tuple[str, bool]:
    """Снимок лексикона записанной версии L2 и признак «лексикон с тех пор изменился».

    Прогоны до M7 не писали `lexicon_hash`. Для них сравнение невозможно, и
    расхождением это не считается: иначе на уже обработанных интервью было бы
    ложное предупреждение.
    """
    run_id = TextUnitRepository(conn).run_id_for_version(interview_db_id, version)
    if run_id is None:
        return "", False
    frozen_hash = str(RunRepository(conn).params(run_id).get("lexicon_hash") or "")
    if not frozen_hash:
        logger.info("L2 v%d записан прогоном без lexicon_hash: сравнение снимков пропущено", version)
        return "", False
    return frozen_hash, frozen_hash != current_hash


def collect_suspects(
    conn: sqlite3.Connection,
    interview_code: str,
    settings: Settings,
) -> SuspectReport:
    """Собрать подозрительные слова интервью без вызовов модели.

    Цель — пополнить глоссарий до прогона: слова, которых нет ни в словаре,
    ни в лексиконе, видны уже на L1. Порог `min_words_for_model` здесь не
    применяется: короткая реплика тоже может содержать искажение. Ни `run`,
    ни `text_unit` не создаются — это подготовительный шаг, а не версия L2.
    """
    interview_db_id = InterviewRepository(conn).interview_db_id_by_code(interview_code)
    if interview_db_id is None:
        msg = f"Интервью {interview_code} не найдено в БД: сначала загрузите транскрипт"
        raise NormalizeError(msg)
    utterances = InterviewRepository(conn).utterances_for_normalize(interview_db_id)
    if not utterances:
        msg = f"У интервью {interview_code} нет реплик: сначала загрузите транскрипт"
        raise NormalizeError(msg)

    lexicon = load_lexicon(conn)
    fuzzy = settings.m1b.fuzzy_config()
    counts: dict[str, int] = {}
    for _utterance_id, role, text, _flag in utterances:
        if role != "RESPONDENT":
            continue
        deterministic = normalize_deterministic(text, lexicon, fuzzy)
        for form, count in suspect_frequencies(deterministic.text, lexicon).items():
            counts[form] = counts.get(form, 0) + count

    terms = NewTermRepository(conn)
    added = 0
    for form in counts:
        if terms.ensure(form, interview_db_id, None):
            added += 1
    conn.commit()

    forms = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return SuspectReport(
        interview_code=interview_code,
        utterances=sum(1 for _uid, role, _text, _flag in utterances if role == "RESPONDENT"),
        forms=forms,
        total=sum(counts.values()),
        added=added,
    )


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
    "SuspectReport",
    "collect_suspects",
    "count_words",
    "load_lexicon",
    "load_markers",
    "normalize_interview",
    "plan_dry_run",
]
