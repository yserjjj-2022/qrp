"""Обработка решений `review-terms` по кандидатам `new_term`.

Оператор вводит решение одной строкой:

- `y` — принять форму как отдельный термин;
- `y Термин` — принять как вариант указанного термина (основной способ);
- `y` без термина — задаётся второй вопрос (запасной путь);
- `n` — отклонить;
- `skip` или пустая строка — оставить без решения.

Замена варианта выполняется детерминированным проходом, поэтому новый вариант
начинает работать сразу после `normalize` и не зависит от модели.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from qa.db.workbook_repo import LexiconRepository, NewTermRepository

logger = logging.getLogger(__name__)

DECISIONS = {"y": "approved", "н": "approved", "n": "rejected", "т": "rejected", "": "skip", "s": "skip"}
_APPROVE = frozenset({"y", "н", "yes", "да"})
_REJECT = frozenset({"n", "т", "no", "нет"})
_SKIP = frozenset({"", "s", "skip", "пропустить"})


@dataclass
class ReviewResult:
    """Итог ревью терминов."""

    approved: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    variants: dict[str, str] = field(default_factory=dict)
    stale: int = 0


def _ask(prompt: str) -> str:
    """Спросить решение у оператора (ввод через stdin)."""
    return input(prompt).strip()


def _parse_answer(answer: str) -> tuple[str, str]:
    """Разобрать ответ: решение и, если указан, канонический термин.

    «y MCP» — принять форму «кандидат» вариантом термина MCP.
    «y» — принять отдельным термином (второй вопрос не задаётся).
    """
    head, _, tail = answer.strip().partition(" ")
    decision = head.casefold()
    canonical = tail.strip()
    if decision in _APPROVE:
        return "approved", canonical
    if decision in _REJECT:
        return "rejected", ""
    if decision in _SKIP:
        return "skip", ""
    return "skip", ""


def review_terms(
    conn: sqlite3.Connection,
    ask: Callable[[str], str] | None = None,
    only: Iterable[str] | None = None,
    ask_canonical: Callable[[str], str] | None = None,
) -> ReviewResult:
    """Провести ревью кандидатов new_term.

    Args:
        conn: подключение к SQLite.
        ask: ввод решения (в тестах подменяется); по умолчанию input().
        only: ограничить список терминов.
        ask_canonical: второй вопрос «искажение какого термина?»; по умолчанию
            input(). Вызывается только когда решение «y» без указания термина.

    Решения review-terms имеют приоритет над листом «Лексикон»: при утверждении
    форма записывается в лексикон с source='review'. Если указан канонический
    термин, форма становится его вариантом и заменяется детерминированным
    проходом, а не моделью.

    В очередь попадают только кандидаты, чья форма есть в актуальной версии L2
    своего интервью. Актуальная версия локальна для интервью: старые версии
    R01 и единственная версия другого интервью равноправны. Кандидаты, которых
    в актуальном тексте нет, не спрашиваются и остаются в статусе `proposed`;
    их число выводится как `stale`.
    """
    ask_fn = ask or _ask
    ask_canonical_fn = ask_canonical or _ask
    terms_repo = NewTermRepository(conn)
    lexicon_repo = LexiconRepository(conn)
    all_candidates = terms_repo.proposed()
    candidates = terms_repo.proposed_on_current_l2()
    if only is not None:
        allowed = {str(item) for item in only}
        candidates = [item for item in candidates if item[1] in allowed]
        all_candidates = [item for item in all_candidates if item[1] in allowed]

    result = ReviewResult(stale=len(all_candidates) - len(candidates))
    if result.stale:
        logger.info("review-terms: пропущено кандидатов вне актуального L2: %s", result.stale)

    for term_id, surface_form, context in candidates:
        context_note = f" (контекст: реплика {context})" if context else ""
        answer = ask_fn(f"Термин {surface_form!r}{context_note}: утвердить? [y/n/skip]: ")
        decision, canonical = _parse_answer(answer)
        if decision == "skip":
            result.skipped.append(surface_form)
            continue
        terms_repo.decide(term_id, decision)
        if decision == "rejected":
            result.rejected.append(surface_form)
            continue
        if not canonical:
            canonical = ask_canonical_fn(
                f"  Искажение какого термина? [Enter — отдельный термин {surface_form!r}]: "
            ).strip()
        if canonical:
            added = lexicon_repo.add_variant(canonical, surface_form)
            if added:
                result.variants[surface_form] = canonical
            else:
                logger.info(
                    "review-terms: вариант %r для %r не добавлен (пустой или уже есть)", surface_form, canonical
                )
        else:
            lexicon_repo.upsert_from_workbook(surface_form, [], None, "approved")
            lexicon_repo.apply_review(surface_form, "approved")
        result.approved.append(surface_form)
    conn.commit()
    logger.info(
        "review-terms: утверждено=%s, вариантов=%s, отклонено=%s, пропущено=%s",
        len(result.approved),
        len(result.variants),
        len(result.rejected),
        len(result.skipped),
    )
    return result
