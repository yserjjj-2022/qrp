"""review-terms: интерактивное утверждение или отклонение кандидатов new_term.

Команда понадобится в M1b; здесь каркас: показать кандидатов, спросить решение,
записать статус и, при утверждении, добавить термин в лексикон как решение review.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from qa.db.workbook_repo import LexiconRepository, NewTermRepository

logger = logging.getLogger(__name__)

DECISIONS = {"y": "approved", "н": "approved", "n": "rejected", "т": "rejected", "": "skip", "s": "skip"}


@dataclass
class ReviewResult:
    """Итог ревью терминов."""

    approved: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def _ask(prompt: str) -> str:
    """Спросить решение у оператора (ввод через stdin)."""
    return input(prompt).strip().lower()


def review_terms(
    conn: sqlite3.Connection,
    ask: Callable[[str], str] | None = None,
    only: Iterable[str] | None = None,
) -> ReviewResult:
    """Провести ревью кандидатов new_term.

    Args:
        conn: подключение к SQLite.
        ask: функция ввода (в тестах подменяется); по умолчанию input().
        only: ограничить список терминов.

    Решения review-terms имеют приоритет над листом «Лексикон»: при утверждении
    термин записывается в лексикон с source='review'.
    """
    ask_fn = ask or _ask
    terms_repo = NewTermRepository(conn)
    lexicon_repo = LexiconRepository(conn)
    candidates = terms_repo.proposed()
    if only is not None:
        allowed = {str(item) for item in only}
        candidates = [item for item in candidates if item[1] in allowed]

    result = ReviewResult()
    for term_id, surface_form, context in candidates:
        context_note = f" (контекст: реплика {context})" if context else ""
        answer = ask_fn(f"Термин {surface_form!r}{context_note}: утвердить? [y/n/skip]: ")
        decision = DECISIONS.get(answer, "skip")
        if decision == "skip":
            result.skipped.append(surface_form)
            continue
        terms_repo.decide(term_id, decision)
        if decision == "approved":
            lexicon_repo.upsert_from_workbook(surface_form, [], None, "approved")
            lexicon_repo.apply_review(surface_form, "approved")
            result.approved.append(surface_form)
        else:
            result.rejected.append(surface_form)
    conn.commit()
    logger.info(
        "review-terms: утверждено=%s, отклонено=%s, пропущено=%s",
        len(result.approved),
        len(result.rejected),
        len(result.skipped),
    )
    return result
