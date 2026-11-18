"""Память терминов и подсказки оператору (M7b).

Подсказка — это вариант замены для слова, которого нет ни в словаре, ни в
лексиконе. Кандидаты даёт код (созвучие, склейка, разделение), оценку по
смыслу — модель: несколько выборок при ненулевой температуре, уверенность
варианта равна доле выборок за него.

Модель не может вернуть слово вне списка кандидатов: ответ содержит индекс,
а не текст. Поэтому подсказка всегда проверяема, а решение оператора остаётся
единственным источником новых терминов.

SQL здесь нет: лексикон читается и пишется через `LexiconRepository`
(CONSTITUTION §4).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from qa.db.workbook_repo import LexiconRepository
from qa.llm.base import LlmProvider
from qa.llm.errors import LlmError, LlmValidationError
from qa.pipeline.normalize_rules import LexiconEntry, is_known_word, lexicon_forms
from qa.pipeline.phonetic import phonetic_candidates, phonetic_score, split_candidates
from qa.settings import M7bSettings

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"

# Варианты «не заменять»: в таблице показываются как есть, в лексикон
# записываются статусами `keep` и `unreadable`.
KEEP = "оставить"
UNREADABLE = "[неразборчиво]"

PICK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["candidate", "keep", "unreadable"]},
        "candidate": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "candidate", "reason"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class TermHint:
    """Вариант замены слова с оценкой уверенности."""

    to: str
    confidence: float
    source: str
    glued: bool = False

    def label(self) -> str:
        """Текст для ячейки таблицы: `OsEngine (0.89)`."""
        return f"{self.to} ({self.confidence:.2f})"


def load_pick_prompt() -> str:
    """Прочитать промпт выбора подсказки."""
    return (PROMPTS_DIR / "normalize_pick.md").read_text(encoding="utf-8")


def _is_known(form: str, forms: dict[str, str]) -> bool:
    """Известно ли слово словарю или лексикону."""
    return is_known_word(form) or form.casefold() in forms


def glue_hint(form: str, next_word: str | None, entries: list[LexiconEntry]) -> TermHint | None:
    """Склейка со следующим словом: «ре» + «балансланировка» -> термин.

    Подсказка помечается `glued=True`: в таблице видно, что замена относится к
    паре слов, а не к одному.
    """
    if not next_word:
        return None
    joined = f"{form}{next_word}".casefold()
    compact = {key.replace(" ", ""): canonical for key, canonical in lexicon_forms(entries).items()}
    best: tuple[str, float] | None = None
    for surface, canonical in compact.items():
        score = fuzz.ratio(joined, surface) / 100
        if best is None or score > best[1]:
            best = (canonical, score)
    if best is None or best[1] < 0.80:
        return None
    return TermHint(to=best[0], confidence=best[1], source="code", glued=True)


def build_code_hints(
    form: str,
    entries: list[LexiconEntry],
    settings: M7bSettings,
    next_word: str | None = None,
) -> list[TermHint]:
    """Подсказки без модели: созвучие, склейка, разделение.

    Случайный набор букв не даёт словесных кандидатов: остаются только
    «оставить» и «[неразборчиво]» — их добавит вызывающая сторона.
    """
    forms = lexicon_forms(entries)
    hints = [
        TermHint(to=canonical, confidence=score, source="phonetic")
        for canonical, score in phonetic_candidates(
            form,
            forms,
            threshold=settings.phonetic_threshold,
            short_threshold=settings.phonetic_short_threshold,
            short_length=settings.phonetic_short_length,
            limit=settings.hint_max,
        )
    ]
    for split in split_candidates(form, lambda part: _is_known(part, forms)):
        hints.append(TermHint(to=split, confidence=0.5, source="code"))
    glued = glue_hint(form, next_word, entries)
    if glued is not None:
        hints.append(glued)
    return hints[: settings.hint_max]


def memory_examples(
    terms: dict[str, dict[str, Any]],
    form: str,
    limit: int,
) -> list[tuple[str, str]]:
    """Похожие прошлые решения оператора: (форма ASR, канон).

    Память показывает модели, как оператор решал похожие случаи. Берутся
    только записи со статусом решения; лист «Лексикон» из книги сюда не
    попадает, иначе примеры не отражали бы решений.
    """
    pairs: list[tuple[str, str, float]] = []
    for canonical, data in terms.items():
        status = str(data.get("status") or "")
        if status in ("keep", "unreadable"):
            pairs.append((canonical, canonical, phonetic_score(form, canonical)))
            continue
        if status not in ("review", "auto"):
            continue
        for variant in data.get("variants") or []:
            surface = str(variant)
            pairs.append((surface, canonical, phonetic_score(form, surface)))
    pairs.sort(key=lambda item: (-item[2], item[0].casefold()))
    return [(surface, canonical) for surface, canonical, _score in pairs[:limit]]


def build_pick_prompt(
    form: str,
    context: str,
    hints: list[TermHint],
    memory: list[tuple[str, str]],
) -> str:
    """Собрать промпт выбора подсказки из шаблона."""
    candidates = "\n".join(f"{index}. {hint.to}" for index, hint in enumerate(hints)) or "нет"
    examples = "\n".join(f"- {surface} -> {canonical}" for surface, canonical in memory) or "нет"
    return (
        load_pick_prompt()
        .replace("{form}", form)
        .replace("{context}", context)
        .replace("{candidates}", candidates)
        .replace("{memory}", examples)
    )


def pick_with_model(
    form: str,
    context: str,
    hints: list[TermHint],
    memory: list[tuple[str, str]],
    providers: Sequence[LlmProvider],
) -> list[TermHint]:
    """Оценить кандидатов моделью: несколько выборок, уверенность — доля.

    Пустой список провайдеров или все неудачные выборки дают пустой результат:
    остаются только подсказки кода.
    """
    if not providers or not hints:
        return []
    prompt = build_pick_prompt(form, context, hints, memory)
    votes: dict[str, float] = {}
    answered = 0
    for provider in providers:
        try:
            data = provider.complete_json(prompt, PICK_SCHEMA)
        except (LlmError, LlmValidationError) as exc:
            logger.info("M7b: выборка подсказки для %r не удалась: %s", form, exc)
            continue
        answered += 1
        verdict = str(data.get("verdict") or "")
        if verdict == "candidate":
            index = data.get("candidate")
            if isinstance(index, int) and 0 <= index < len(hints):
                target = hints[index].to
            else:
                logger.info("M7b: модель вернула индекс %r вне списка для %r", index, form)
                continue
        elif verdict == "keep":
            target = KEEP
        elif verdict == "unreadable":
            target = UNREADABLE
        else:
            continue
        votes[target] = votes.get(target, 0.0) + 1.0
    if not answered:
        return []
    result = [TermHint(to=target, confidence=count / answered, source="model") for target, count in votes.items()]
    result.sort(key=lambda hint: (-hint.confidence, hint.to.casefold()))
    return result


def merge_hints(code_hints: list[TermHint], model_hints: list[TermHint], limit: int) -> list[TermHint]:
    """Объединить подсказки модели и кода без повторов варианта.

    Модельная оценка приоритетнее: она учитывает смысл. Одинаковые варианты
    схлопываются, «оставить» и «[неразборчиво]» остаются последними.
    """
    merged: dict[str, TermHint] = {}
    for hint in [*model_hints, *code_hints]:
        if hint.to not in merged:
            merged[hint.to] = hint
    ordered = sorted(
        merged.values(),
        key=lambda hint: (hint.to in (KEEP, UNREADABLE), -hint.confidence, hint.to.casefold()),
    )
    return ordered[:limit]


def build_hints(
    form: str,
    entries: list[LexiconEntry],
    settings: M7bSettings,
    *,
    context: str = "",
    next_word: str | None = None,
    providers: Sequence[LlmProvider] = (),
    terms: dict[str, dict[str, Any]] | None = None,
) -> list[TermHint]:
    """Полный набор подсказок для слова: код, память и модель."""
    code_hints = build_code_hints(form, entries, settings, next_word)
    memory = memory_examples(terms or {}, form, settings.memory_neighbours)
    model_hints = pick_with_model(form, context, code_hints, memory, providers)
    hints = merge_hints(code_hints, model_hints, settings.hint_max)
    if not hints:
        return [TermHint(to=UNREADABLE, confidence=0.0, source="code")]
    return hints


def apply_choice(conn: Any, form: str, choice: str) -> str:  # noqa: ANN401
    """Записать выбор оператора по форме.

    choice: `=` — оставить, `?` — `[неразборчиво]`, иначе текст замены.
    """
    repo = LexiconRepository(conn)
    value = choice.strip()
    if value == "=":
        return repo.set_form_decision(form, form, "keep")
    if value == "?":
        return repo.set_form_decision(form, form, "unreadable")
    return repo.set_form_decision(form, value, "review")


__all__ = [
    "KEEP",
    "PICK_SCHEMA",
    "UNREADABLE",
    "TermHint",
    "apply_choice",
    "build_code_hints",
    "build_hints",
    "build_pick_prompt",
    "glue_hint",
    "load_pick_prompt",
    "memory_examples",
    "merge_hints",
    "pick_with_model",
]
