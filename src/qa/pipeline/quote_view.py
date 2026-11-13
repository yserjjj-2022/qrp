"""Представление цитат в отчёте: отбор, обрезка и необязательная правка (M8).

Слой отчёта: L1 и L2 не меняются. Цитата — срез L2 по позициям аннотации;
рядом показывается оригинал с тайм-кодом, чтобы читатель мог проверить текст
(MANIFEST §8.2: нет цитат, которых нет в тексте).

Правила отбора (SPEC-M8, C.1):

- `garbled` не попадает: текст нечитаем, цитата из него ничего не доказывает;
- `doubtful` допускается, но идёт после `ok` и помечается;
- предпочтение — `ok`, длина 80–400 символов, `reliability >= 2`, не
  `interpretation_only`.

Обрезка детерминированная, без модели: обрывок режется по границе слова и
отмечается `[…]`, а не `…`, чтобы отличать вырезку от авторского многоточия.

Правка моделью включается настройкой `m8.quote_edit_enabled` (по умолчанию
выключена) и проходит автопроверки: числа и отрицания сохранены, добавления
только в квадратных скобках, доля изменений не выше порога. Не прошедшая
проверку правка отбрасывается, показывается оригинал.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qa.db.analytics_repo import AnnotationRow
from qa.llm.base import LlmProvider
from qa.llm.errors import LlmError
from qa.settings import Settings

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"

CUT_MARKER = "[…]"
# Предпочтительная длина цитаты: короче — обрывок мысли, длиннее — пересказ.
PREFERRED_MIN_CHARS = 80
PREFERRED_MAX_CHARS = 400

# Правка цитаты: (текст) -> (правка или None, причина отказа, название правила).
# Тип объявлен здесь, а не в `aggregate`: `aggregate` импортирует этот модуль.
QuoteEditor = Callable[[str], tuple[str | None, str, str]]

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_NEGATIONS = ("не", "нет", "ни", "без")
_WORD_RE = re.compile(r"[а-яёa-z]+(?:-[а-яёa-z]+)?", re.IGNORECASE)
_LATIN_RE = re.compile(r"[A-Za-z]")
_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
_BRACKET_RE = re.compile(r"\[[^\]]*\]")

QUOTE_EDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "edited_text": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["edited_text", "reason"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Quote:
    """Цитата отчёта: текст для показа и проверяемый оригинал.

    `text` — то, что читает человек (возможно, обрезанное и, если включена
    правка, отредактированное). `original_l2` — дословный срез L2, `original_l1`
    — исходная реплика, `timecode` — тайм-код реплики в секундах.
    """

    code_name: str
    text: str
    question_identifier: str | None
    original_l2: str
    original_l1: str
    timecode: int | None
    quality: str
    truncated: bool
    edited: bool = False
    edit_rule: str = ""


def truncate_quote(text: str, limit: int) -> tuple[str, bool]:
    """Обрезать цитату по границе слова и отметить вырезку `[…]`.

    Returns:
        (текст, признак обрезки). Текст короче предела возвращается как есть.
    """
    if limit <= 0 or len(text) <= limit:
        return text, False
    cut = text[:limit]
    space = cut.rfind(" ")
    if space > 0:
        cut = cut[:space]
    return cut.rstrip(" ,;:—-") + " " + CUT_MARKER, True


def quote_source_text(annotation: AnnotationRow) -> str:
    """Дословный текст цитаты: срез L2 по позициям, иначе сохранённая цитата.

    Позиции относятся к тексту единицы L2, поэтому срез по ним и есть цитата
    (SPEC-M6, критерий 9). `annotation.quote` используется, только если позиций
    нет: сохранённая строка может быть уже обрезана.
    """
    if annotation.quote_start is not None and annotation.quote_end is not None:
        sliced = annotation.l2_text[annotation.quote_start : annotation.quote_end]
        if sliced:
            return sliced
    return annotation.quote


def _selection_key(annotation: AnnotationRow, min_chars: int) -> tuple[int, int, int, int, int]:
    """Порядок цитат: `ok` раньше `doubtful`, надёжные раньше, длинные в норме.

    Первый элемент — качество (`garbled` уже отфильтрован), второй —
    предпочтительная длина, третий — надёжность по убыванию, четвёртый —
    признак «только интерпретация» (такие в конец), пятый — стабильный id.
    """
    quality_rank = 0 if annotation.fragment_quality == "ok" else 1
    length = len(quote_source_text(annotation))
    length_rank = 0 if min_chars <= length <= PREFERRED_MAX_CHARS else 1
    reliability = annotation.reliability if annotation.reliability is not None else 0
    interpretation = 1 if annotation.interpretation_only else 0
    return (quality_rank, length_rank, -reliability, interpretation, annotation.id)


def select_annotations(
    annotations: list[AnnotationRow],
    limit: int,
    *,
    min_reliability: int = 2,
    min_chars: int = PREFERRED_MIN_CHARS,
) -> list[AnnotationRow]:
    """Отобрать цитаты по правилам SPEC-M8: без `garbled`, `ok` впереди.

    `doubtful` не отбрасывается: иначе тема осталась бы без цитат на живом
    интервью, где сомнительных фрагментов около половины. Порядок ставит их
    после `ok`, а рендер помечает.

    Args:
        annotations: аннотации темы (или подраздела).
        limit: сколько цитат нужно.
        min_reliability: предпочтительная нижняя граница надёжности; аннотации
            ниже не отбрасываются, а идут в конце.
        min_chars: нижняя граница предпочтительной длины (`m8.quote_min_chars`).
    """
    if limit <= 0:
        return []
    candidates = [item for item in annotations if item.fragment_quality != "garbled"]
    if not candidates:
        return []
    preferred = [item for item in candidates if (item.reliability or 0) >= min_reliability]
    rest = [item for item in candidates if (item.reliability or 0) < min_reliability]
    ordered = sorted(preferred, key=lambda item: _selection_key(item, min_chars)) + sorted(
        rest, key=lambda item: _selection_key(item, min_chars)
    )
    return ordered[:limit]


def build_quote(
    annotation: AnnotationRow,
    code_name: str,
    limit: int,
    *,
    edited_text: str | None = None,
    edit_rule: str = "",
) -> Quote:
    """Собрать цитату: показать текст, рядом оставить оригинал."""
    source = quote_source_text(annotation)
    text, truncated = truncate_quote(source, limit)
    edited = edited_text is not None
    if edited_text is not None:
        text = edited_text
    return Quote(
        code_name=code_name,
        text=text,
        question_identifier=annotation.question_identifier,
        original_l2=source,
        original_l1=annotation.l1_text,
        timecode=annotation.timecode,
        quality=annotation.fragment_quality,
        truncated=truncated,
        edited=edited,
        edit_rule=edit_rule,
    )


def load_prompt() -> str:
    """Прочитать промпт правки цитаты."""
    return (PROMPTS_DIR / "quote_edit.md").read_text(encoding="utf-8")


def prompt_hash(text: str) -> str:
    """Хеш промпта для записи правила в отчёт."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_edit_prompt(text: str) -> str:
    """Собрать промпт правки цитаты."""
    return load_prompt().replace("{quote}", text)


def _numbers(text: str) -> list[str]:
    """Числа в тексте в сопоставимом виде."""
    return [value.replace(",", ".") for value in _NUMBER_RE.findall(text)]


def _tokens(text: str) -> list[str]:
    """Слова в нижнем регистре."""
    return [value.casefold() for value in _WORD_RE.findall(text)]


def _change_share(source: str, target: str) -> float:
    """Доля изменённых символов относительно исходной длины."""
    if not source:
        return 1.0
    common = sum(1 for left, right in zip(source, target, strict=False) if left == right)
    return 1.0 - common / max(len(source), len(target))


def check_quote_edit(
    source: str,
    edited: str,
    *,
    max_change_share: float = 0.25,
) -> str | None:
    """Проверить правку цитаты; вернуть причину отказа или None.

    Проверки повторяют логику нормализации в объёме, нужном для цитаты:
    числа и отрицания сохранены, добавления только в `[квадратных скобках]`,
    язык тот же, доля изменений не выше порога.
    """
    if not edited.strip():
        return "правка пуста"
    if sorted(_numbers(source)) != sorted(_numbers(edited)):
        return "изменены числа"
    source_tokens = _tokens(source)
    edited_tokens = _tokens(edited)
    for negation in _NEGATIONS:
        if source_tokens.count(negation) > edited_tokens.count(negation):
            return f"потеряно отрицание {negation!r}"
    # Добавления допустимы только в квадратных скобках: всё остальное —
    # переписывание текста респондента.
    without_brackets = _BRACKET_RE.sub(" ", edited)
    if _CYRILLIC_RE.search(source) and not _CYRILLIC_RE.search(without_brackets):
        return "правка не на русском языке"
    if _LATIN_RE.search(edited) and not _LATIN_RE.search(source):
        return "в правке появилась латиница"
    if _change_share(source, edited) > max_change_share:
        return "правка меняет слишком много текста"
    return None


def edit_quote(
    text: str,
    provider: LlmProvider | None,
    *,
    max_change_share: float = 0.25,
) -> tuple[str | None, str, str]:
    """Предложить правку цитаты моделью и проверить её.

    Returns:
        (правка или None, причина отказа, название правила). Правка возвращается
        только если прошла автопроверки; иначе None и рендер покажет оригинал.
    """
    if provider is None:
        return None, "модель не подключена", ""
    rule = f"правка цитаты моделью, промпт {prompt_hash(load_prompt())[:12]}"
    try:
        result = provider.complete_json_detailed(build_edit_prompt(text), QUOTE_EDIT_SCHEMA)
    except LlmError as exc:
        logger.info("Правка цитаты не выполнена: %s", exc)
        return None, "вызов модели не удался", rule
    edited = str(result.data.get("edited_text") or "").strip()
    failure = check_quote_edit(text, edited, max_change_share=max_change_share)
    if failure is not None:
        logger.info("Правка цитаты отклонена: %s", failure)
        return None, failure, rule
    return edited, "", rule


def quote_editor_from_settings(settings: Settings) -> QuoteEditor | None:
    """Правка цитат моделью, если её включил оператор; иначе None.

    Возвращает функцию в форме `aggregate.QuoteEditor`, чтобы вызывающая
    сторона не знала о провайдере. Выключенная настройка — не ошибка: отчёт
    собирается с дословными цитатами, вызовов модели нет (SPEC-M8, C.3).
    """
    if not settings.m8.quote_edit_enabled:
        return None
    provider: LlmProvider | None = None
    if settings.llm.provider != "stub":
        from qa.llm import provider_from_settings

        provider = provider_from_settings(settings)
    max_share = settings.m8.quote_edit_max_change_share

    def editor(text: str) -> tuple[str | None, str, str]:
        return edit_quote(text, provider, max_change_share=max_share)

    return editor


__all__ = [
    "CUT_MARKER",
    "PREFERRED_MAX_CHARS",
    "PREFERRED_MIN_CHARS",
    "QUOTE_EDIT_SCHEMA",
    "Quote",
    "build_edit_prompt",
    "build_quote",
    "check_quote_edit",
    "edit_quote",
    "load_prompt",
    "prompt_hash",
    "quote_editor_from_settings",
    "quote_source_text",
    "select_annotations",
    "truncate_quote",
]
