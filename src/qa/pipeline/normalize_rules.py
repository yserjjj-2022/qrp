"""Детерминированный проход нормализации L1 -> L2.

Правила без модели: замены по лексикону, удаление звуков-паразитов, схлопывание
механических повторов служебных слов, пробелы и пунктуация. Проход применяется
ко всем репликам, включая интервьюера и бэкканалы.

Смысловые повторы («не знаю, не знаю») сохраняются: схлопываются только
подряд идущие одинаковые служебные слова. Повтор смыслового слова остаётся,
а в журнал попадает `repeat_unclear` — консервативное решение по SPEC-M1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Служебные слова: предлоги, союзы, частицы, местоимения. Только их повтор
# считается механическим заиканием.
FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        "а",
        "без",
        "бы",
        "в",
        "вам",
        "вас",
        "весь",
        "во",
        "вот",
        "все",
        "всё",
        "вы",
        "да",
        "для",
        "до",
        "его",
        "ее",
        "её",
        "ей",
        "ему",
        "если",
        "есть",
        "же",
        "за",
        "и",
        "из",
        "или",
        "им",
        "их",
        "к",
        "как",
        "ко",
        "когда",
        "кто",
        "ли",
        "мне",
        "мы",
        "на",
        "над",
        "нам",
        "нас",
        "не",
        "нет",
        "ни",
        "но",
        "ну",
        "о",
        "об",
        "он",
        "она",
        "они",
        "оно",
        "от",
        "по",
        "под",
        "при",
        "про",
        "с",
        "со",
        "так",
        "там",
        "те",
        "то",
        "тоже",
        "тот",
        "ты",
        "у",
        "уже",
        "что",
        "чтобы",
        "это",
        "этот",
        "я",
    }
)

# Звуки-паразиты и междометия без смысла. Требуется отдельный токен,
# чтобы не задеть единицы измерения и части слов.
_SOUND_RE = re.compile(
    r"(?<![а-яёa-z0-9])(?:э+(?:-э+)*|м+(?:-м+)*|аа+|оо+|угу-угу)(?![а-яёa-z0-9])",
    re.IGNORECASE,
)

# Подряд идущий повтор одного слова с разделителями.
_REPEAT_RE = re.compile(
    r"(?<![а-яёa-z0-9])([а-яёa-z]+)((?:[\s,;.\-—]+\1(?![а-яёa-z0-9]))+)",
    re.IGNORECASE,
)

_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
_SPACE_AFTER_PUNCT_RE = re.compile(r"([,.;:!?])(?=[^\s,.;:!?»)\]])")
_MULTI_SPACE_RE = re.compile(r"[ 	\u00a0]+")
_LEADING_PUNCT_RE = re.compile(r"^[\s,;:.\-—]+")

# Повтор смысловой фразы из 2-4 слов: «не знаю, не знаю». Такой повтор
# сохраняется, но помечается как неясный (SPEC-M1: при сомнении оставить).
_PHRASE_REPEAT_RE = re.compile(
    r"(?<![а-яёa-z0-9])((?:[а-яёa-z]+[\s,]+){1,3}[а-яёa-z]+)[\s,;.\-—]+\1(?![а-яёa-z0-9])",
    re.IGNORECASE,
)

_TERMINAL_PUNCT = ".!?…»\"'"


@dataclass(frozen=True)
class LexiconEntry:
    """Термин лексикона: каноническая форма и варианты ASR-искажений."""

    canonical: str
    variants: tuple[str, ...] = ()


@dataclass
class DeterministicResult:
    """Итог детерминированного прохода."""

    text: str
    changes: list[dict[str, Any]] = field(default_factory=list)
    fillers_removed: int = 0
    stutter_removed: int = 0
    repeat_unclear: int = 0
    term_replacements: dict[str, int] = field(default_factory=dict)


def apply_lexicon(text: str, entries: list[LexiconEntry]) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
    """Заменить варианты терминов канонической формой.

    Совпадение целыми словами, без учёта регистра. Каноническая форма берётся
    из лексикона, поэтому «эсипи» и «Эсипи» дают один и тот же вид.
    """
    pairs: list[tuple[str, str]] = []
    for entry in entries:
        for variant in entry.variants:
            if variant.strip() and variant.strip().casefold() != entry.canonical.casefold():
                pairs.append((variant.strip(), entry.canonical))
    if not pairs:
        return text, [], {}

    # Длинные варианты первыми: «чат GPT» не должен перехватываться «GPT».
    pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
    changes: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    result = text
    for variant, canonical in pairs:
        pattern = re.compile(
            rf"(?<![а-яёa-z0-9]){re.escape(variant)}(?![а-яёa-z0-9])",
            re.IGNORECASE,
        )
        found = pattern.findall(result)
        if not pattern.search(result):
            continue
        result = pattern.sub(canonical, result)
        changes.append({"type": "term", "from": variant, "to": canonical})
        counts[canonical] = counts.get(canonical, 0) + len(found)
    return result, changes, counts


def remove_sounds(text: str) -> tuple[str, int, list[dict[str, Any]]]:
    """Удалить звуки-паразиты («э-э», «м-м»)."""
    matches = [match.group(0) for match in _SOUND_RE.finditer(text)]
    if not matches:
        return text, 0, []
    cleaned = _SOUND_RE.sub(" ", text)
    changes = [{"type": "filler", "from": value, "to": ""} for value in matches]
    return cleaned, len(matches), changes


def collapse_repeats(text: str) -> tuple[str, int, int, list[dict[str, Any]]]:
    """Схлопнуть механические повторы служебных слов.

    Returns:
        (текст, удалено повторов, неясных повторов, изменения).
    """
    removed = 0
    unclear = 0
    changes: list[dict[str, Any]] = []

    def replace(match: re.Match[str]) -> str:
        nonlocal removed, unclear
        word = match.group(1)
        occurrences = len(re.findall(rf"(?<![а-яёa-z0-9]){re.escape(word)}(?![а-яёa-z0-9])", match.group(0), re.I))
        if word.casefold() in FUNCTION_WORDS:
            removed += max(0, occurrences - 1)
            changes.append({"type": "stutter", "from": match.group(0).strip(), "to": word})
            return word
        # Смысловой повтор: оставить как есть, отметить для человека.
        unclear += 1
        return match.group(0)

    result = _REPEAT_RE.sub(replace, text)
    return result, removed, unclear, changes


def count_phrase_repeats(text: str) -> int:
    """Сколько раз в тексте повторяется фраза из нескольких слов.

    Такие повторы не удаляются: они могут быть обдумыванием или усилением.
    """
    return len(_PHRASE_REPEAT_RE.findall(text))


def normalize_punctuation(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Привести пробелы и пунктуацию к аккуратному виду."""
    original = text
    result = _MULTI_SPACE_RE.sub(" ", text).strip()
    result = _LEADING_PUNCT_RE.sub("", result)
    result = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", result)
    result = _SPACE_AFTER_PUNCT_RE.sub(r"\1 ", result)
    result = _MULTI_SPACE_RE.sub(" ", result).strip()
    if result and result[0].islower():
        result = result[0].upper() + result[1:]
    if result and result[-1] not in _TERMINAL_PUNCT:
        result = f"{result}."
    changes: list[dict[str, Any]] = []
    if result != original:
        changes.append({"type": "punctuation", "from": original, "to": result})
    return result, changes


def normalize_deterministic(text: str, entries: list[LexiconEntry]) -> DeterministicResult:
    """Выполнить полный детерминированный проход по одной реплике."""
    changes: list[dict[str, Any]] = []

    result, term_changes, term_counts = apply_lexicon(text, entries)
    changes.extend(term_changes)

    result, fillers, filler_changes = remove_sounds(result)
    changes.extend(filler_changes)

    result, stutter, unclear, stutter_changes = collapse_repeats(result)
    changes.extend(stutter_changes)

    # Повторы смысловых фраз не удаляются, но отмечаются для человека.
    unclear += count_phrase_repeats(result)

    result, punctuation_changes = normalize_punctuation(result)
    changes.extend(punctuation_changes)

    return DeterministicResult(
        text=result,
        changes=changes,
        fillers_removed=fillers,
        stutter_removed=stutter,
        repeat_unclear=unclear,
        term_replacements=term_counts,
    )
