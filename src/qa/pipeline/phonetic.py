"""Фонетические подсказки для неопознанных слов ASR (M7b).

Подсказка отвечает на вопрос «на что это похоже по звучанию», а не «что это
значит»: решение принимает модель по смыслу и оператор. Поэтому фонетика
сравнивает слова в одном пространстве — кириллическом, без гласных, — и
возвращает несколько кандидатов с оценкой, а не одну замену.

Латинские термины («OsEngine») расшифровка произносит по-русски, поэтому
латиница переводится в кириллицу по таблице, а не сравнивается с кириллицей
напрямую. Свёртки (`дж`/`ж` → `г`, `й` → `и`) убирают разницу между
написанием и произношением: «Сенжин», «Всенж» и «Усеннджин» дают один скелет
с «OsEngine».
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from rapidfuzz import fuzz

_LATIN_TO_CYRILLIC: dict[str, str] = {
    "a": "а",
    "b": "б",
    "c": "к",
    "d": "д",
    "e": "е",
    "f": "ф",
    "g": "г",
    "h": "х",
    "i": "и",
    "j": "дж",
    "k": "к",
    "l": "л",
    "m": "м",
    "n": "н",
    "o": "о",
    "p": "п",
    "q": "к",
    "r": "р",
    "s": "с",
    "t": "т",
    "u": "у",
    "v": "в",
    "w": "в",
    "x": "кс",
    "y": "и",
    "z": "з",
}

_VOWELS = frozenset("аеиоуыэюя")
_CYRILLIC_RANGE = ("а", "я")
# Минимальная длина скелета: на двух согласных совпадение случайно.
_MIN_SKELETON = 3


def transliterate(word: str) -> str:
    """Перевести латиницу в кириллицу и убрать «ё».

    Кириллица остаётся как есть: смешанное написание («OsEngine», «осенгин»)
    после перевода даёт одну и ту же строку.
    """
    lowered = word.casefold().replace("ё", "е")
    return "".join(_LATIN_TO_CYRILLIC.get(char, char) for char in lowered)


def _letters(word: str) -> str:
    """Только кириллические буквы и пробелы."""
    low, high = _CYRILLIC_RANGE
    return "".join(char for char in transliterate(word) if low <= char <= high or char == " ")


def _collapse(text: str) -> str:
    """Свёртки произношения и сжатие двойных букв."""
    text = text.replace("дж", "г").replace("ж", "г").replace("й", "и")
    text = text.replace("ь", "").replace("ъ", "")
    collapsed: list[str] = []
    for char in text:
        if not collapsed or collapsed[-1] != char:
            collapsed.append(char)
    return "".join(collapsed)


def phonetic_form(word: str) -> str:
    """Полная фонетическая форма слова без свёрнутых повторов."""
    return _collapse(_letters(word)).strip()


def phonetic_key(word: str) -> str:
    """Скелет слова: фонетическая форма без гласных.

    Короткий скелет ненадёжен, поэтому для слов, где согласных меньше трёх,
    берётся полная форма.
    """
    form = phonetic_form(word)
    skeleton = "".join(char for char in form if char not in _VOWELS and char != " ")
    return skeleton if len(skeleton) >= _MIN_SKELETON else form


def phonetic_score(left: str, right: str) -> float:
    """Сходство двух слов по звучанию (0–1).

    Сравниваются и скелеты, и полные формы: скелет ловит совпадение корня,
    полная форма — различие в гласных, когда согласные совпали случайно.
    """
    left_key, right_key = phonetic_key(left), phonetic_key(right)
    if not left_key or not right_key:
        return 0.0
    skeleton = fuzz.ratio(left_key, right_key) / 100
    full = fuzz.ratio(phonetic_form(left), phonetic_form(right)) / 100
    return max(skeleton, full)


def phonetic_candidates(
    form: str,
    targets: Mapping[str, str],
    *,
    threshold: float,
    short_threshold: float,
    short_length: int,
    limit: int,
) -> list[tuple[str, float]]:
    """Кандидаты лексикона, похожие на слово по звучанию.

    Args:
        form: неопознанное слово.
        targets: карта «форма лексикона (folded) -> каноническая форма».
        threshold: порог сходства для обычных слов.
        short_threshold: порог для слов короче `short_length`.
        short_length: граница «короткого» слова.
        limit: сколько кандидатов вернуть.

    Returns:
        Пары (канон, сходство) по убыванию сходства, без повторов канона.
    """
    if not form:
        return []
    min_score = short_threshold if len(form) < short_length else threshold
    best: dict[str, float] = {}
    for surface, canonical in targets.items():
        score = phonetic_score(form, surface)
        if score < min_score:
            continue
        if score > best.get(canonical, 0.0):
            best[canonical] = score
    ordered = sorted(best.items(), key=lambda item: (-item[1], item[0].casefold()))
    return ordered[:limit]


def split_candidates(form: str, is_known: Callable[[str], bool], *, min_part: int = 3, limit: int = 2) -> list[str]:
    """Разделить слово на два известных: «сенжин» -> «сен жин».

    Склейка в обратную сторону — `apply_glue`; здесь проверяется только
    граница внутри одного слова. Обе части должны быть известны словарю или
    лексикону, иначе деление ничего не объясняет.
    """
    folded = form.casefold()
    found: list[str] = []
    for position in range(min_part, len(folded) - min_part + 1):
        left, right = folded[:position], folded[position:]
        if len(left) < min_part or len(right) < min_part:
            continue
        if is_known(left) and is_known(right):
            found.append(f"{left} {right}")
            if len(found) >= limit:
                break
    return found


__all__ = [
    "phonetic_candidates",
    "phonetic_form",
    "phonetic_key",
    "phonetic_score",
    "split_candidates",
    "transliterate",
]
