"""Качество распознавания фрагмента (`fragment.asr_quality`).

Качество считается по самому фрагменту, а не переносится с единицы L2: кодируется
фрагмент, и его текст — единственное основание для оценки. Перенос «единица ->
все её фрагменты» давал `garbled` там, где в срезе нет ни одного неразборчивого
места.

Правило (SPEC-M2, «Искажения ASR»):
1. `garbled` — в тексте фрагмента есть маркер `[неразборчиво]`;
2. `doubtful` — есть слово, заменённое по `asr_fix` своей единицы, слово из
   `new_term` этого интервью или подозрительное слово по лексикону;
3. `ok` — ни одного из условий.

Сравнение идёт по токенам в нижнем регистре: регистр и пунктуация не должны
превращать обычное слово в искажение.
"""

from __future__ import annotations

import re

from qa.pipeline.normalize_rules import LexiconEntry, suspect_words

GARBLED_MARKER = "[неразборчиво]"

# Токен — слово с внутренними дефисами и точками («т.е.», «C#»), но без
# завершающей точки: иначе «сверяю.» не совпало бы с «сверяю» из журнала замен.
_WORD_RE = re.compile(r"[\w#+]+(?:[.-][\w#+]+)*", re.UNICODE)


def tokens_of(text: str) -> set[str]:
    """Токены текста в нижнем регистре."""
    return {match.group(0).casefold() for match in _WORD_RE.finditer(text)}


def fragment_asr_quality(
    text: str,
    *,
    asr_fix_words: set[str],
    new_term_words: set[str],
    lexicon: list[LexiconEntry],
) -> str:
    """Оценка качества фрагмента по его собственному тексту.

    Args:
        text: дословный срез L2 (текст фрагмента).
        asr_fix_words: слова, заменённые правилом `asr_fix` в единице L2.
        new_term_words: формы кандидатов `new_term` этого интервью.
        lexicon: термины лексикона для поиска подозрительных слов.
    """
    if GARBLED_MARKER in text:
        return "garbled"
    tokens = tokens_of(text)
    if tokens & asr_fix_words or tokens & new_term_words:
        return "doubtful"
    if suspect_words(text, lexicon):
        return "doubtful"
    return "ok"
