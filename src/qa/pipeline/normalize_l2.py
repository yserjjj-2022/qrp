"""Нормализация L1 -> L2: отбор реплик, проход модели, автопроверки.

Проход модели применяется только к репликам респондента достаточной длины.
Реплики интервьюера, наблюдателя и бэкканалы получают результат
детерминированного прохода и в модель не отправляются.

Ответ модели никогда не попадает в L2 без автопроверок: при любом расхождении
единица получает `status=rejected`, а в `text` остаётся детерминированный
результат — он и есть L2.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from qa.llm.base import LlmProvider
from qa.llm.errors import LlmError, LlmValidationError
from qa.pipeline.normalize_rules import (
    _SOUND_TOKEN_RE,
    FILLER_PHRASES,
    FuzzyConfig,
    LexiconEntry,
    comparison_tokens,
    is_known_word,
    is_service_word,
    lemma_of,
    normalize_deterministic,
    suspect_words,
)

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"

NORMALIZE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "normalized_text": {"type": "string"},
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "term",
                            "asr_fix",
                            "filler",
                            "stutter",
                            "order",
                            "punctuation",
                            "garbled",
                        ],
                    },
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                },
                "required": ["type", "from", "to"],
                "additionalProperties": False,
            },
        },
        "uncertain_spans": {"type": "array", "items": {"type": "string"}},
        "new_terms": {"type": "array", "items": {"type": "string"}},
        "asr_quality": {"type": "string", "enum": ["ok", "doubtful", "garbled"]},
    },
    "required": ["normalized_text", "changes", "uncertain_spans", "new_terms", "asr_quality"],
    "additionalProperties": False,
}

ASR_QUALITIES = ("ok", "doubtful", "garbled")

# Порог близости для молчаливого исправления ASR: модель заменила искажение
# известным словом, но не записала `asr_fix`. Ниже порога замена считается
# переписыванием смысла и остаётся жёстким отказом.
IMPLICIT_ASR_THRESHOLD = 0.80

_NEGATIONS = ("не", "нет", "ни", "без")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_WORD_RE = re.compile(r"[а-яёa-z]+(?:-[а-яёa-z]+)?", re.IGNORECASE)
# Начало предложения: после терминатора, двоеточия, тире, открывающей кавычки
# или скобки и перевода строки. Модель расставляет регистр по своим правилам
# пунктуации, поэтому заглавная в такой позиции не считается выдуманным словом.
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?…:—«(\[\n]")
_CAPITAL_WORD_RE = re.compile(r"[А-ЯЁA-Z][а-яёa-z\-]+")
_LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")


@dataclass(frozen=True)
class CheckFailure:
    """Причина отказа автопроверки."""

    check: str
    detail: str


@dataclass
class NormalizeOutcome:
    """Итог обработки одной реплики."""

    utterance_id: int
    text: str
    status: str
    asr_quality: str
    changes: list[dict[str, Any]] = field(default_factory=list)
    uncertain_spans: list[str] = field(default_factory=list)
    fillers_removed: int = 0
    stutter_removed: int = 0
    repeat_unclear: int = 0
    new_terms: list[str] = field(default_factory=list)
    failures: list[CheckFailure] = field(default_factory=list)
    proposed_text: str = ""
    used_model: bool = False
    model_used: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class NormalizeStats:
    """Счётчики прогона нормализации."""

    total: int = 0
    proposed: int = 0
    rejected: int = 0
    model_calls: int = 0
    skipped_model: int = 0
    asr_quality: dict[str, int] = field(default_factory=dict)
    fillers_removed: int = 0
    stutter_removed: int = 0
    repeat_unclear: int = 0
    term_replacements: dict[str, int] = field(default_factory=dict)
    rejections: dict[str, int] = field(default_factory=dict)
    new_terms: int = 0
    names_review: int = 0
    asr_fixes: int = 0
    implicit_asr_fixes: int = 0
    soft_thought_loss: int = 0
    fuzzy_replacements: int = 0
    glued_replacements: int = 0
    doubtful: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


def load_prompt(name: str = "normalize.md") -> str:
    """Прочитать файл промпта из `prompts/`."""
    path = PROMPTS_DIR / name
    return path.read_text(encoding="utf-8")


def load_examples() -> str:
    """Прочитать синтетические примеры для промпта."""
    return load_prompt("normalize_examples.md")


def prompt_hash(prompt: str) -> str:
    """Хеш промпта для записи в run."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def word_count(text: str) -> int:
    """Число слов в реплике."""
    return len(_WORD_RE.findall(text))


def should_use_model(role: str, is_backchannel: bool, text: str, min_words: int) -> bool:
    """Отправлять ли реплику в модель.

    Только реплики респондента: у интервьюера точная формулировка нужна как есть,
    бэкканалы смысла не несут.
    """
    if role != "RESPONDENT" or is_backchannel:
        return False
    return word_count(text) >= min_words


def build_prompt(
    utterance_text: str,
    previous_question: str | None,
    lexicon: list[LexiconEntry],
    markers: list[str],
    examples: str,
    suspect: list[str] | None = None,
    domain_hint: str = "",
) -> str:
    """Собрать промпт из шаблона и данных реплики."""
    template = load_prompt()
    lexicon_text = (
        "\n".join(
            f"- {entry.canonical}: {', '.join(entry.variants)}" if entry.variants else f"- {entry.canonical}"
            for entry in lexicon
        )
        or "- (лексикон пуст)"
    )
    marker_text = ", ".join(markers) or "(список пуст)"
    suspect_text = ", ".join(suspect or []) or "(нет)"
    return (
        template.replace("{previous_question}", previous_question or "(нет)")
        .replace("{domain_hint}", domain_hint or "(не задан)")
        .replace("{lexicon}", lexicon_text)
        .replace("{markers}", marker_text)
        .replace("{suspect}", suspect_text)
        .replace("{utterance}", utterance_text)
        .replace("{examples}", examples)
    )


def check_numbers(source: str, target: str) -> CheckFailure | None:
    """Проверка 1: все числа из L1 есть в L2."""
    missing = [value for value in _numbers(source) if value not in _numbers(target)]
    if missing:
        return CheckFailure("numbers", f"потеряны числа: {len(missing)}")
    return None


def check_negations(source: str, target: str) -> CheckFailure | None:
    """Проверка 2: все отрицания сохранены."""
    source_tokens = _tokens(source)
    target_tokens = _tokens(target)
    for negation in _NEGATIONS:
        if source_tokens.count(negation) > target_tokens.count(negation):
            return CheckFailure("negations", f"потеряно отрицание {negation!r}")
    return None


def check_uncertainty(source: str, target: str, markers: list[str]) -> CheckFailure | None:
    """Проверка 3: все маркеры неуверенности из входа есть в выходе."""
    source_folded = source.casefold()
    target_folded = target.casefold()
    for marker in markers:
        needle = marker.strip().casefold()
        if not needle:
            continue
        if source_folded.count(needle) > target_folded.count(needle):
            return CheckFailure("uncertainty", f"потерян маркер неуверенности {marker!r}")
    return None


def check_names(source: str, target: str, lexicon: list[LexiconEntry]) -> CheckFailure | None:
    """Проверка 4: латиница и заглавные слова не выдуманы моделью.

    Слово с заглавной буквы (кроме начала предложения) и любая латиница в L2
    должны встречаться во входе или в лексиконе. Термин лексикона может состоять
    из нескольких слов («Transaq Connector»): тогда его слова разрешены, если
    термин целиком присутствует в L2.
    """
    allowed = {entry.canonical.casefold() for entry in lexicon}
    for entry in lexicon:
        allowed.update(variant.casefold() for variant in entry.variants)
    target_folded = target.casefold()
    # Слова многословных терминов разрешены, только если термин есть в L2 целиком.
    for phrase in list(allowed):
        if " " in phrase and phrase in target_folded:
            allowed.update(phrase.split())
    source_folded = source.casefold()
    for word in _capitalized_words(target):
        folded = word.casefold()
        if folded in allowed or folded in source_folded:
            continue
        return CheckFailure("names", f"слово {word!r} отсутствует во входе и в лексиконе")
    return None


def check_capital_words(source: str, target: str, lexicon: list[LexiconEntry]) -> CheckFailure | None:
    """Мягкая часть проверки названий: заглавные слова без латиницы.

    Заглавное слово, которого нет во входе и в лексиконе с тем же регистром, —
    повод посмотреть единицу, а не повод её отклонить. Так ловится случай, когда
    модель капитализировала обычное слово («Приложением» вместо «приложением»).
    Латиница проверяется отдельно и жёстко.
    """
    allowed = {entry.canonical for entry in lexicon}
    for entry in lexicon:
        allowed.update(variant for variant in entry.variants)
    target_folded = target.casefold()
    for phrase in list(allowed):
        if " " in phrase.casefold() and phrase.casefold() in target_folded:
            allowed.update(phrase.split())
    source_words = set(_WORD_RE.findall(source))
    for word in _capitalized_words(target):
        if word in allowed or word in source:
            continue
        if word in source_words:
            continue
        return CheckFailure("names", f"слово {word!r} отсутствует во входе и в лексиконе")
    return None


def check_latin(source: str, target: str, lexicon: list[LexiconEntry]) -> CheckFailure | None:
    """Жёсткая часть проверки названий: латиница не выдумана моделью.

    Латиница проверяется отдельно от заглавных слов, потому что при отказе
    только по заглавным единица сохраняется как `doubtful`, а выдуманная
    латиница по-прежнему отклоняет единицу.
    """
    allowed = {entry.canonical.casefold() for entry in lexicon}
    for entry in lexicon:
        allowed.update(variant.casefold() for variant in entry.variants)
    target_folded = target.casefold()
    for phrase in list(allowed):
        if " " in phrase and phrase in target_folded:
            allowed.update(phrase.split())
    source_folded = source.casefold()
    for latin in _LATIN_RE.findall(target):
        folded = latin.casefold()
        if folded in allowed or folded in source_folded:
            continue
        return CheckFailure("latin", f"латиница {latin!r} отсутствует во входе и в лексиконе")
    return None


def _deleted_tokens(source: str, target: str) -> list[str]:
    """Слова L1, чьей леммы не стало в L2 (с учётом кратности).

    Сравнение по лемам: «ноутбук» -> «ноутбука» — это словоформа, а не потеря
    мысли. У неизвестного словарю искажения лемма равна форме, поэтому замена
    такого слова видна проверке. Токенизация — `comparison_tokens`: дефисные
    слова делятся на части, служебные сочетания и варианты союзов сводятся к
    одной записи. Без этого «робот-алгоритм» выглядел бы как потеря «робот» и
    «алгоритм», а «в виду» — как потеря «ввиду».
    """
    remaining = Counter(lemma_of(token) for token in comparison_tokens(target))
    missing: list[str] = []
    for token in comparison_tokens(source):
        lemma = lemma_of(token)
        if remaining.get(lemma, 0) > 0:
            remaining[lemma] -= 1
            continue
        missing.append(token)
    return missing


def _filler_span(text: str) -> bool:
    """Состоит ли фрагмент целиком из фраз-паразитов и звуков.

    Фразы вырезаются целиком, чтобы «как бы» не превращалось в два отдельных
    слова: каждое из них само по себе обычное. Остаток обязан быть пустым или
    звуками-паразитами.
    """
    remainder = text.casefold().replace("ё", "е")
    for phrase in sorted(FILLER_PHRASES, key=len, reverse=True):
        phrase_folded = phrase.casefold().replace("ё", "е")
        pattern = rf"(?<![а-яёa-z0-9]){re.escape(phrase_folded)}(?![а-яёa-z0-9])"
        remainder = re.sub(pattern, " ", remainder)
    return all(_SOUND_TOKEN_RE.match(token) for token in _tokens(remainder))


def _declared_removals(changes: list[dict[str, Any]]) -> Counter[str]:
    """Леммы слов, которые модель заявила как законно удалённые или заменённые.

    Удаление `to=""` и замена `from -> to` объясняют исчезновение слов из `from`.
    Записи `filler` и `stutter` принимаются только для настоящего паразита
    (включая фразовые) или механического повтора: иначе модель могла бы
    объявить удаление любого слова.
    """
    removed: Counter[str] = Counter()
    for change in changes:
        if not isinstance(change, dict):
            continue
        kind = change.get("type")
        if kind not in ("filler", "stutter", "term", "asr_fix", "order", "garbled"):
            continue
        from_text = str(change.get("from") or "")
        from_tokens = _tokens(from_text)
        if not from_tokens:
            continue
        if kind == "filler" and not _filler_span(from_text):
            continue
        if kind == "stutter" and from_tokens.count(from_tokens[0]) < 2:
            continue
        for token in from_tokens:
            removed[lemma_of(token)] += 1
    return removed


def _is_filler_token(token: str) -> bool:
    """Паразит из списка или звук-паразит: удалять можно без потери мысли."""
    folded = token.casefold()
    return folded in FILLER_PHRASES or bool(_SOUND_TOKEN_RE.match(folded))


def _phrase_filler_removals(source: str, target: str) -> Counter[str]:
    """Сколько вхождений каждого токена объясняется фразами-паразитами L1.

    Фразовый паразит («как бы», «то есть») состоит из обычных слов, поэтому
    по отдельности они не считаются паразитами. Каждое вхождение фразы в L1
    разрешает удалить её слова один раз. Вычитать вхождения в L2 нельзя:
    модель может удалить одну фразу и оставить другую, и подсчёт по позициям
    дал бы ложный отказ.
    """
    source_folded = source.casefold().replace("ё", "е")
    target_folded = target.casefold().replace("ё", "е")
    removed: Counter[str] = Counter()
    for phrase in sorted(FILLER_PHRASES, key=len, reverse=True):
        phrase_folded = phrase.casefold().replace("ё", "е")
        pattern = rf"(?<![а-яёa-z0-9]){re.escape(phrase_folded)}(?![а-яёa-z0-9])"
        missing = len(re.findall(pattern, source_folded)) - len(re.findall(pattern, target_folded))
        for _ in range(max(0, missing)):
            for token in _tokens(phrase):
                removed[lemma_of(token)] += 1
    return removed


def _mechanical_repeat_counts(source: str) -> Counter[str]:
    """Сколько раз слово повторяется подряд в L1 (сверх первого вхождения).

    Механический повтор служебного слова («и и», «они, они, они») можно
    схлопнуть: лишние вхождения не несут смысла. Ключ — лемма, как и в
    остальных проверках потери мысли. Повтор считается и для повторов фразы:
    «там уже всё, там уже всё настроено» — тот же механический дубль.
    """
    extra: Counter[str] = Counter()
    matches = list(_WORD_RE.finditer(source))
    previous = ""
    for match in matches:
        lemma = lemma_of(match.group(0))
        if lemma == previous:
            extra[lemma] += 1
        previous = lemma
    return extra


def _repeat_ngram_counts(source: str, size: int) -> Counter[str]:
    """Сколько раз n-грамм повторяется подряд (сверх первого вхождения).

    Механический дубль бывает не только из одного слова: «там уже всё, там уже
    всё» — повтор двух-трёх слов. Каждое вхождение повторённой фразы разрешает
    удалить её слова один раз, ключ — лемма.
    """
    tokens = comparison_tokens(source)
    extra: Counter[str] = Counter()
    position = 0
    while position <= len(tokens) - 2 * size:
        left = tuple(tokens[position : position + size])
        right = tuple(tokens[position + size : position + 2 * size])
        if left == right:
            for token in left:
                extra[lemma_of(token)] += 1
            position += size
        else:
            position += 1
    return extra


def _close_replacement(token: str, source: str, target: str) -> str | None:
    """Известное слово L2, которым модель молча заменила искажение.

    Модель исправляет ошибку ASR по контексту, но не всегда пишет `asr_fix` в
    `changes`. Проверка видит только исчезновение искажённого слова и
    отклоняет единицу, хотя смысл сохранён. Здесь ищется замена: исходное
    слово обязано быть неизвестно словарю (иначе это подмена обычной лексики),
    замена — известное слово, которого не было в L1, и совпадение не ниже
    `IMPLICIT_ASR_THRESHOLD`.

    Returns:
        Замена в нижнем регистре или None, если близкой замены нет.
    """
    folded = token.casefold()
    if is_known_word(token):
        return None
    source_tokens = set(comparison_tokens(source))
    candidates = [word for word in set(comparison_tokens(target)) if word not in source_tokens]
    best = ""
    best_score = 0.0
    for word in candidates:
        if not is_known_word(word):
            continue
        score = fuzz.ratio(folded, word) / 100
        if score > best_score:
            best, best_score = word, score
    if best_score < IMPLICIT_ASR_THRESHOLD:
        return None
    return best


def check_thought_loss(source: str, target: str, changes: list[dict[str, Any]]) -> CheckFailure | None:
    """Проверка 5: из L1 не исчезают смысловые слова.

    Удаление принимается, если слово — паразит из списка (или звук), если это
    лишнее вхождение подряд идущего повтора (в том числе повтора фразы), либо
    если лемма слова есть среди законных удалений из `changes`. Законность
    записи проверяет `_declared_removals`: `filler` только для настоящей
    фразы-паразита, `stutter` только для механического повтора, иначе модель
    могла бы объявить удаление любого слова.

    Потеря служебного слова (частица, союз, предлог, местоимение, вводное
    наречие) и молчаливое исправление искажения ASR — не повод отклонять
    единицу: такие случаи разбирает `analyze_thought_loss` и понижает её до
    `doubtful`. Эта функция возвращает только жёсткий отказ.
    """
    return analyze_thought_loss(source, target, changes).failure


@dataclass(frozen=True)
class ThoughtLoss:
    """Итог проверки потери мысли.

    `failure` — жёсткий отказ (потеря смыслового слова). `soft` — потерянные
    служебные слова, из-за которых единица становится `doubtful`.
    `implicit_fixes` — молчаливые исправления искажений ASR: пары
    «исходная форма -> замена», которые дописываются в `changes` как `asr_fix`.
    """

    failure: CheckFailure | None = None
    soft: tuple[str, ...] = ()
    implicit_fixes: tuple[tuple[str, str], ...] = ()


def analyze_thought_loss(source: str, target: str, changes: list[dict[str, Any]]) -> ThoughtLoss:
    """Разобрать исчезнувшие слова L1 на три класса.

    1. Объяснённые: паразиты, звуки, механические повторы, законные записи
       `changes` — пропускаются.
    2. Служебные — дают `doubtful`, но не отказ: «там», «к», «если» смысла
       не несут.
    3. Искажения ASR, для которых в L2 нашлась близкая известная замена, —
       тоже `doubtful`: модель исправила слово, но не записала `asr_fix`.

    Всё остальное — жёсткий отказ: потеряно смысловое слово либо искажение
    выброшено или заменено далёкой фразой.
    """
    deleted = _deleted_tokens(source, target)
    if not deleted:
        return ThoughtLoss()
    declared = _declared_removals(changes)
    declared.update(_phrase_filler_removals(source, target))
    repeats = _mechanical_repeat_counts(source)
    for size in (2, 3):
        for lemma, count in _repeat_ngram_counts(source, size).items():
            repeats[lemma] = max(repeats[lemma], count)

    soft: list[str] = []
    implicit: list[tuple[str, str]] = []
    hard: list[str] = []
    for token in deleted:
        if _is_filler_token(token):
            continue
        lemma = lemma_of(token)
        if repeats.get(lemma, 0) > 0:
            repeats[lemma] -= 1
            continue
        if declared.get(lemma, 0) > 0:
            declared[lemma] -= 1
            continue
        if is_service_word(token):
            soft.append(token)
            continue
        replacement = _close_replacement(token, source, target)
        if replacement is not None:
            implicit.append((token, replacement))
            continue
        hard.append(token)
    failure = None
    if hard:
        failure = CheckFailure("thought_loss", f"потеряны слова без основания: {len(hard)}")
    return ThoughtLoss(failure=failure, soft=tuple(soft), implicit_fixes=tuple(implicit))


def _allowed_new_words(lexicon: list[LexiconEntry]) -> set[str]:
    """Леммы канонов и вариантов лексикона, которые допустимы в L2."""
    allowed: set[str] = set()
    for entry in lexicon:
        allowed.add(lemma_of(entry.canonical))
        for variant in entry.variants:
            allowed.add(lemma_of(variant))
    return allowed


def check_new_words(
    source: str,
    target: str,
    lexicon: list[LexiconEntry],
    changes: list[dict[str, Any]],
) -> tuple[CheckFailure | None, bool]:
    """Проверка 6: в L2 нет новых смысловых слов.

    Сравнение по лемам: законная словоформа не считается новым словом.
    Returns:
        (жёсткий отказ, признак `doubtful`). Жёстко отклоняются только новые
        заглавные слова и латиница: это может быть выдуманное название.
        Остальные новые леммы дают флаг `doubtful`, а не отказ.
    """
    source_lemmas = {lemma_of(token) for token in _tokens(source)}
    allowed = source_lemmas | _allowed_new_words(lexicon)
    for change in changes:
        if not isinstance(change, dict):
            continue
        for token in _tokens(str(change.get("to") or "")):
            allowed.add(lemma_of(token))
    new_lemmas: list[str] = []
    for token in _tokens(target):
        lemma = lemma_of(token)
        if lemma in allowed:
            continue
        if token.casefold() in allowed:
            continue
        if lemma not in new_lemmas:
            new_lemmas.append(lemma)
    if not new_lemmas:
        return None, False
    capital_or_latin = [
        value for value in new_lemmas if any(char.isupper() for char in value) or bool(_LATIN_RE.search(value))
    ]
    if capital_or_latin:
        return CheckFailure("new_words", f"новые названия: {len(capital_or_latin)}"), True
    return None, True


def check_asr_fix(
    source: str,
    target: str,
    lexicon: list[LexiconEntry],
    changes: list[dict[str, Any]],
) -> CheckFailure | None:
    """Проверка `asr_fix`: исправлять по контексту можно только подозрительные слова.

    Подозрительное слово — то, которого нет ни в словаре pymorphy3, ни в
    лексиконе. Иначе модель могла бы подменить обычную лексику «по контексту»,
    а это уже переписывание смысла. Исправленное слово должно остаться в L2.
    """
    suspicious = set(suspect_words(source, lexicon))
    target_folded = target.casefold()
    for change in changes:
        if not isinstance(change, dict) or change.get("type") != "asr_fix":
            continue
        from_tokens = _tokens(str(change.get("from") or ""))
        if not from_tokens:
            return CheckFailure("asr_fix", "asr_fix без исходного слова")
        unknown = [token for token in from_tokens if token not in suspicious]
        if unknown:
            return CheckFailure("asr_fix", f"asr_fix для слова не из подозрительных: {len(unknown)}")
        to_text = str(change.get("to") or "").strip()
        if not to_text or to_text.casefold() not in target_folded:
            return CheckFailure("asr_fix", "исправление asr_fix отсутствует в L2")
    return None


@dataclass(frozen=True)
class CheckReport:
    """Итог автопроверок: жёсткие отказы и мягкие пометки.

    `failures` отклоняют единицу. `soft_thought_loss` и `implicit_asr_fixes`
    единицу не отклоняют: они переводят её в `doubtful`, чтобы человек
    посмотрел реплику. Разделение нужно, потому что потеря служебного слова
    или молчаливое исправление ASR — не выброшенный смысл.
    """

    failures: list[CheckFailure] = field(default_factory=list)
    soft_thought_loss: tuple[str, ...] = ()
    implicit_asr_fixes: tuple[tuple[str, str], ...] = ()


def run_checks(
    source: str,
    target: str,
    markers: list[str],
    lexicon: list[LexiconEntry],
    changes: list[dict[str, Any]] | None = None,
) -> list[CheckFailure]:
    """Выполнить жёсткие автопроверки вывода модели."""
    return run_checks_detailed(source, target, markers, lexicon, changes).failures


def run_checks_detailed(
    source: str,
    target: str,
    markers: list[str],
    lexicon: list[LexiconEntry],
    changes: list[dict[str, Any]] | None = None,
) -> CheckReport:
    """Выполнить автопроверки и вернуть жёсткие отказы вместе с мягкими пометками."""
    model_changes = changes or []
    failures: list[CheckFailure] = []
    for failure in (
        check_numbers(source, target),
        check_negations(source, target),
        check_uncertainty(source, target, markers),
        check_latin(source, target, lexicon),
        check_asr_fix(source, target, lexicon, model_changes),
    ):
        if failure is not None:
            failures.append(failure)
    thought = analyze_thought_loss(source, target, model_changes)
    if thought.failure is not None:
        failures.append(thought.failure)
    hard, _doubtful = check_new_words(source, target, lexicon, model_changes)
    if hard is not None:
        failures.append(hard)
    return CheckReport(
        failures=failures,
        soft_thought_loss=thought.soft,
        implicit_asr_fixes=thought.implicit_fixes,
    )


def normalize_utterance(
    utterance_id: int,
    role: str,
    is_backchannel: bool,
    text: str,
    *,
    previous_question: str | None,
    lexicon: list[LexiconEntry],
    markers: list[str],
    examples: str,
    provider: LlmProvider | None,
    min_words: int,
    fuzzy: FuzzyConfig | None = None,
    suspect: list[str] | None = None,
    domain_hint: str = "",
) -> NormalizeOutcome:
    """Нормализовать одну реплику.

    Args:
        utterance_id: идентификатор реплики L1.
        role: роль говорящего.
        is_backchannel: короткая вставка-подтверждение.
        text: исходный текст L1.
        previous_question: текст предыдущей реплики интервьюера.
        lexicon: термины с вариантами.
        markers: маркеры неуверенности.
        examples: синтетические примеры для промпта.
        provider: провайдер LLM; None — только детерминированный проход.
        min_words: минимальная длина реплики для вызова модели.
        fuzzy: пороги нечёткого сопоставления с лексиконом.
        suspect: подозрительные слова (нет в словаре и лексиконе) для промпта.
        domain_hint: строка предметной области для промпта.
    """
    # Нечёткая замена терминов — правило для реплик респондента. У интервьюера
    # формулировка нужна дословной, а его текст и так проходит точный лексикон.
    deterministic = normalize_deterministic(text, lexicon, fuzzy if role == "RESPONDENT" else None)
    if suspect is None:
        suspect = suspect_words(deterministic.text, lexicon)
    outcome = NormalizeOutcome(
        utterance_id=utterance_id,
        text=deterministic.text,
        status="proposed",
        asr_quality="ok",
        changes=list(deterministic.changes),
        fillers_removed=deterministic.fillers_removed,
        stutter_removed=deterministic.stutter_removed,
        repeat_unclear=deterministic.repeat_unclear,
    )

    if provider is None or not should_use_model(role, is_backchannel, text, min_words):
        return outcome

    prompt = build_prompt(
        deterministic.text,
        previous_question,
        lexicon,
        markers,
        examples,
        suspect,
        domain_hint,
    )
    try:
        result = provider.complete_json_detailed(prompt, NORMALIZE_SCHEMA)
    except LlmValidationError as exc:
        logger.warning("LLM: реплика %d не прошла валидацию JSON: %s", utterance_id, exc)
        outcome.status = "rejected"
        outcome.failures.append(CheckFailure("validation", "ответ модели не прошёл валидацию"))
        return outcome
    except LlmError as exc:
        logger.warning("LLM: вызов для реплики %d не удался: %s", utterance_id, exc)
        outcome.status = "rejected"
        outcome.failures.append(CheckFailure("transport", "вызов модели не удался"))
        return outcome

    outcome.used_model = True
    outcome.model_used = result.model_used
    outcome.prompt_tokens = result.prompt_tokens
    outcome.completion_tokens = result.completion_tokens
    return _apply_model_output(outcome, deterministic.text, result.data, markers, lexicon)


def _at_least_doubtful(quality: str) -> str:
    """Поднять оценку качества до `doubtful`, не понижая более тяжёлую.

    `garbled` означает невосстановимый фрагмент и строже `doubtful`: мягкая
    пометка (потеря служебного слова, молчаливое исправление ASR) не должна
    «улучшать» такую единицу.
    """
    return quality if quality in ("doubtful", "garbled") else "doubtful"


def _apply_model_output(
    outcome: NormalizeOutcome,
    deterministic_text: str,
    data: dict[str, Any],
    markers: list[str],
    lexicon: list[LexiconEntry],
) -> NormalizeOutcome:
    """Применить вывод модели, если он прошёл автопроверки.

    Отклонённый вывод сохраняется в `proposed_text`: без него нельзя понять,
    что именно предложила модель и почему проверка сработала.
    """
    normalized = str(data.get("normalized_text") or "").strip()
    outcome.proposed_text = normalized
    quality = str(data.get("asr_quality") or "ok")
    if quality not in ASR_QUALITIES:
        quality = "doubtful"
    outcome.asr_quality = quality
    outcome.uncertain_spans = [str(item) for item in data.get("uncertain_spans") or []]
    outcome.new_terms = [str(item).strip() for item in data.get("new_terms") or [] if str(item).strip()]

    model_changes = [item for item in data.get("changes") or [] if isinstance(item, dict)]
    report = run_checks_detailed(deterministic_text, normalized, markers, lexicon, model_changes)
    failures = list(report.failures)
    if not normalized:
        failures.append(CheckFailure("empty", "модель вернула пустой текст"))
    if failures:
        outcome.status = "rejected"
        outcome.failures = failures
        outcome.changes = [
            *outcome.changes,
            *[{"type": "rejected", "check": failure.check} for failure in failures],
            {"type": "model_output", "text": normalized},
        ]
        logger.info(
            "LLM: реплика %d отклонена автопроверкой (%s)",
            outcome.utterance_id,
            ",".join(failure.check for failure in failures),
        )
        return outcome

    outcome.changes = [*outcome.changes, *model_changes]
    outcome.text = normalized
    if report.soft_thought_loss:
        # Потеря служебных слов — сигнал человеку, а не отказ: «там», «к»,
        # «если» смысла не несут. В L2 остаётся текст модели.
        outcome.asr_quality = _at_least_doubtful(outcome.asr_quality)
        outcome.changes = [
            *outcome.changes,
            {
                "type": "soft_thought_loss",
                "words": list(report.soft_thought_loss),
            },
        ]
    if report.implicit_asr_fixes:
        # Модель исправила искажение ASR по контексту, но не записала замену.
        # Дописываем `asr_fix` сами: без записи следующий проход и ручная
        # проверка не увидят, что слово изменено.
        outcome.asr_quality = _at_least_doubtful(outcome.asr_quality)
        for source_word, replacement in report.implicit_asr_fixes:
            outcome.changes = [
                *outcome.changes,
                {"type": "asr_fix", "from": source_word, "to": replacement, "implicit": True},
            ]
    _hard, new_doubtful = check_new_words(deterministic_text, normalized, lexicon, model_changes)
    if new_doubtful:
        outcome.asr_quality = _at_least_doubtful(outcome.asr_quality)
    names_failure = check_capital_words(deterministic_text, normalized, lexicon)
    if names_failure is not None:
        outcome.asr_quality = _at_least_doubtful(outcome.asr_quality)
        outcome.changes = [*outcome.changes, {"type": "names_review", "detail": names_failure.detail}]
    return outcome


def _numbers(text: str) -> list[str]:
    """Числа в тексте в нормализованном виде."""
    return [value.replace(",", ".") for value in _NUMBER_RE.findall(text)]


def _tokens(text: str) -> list[str]:
    """Слова в нижнем регистре."""
    return [value.casefold() for value in _WORD_RE.findall(text)]


def _capitalized_words(text: str) -> list[str]:
    """Слова с заглавной буквы не в начале предложения, плюс латиница.

    Заглавная в начале предложения — следствие расстановки пунктуации моделью,
    а не выдуманное название: после «:», «—», ««», «(», перевода строки и
    терминаторов такие слова не проверяются. Латиница проверяется всегда.
    """
    found: list[str] = []
    for match in _CAPITAL_WORD_RE.finditer(text):
        prefix = text[: match.start()].rstrip()
        if not prefix or _SENTENCE_BOUNDARY_RE.fullmatch(prefix[-1]):
            continue
        found.append(match.group(0))
    for latin in _LATIN_RE.findall(text):
        if latin not in found:
            found.append(latin)
    return found


def surviving_new_terms(outcome: NormalizeOutcome) -> list[str]:
    """Кандидаты в новые термины, форма которых есть в тексте L2 или в выводе модели.

    Кандидат имеет смысл, только если человек увидит форму в тексте. Модель
    иногда присылает `new_terms` вместе с исправлением слова через `asr_fix`
    или с переписанным фрагментом: в L2 такой формы нет, и лексикон засоряется.

    Для отклонённой единицы в L2 остаётся детерминированный текст, а вывод
    модели — в `proposed_text`. Отклонённый вывод не попадает в L2, но
    искажение в нём осталось неисправленным, поэтому проверяются оба текста.
    """
    texts = [outcome.text.casefold()]
    if outcome.proposed_text:
        texts.append(outcome.proposed_text.casefold())
    return [
        term for term in outcome.new_terms if term.strip() and any(term.strip().casefold() in text for text in texts)
    ]


def summarize(outcomes: list[NormalizeOutcome]) -> NormalizeStats:
    """Собрать счётчики по результатам обработки."""
    quality: dict[str, int] = {}
    rejections: dict[str, int] = {}
    terms: dict[str, int] = {}
    names_review = 0
    asr_fixes = 0
    implicit_asr_fixes = 0
    soft_thought_loss = 0
    fuzzy_replacements = 0
    glued_replacements = 0
    for outcome in outcomes:
        quality[outcome.asr_quality] = quality.get(outcome.asr_quality, 0) + 1
        for failure in outcome.failures:
            rejections[failure.check] = rejections.get(failure.check, 0) + 1
        for change in outcome.changes:
            if change.get("type") == "term" and isinstance(change.get("to"), str):
                terms[change["to"]] = terms.get(change["to"], 0) + 1
                if change.get("fuzzy"):
                    fuzzy_replacements += 1
                if change.get("glued"):
                    glued_replacements += 1
            elif change.get("type") == "names_review":
                names_review += 1
            elif change.get("type") == "asr_fix":
                asr_fixes += 1
                if change.get("implicit"):
                    implicit_asr_fixes += 1
            elif change.get("type") == "soft_thought_loss":
                soft_thought_loss += 1
    return NormalizeStats(
        total=len(outcomes),
        proposed=sum(1 for item in outcomes if item.status == "proposed"),
        rejected=sum(1 for item in outcomes if item.status == "rejected"),
        model_calls=sum(1 for item in outcomes if item.used_model),
        skipped_model=sum(1 for item in outcomes if not item.used_model),
        asr_quality=quality,
        fillers_removed=sum(item.fillers_removed for item in outcomes),
        stutter_removed=sum(item.stutter_removed for item in outcomes),
        repeat_unclear=sum(item.repeat_unclear for item in outcomes),
        term_replacements=terms,
        rejections=rejections,
        new_terms=sum(len(surviving_new_terms(item)) for item in outcomes),
        names_review=names_review,
        asr_fixes=asr_fixes,
        implicit_asr_fixes=implicit_asr_fixes,
        soft_thought_loss=soft_thought_loss,
        fuzzy_replacements=fuzzy_replacements,
        glued_replacements=glued_replacements,
        doubtful=quality.get("doubtful", 0),
        prompt_tokens=sum(item.prompt_tokens for item in outcomes),
        completion_tokens=sum(item.completion_tokens for item in outcomes),
    )


def estimate_dry_run(texts: list[str], previous: str | None = None) -> dict[str, int]:
    """Оценка объёма работы без вызовов модели."""
    examples_len = len(load_examples()) if (PROMPTS_DIR / "normalize_examples.md").is_file() else 0
    prompt_tokens = 0
    for text in texts:
        prompt_tokens += max(1, (len(text) + examples_len + len(previous or "")) // 4)
    return {"calls": len(texts), "prompt_tokens": prompt_tokens}


def dump_json(value: Any) -> str:  # noqa: ANN401
    """Сериализовать журнал изменений для БД."""
    return json.dumps(value, ensure_ascii=False)
