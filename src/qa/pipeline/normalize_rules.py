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
from functools import lru_cache
from typing import Any

import pymorphy3
from rapidfuzz import fuzz

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

# Слово как токен: буквы, дефис внутри. Тот же шаблон, что у проверок L2.
_WORD_TOKEN_RE = re.compile(r"[а-яёa-z]+(?:-[а-яёa-z]+)?", re.IGNORECASE)

# Паразиты, которые разрешено удалять без потери смысла. Список совпадает с
# промптом: удаление любого другого слова — потеря мысли, а не нормализация.
FILLER_PHRASES: frozenset[str] = frozenset(
    {
        "вот",
        "как бы",
        "короче",
        "ну",
        "скажем так",
        "типа",
        "то есть",
        "это самое",
    }
)

# Звуки-паразиты после токенизации: «э», «ээ», «м», «мм», «аа», «оо», «э-э».
_SOUND_TOKEN_RE = re.compile(r"^(?:э+|м+|аа+|оо+)(?:[- ](?:э+|м+|аа+|оо+))*$", re.IGNORECASE)

# Служебные части речи: потеря такого слова — шум речи, а не потеря мысли.
# Набор намеренно широкий: предлоги, союзы, частицы, местоимения, предикативы
# и междометия. Проверка потери мысли понижает такую единицу до `doubtful`,
# а не отклоняет: смысл фразы от пропуска «там» или «если» не рушится.
SERVICE_POS: frozenset[str] = frozenset({"PREP", "CONJ", "PRCL", "NPRO", "PRTS", "INTJ"})

# Местоименные прилагательные: pymorphy3 размечает их как ADJF/ADJS, но по
# смыслу это местоимения («какой», «такой», «весь»). Ключ — нормальная форма.
_PRONOUN_ADJECTIVES: frozenset[str] = frozenset(
    {
        "какой",
        "какой-то",
        "который",
        "такой",
        "весь",
        "этот",
        "свой",
        "наш",
        "ваш",
        "мой",
        "твой",
        "чей",
        "никакой",
        "некоторый",
        "каждый",
        "любой",
        "другой",
        "иной",
        "самый",
        "сам",
    }
)

# Дискурсивные наречия: вводные слова и заполнители, смысловой нагрузки не несут.
_DISCOURSE_ADVERBS: frozenset[str] = frozenset(
    {
        "там",
        "тут",
        "вот",
        "просто",
        "вообще",
        "уже",
        "ещё",
        "еще",
        "прямо",
        "так",
        "тогда",
        "потом",
        "значит",
        "короче",
        "может",
        "наверное",
        "по-моему",
        "как-то",
        "что-то",
        "скажем",
    }
)

# Слитно/раздельно: варианты записи служебных сочетаний. Модель ставит пробел
# там, где в L1 слово слитное (и наоборот), поэтому при сравнении потерь обе
# записи сводятся к одной форме. Это только ключ сравнения, текст не меняется.
_SPLIT_SERVICE_MERGES: tuple[tuple[str, str], ...] = (
    ("в виду", "ввиду"),
    ("в виде", "ввиде"),
    ("на счёт", "насчёт"),
    ("не смотря на", "несмотря"),
)

# Варианты союзов, различающиеся только написанием: «что» / «чтоб» / «чтобы».
_CONJUNCTION_VARIANTS: dict[str, str] = {"чтоб": "что", "чтобы": "что"}


@lru_cache(maxsize=16384)
def is_service_word(word: str) -> bool:
    """Служебное ли слово: частица, союз, предлог, местоимение, вводное наречие.

    Служит для мягкой оценки потери мысли: пропуск служебного слова — шум
    речи, а не выброшенный смысл. Проверка идёт по разбору pymorphy3 с
    расширением на местоименные прилагательные и дискурсивные наречия,
    которые размечаются как ADJF/ADVB, но смысла не несут.
    """
    if not word:
        return False
    folded = _fold(word)
    if folded in FUNCTION_WORDS or folded in _DISCOURSE_ADVERBS:
        return True
    parsed = morph_analyzer().parse(folded)[0]
    if parsed.tag.POS in SERVICE_POS:
        return True
    return parsed.tag.POS in ("ADJF", "ADJS") and str(parsed.normal_form) in _PRONOUN_ADJECTIVES


def comparison_tokens(text: str) -> list[str]:
    """Токены для сравнения потерь, устойчивые к разнице записи.

    Отличия от обычной токенизации:
    - дефисные слова делятся на части («робот-алгоритм» = «робот» + «алгоритм»),
      иначе модель, поставившая дефис, выглядит как потерявшая оба слова;
    - служебные сочетания сводятся к одной записи («в виду» = «ввиду»);
    - варианты союзов сводятся к одной форме («чтоб» = «что»).

    Регистр и «ё» не различаются. Текст реплики функция не меняет: это только
    ключ сравнения для проверки потери мысли.
    """
    folded = _fold(text)
    for phrase, merged in _SPLIT_SERVICE_MERGES:
        pattern = rf"(?<![а-яёa-z0-9]){re.escape(phrase)}(?![а-яёa-z0-9])"
        folded = re.sub(pattern, merged, folded)
    tokens: list[str] = []
    for match in _WORD_TOKEN_RE.finditer(folded):
        for part in match.group(0).split("-"):
            if part:
                tokens.append(_CONJUNCTION_VARIANTS.get(part, part))
    return tokens


@dataclass(frozen=True)
class LexiconEntry:
    """Термин лексикона: каноническая форма и варианты ASR-искажений."""

    canonical: str
    variants: tuple[str, ...] = ()


@dataclass(frozen=True)
class FuzzyConfig:
    """Пороги нечёткого сопоставления с лексиконом.

    Для слов короче `short_word_length` действует более строгий порог: на
    коротких строках случайное совпадение вероятнее.
    """

    threshold: float = 0.80
    short_threshold: float = 0.93
    short_word_length: int = 5


@dataclass
class DeterministicResult:
    """Итог детерминированного прохода."""

    text: str
    changes: list[dict[str, Any]] = field(default_factory=list)
    fillers_removed: int = 0
    stutter_removed: int = 0
    repeat_unclear: int = 0
    term_replacements: dict[str, int] = field(default_factory=dict)
    fuzzy_replacements: int = 0
    glued_replacements: int = 0


@lru_cache(maxsize=1)
def morph_analyzer() -> pymorphy3.MorphAnalyzer:
    """Общий анализатор pymorphy3: словари читаются один раз за процесс."""
    return pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=16384)
def is_known_word(word: str) -> bool:
    """Известно ли слово словарю pymorphy3.

    Неизвестное слово — кандидат в искажение ASR: словарь его не знает, и
    обычная морфология к нему не применяется.
    """
    if not word:
        return False
    return bool(morph_analyzer().parse(word)[0].is_known)


@lru_cache(maxsize=16384)
def lemma_of(word: str) -> str:
    """Лемма слова по pymorphy3; неизвестное слово возвращается как есть.

    Сравнение по лемам, а не по формам: законная словоформа («акции» при
    «акция») не считается новым словом. У неизвестного слова лемма
    ненадёжна, поэтому берётся сама форма — так искажение ASR видно проверке.
    """
    if not word:
        return ""
    folded = word.casefold().replace("ё", "е")
    parsed = morph_analyzer().parse(folded)[0]
    if not parsed.is_known:
        return folded
    return str(parsed.normal_form)


def _fold(value: str) -> str:
    """Ключ сравнения: регистр и «ё» не различаются."""
    return value.casefold().replace("ё", "е")


def lexicon_forms(entries: list[LexiconEntry]) -> dict[str, str]:
    """Карта «форма лексикона (folded) -> каноническая форма»."""
    forms: dict[str, str] = {}
    for entry in entries:
        forms.setdefault(_fold(entry.canonical), entry.canonical)
        for variant in entry.variants:
            value = variant.strip()
            if value:
                forms.setdefault(_fold(value), entry.canonical)
    return forms


def _compact_forms(forms: dict[str, str]) -> dict[str, str]:
    """Формы без пробелов: для сравнения со склеенной парой слов."""
    return {key.replace(" ", ""): canonical for key, canonical in forms.items()}


def suspect_words(text: str, entries: list[LexiconEntry]) -> list[str]:
    """Слова, которых нет ни в словаре pymorphy3, ни в лексиконе.

    Такие слова — кандидаты в искажения ASR: список передаётся модели, чтобы
    она могла исправить очевидное по контексту через `asr_fix`. Сравнение с
    лексиконом идёт по целым терминам и по отдельным словам многословных
    терминов, и по lowercase-формам, и по лемам. Звуки-паразиты, слова короче
    трёх букв и числа в список не попадают; числа отсекает сама токенизация.

    Returns:
        Уникальные слова в нижнем регистре в порядке первого появления.
    """
    allowed: set[str] = set()
    forms: list[str] = []
    for entry in entries:
        for phrase in (entry.canonical, *entry.variants):
            value = phrase.strip()
            if not value:
                continue
            folded_phrase = _fold(value)
            allowed.add(folded_phrase)
            forms.append(folded_phrase)
            for word in _WORD_TOKEN_RE.findall(value):
                folded_word = _fold(word)
                allowed.add(folded_word)
                allowed.add(_fold(lemma_of(word)))
                forms.append(folded_word)

    found: list[str] = []
    seen: set[str] = set()
    for match in _WORD_TOKEN_RE.finditer(text):
        token = match.group(0)
        folded = _fold(token)
        if len(token) < 3 or folded in seen:
            continue
        if _SOUND_TOKEN_RE.match(folded):
            continue
        if is_known_word(token):
            continue
        if folded in allowed or _fold(lemma_of(token)) in allowed:
            continue
        # Словоформа термина: у неизвестного словарю слова лемма ненадёжна,
        # поэтому сравниваются и нормальная форма pymorphy3, и падежное
        # окончание относительно формы термина.
        if _fold(str(morph_analyzer().parse(folded)[0].normal_form)) in allowed:
            continue
        if any(" " not in form and _is_word_form(folded, form) for form in forms):
            continue
        seen.add(folded)
        found.append(folded)
    return found


def _best_match(value: str, candidates: dict[str, str], threshold: float) -> tuple[str, str] | None:
    """Ближайшая форма лексикона не ниже порога: (канон, исходная форма)."""
    best_score = 0.0
    best: tuple[str, str] | None = None
    for form, canonical in candidates.items():
        score = fuzz.ratio(value, form) / 100
        if score > best_score:
            best_score = score
            best = (canonical, form)
    if best is None or best_score < threshold:
        return None
    return best


def _fuzzy_threshold(value: str, config: FuzzyConfig) -> float:
    """Порог для строки: короткие слова сравниваются строже."""
    return config.short_threshold if len(value) < config.short_word_length else config.threshold


def _fuzzy_allowed(token: str, forms: dict[str, str]) -> bool:
    """Можно ли нечётко заменять слово.

    Слова, известные словарю, не трогаем: это обычная лексика, а не искажение
    ASR. Точные формы лексикона уже заменены детерминированно.
    """
    folded = _fold(token)
    if folded in forms:
        return False
    return not is_known_word(token)


def _pair_allowed(left: str, right: str, forms: dict[str, str]) -> bool:
    """Годится ли пара слов для склейки.

    Склейка допустима, когда левое слово может быть приставкой (короткое и не
    служебное: «ре» + «балансланировка») либо оба слова неизвестны словарю.
    Служебное слово слева («и», «с», «в», «про») склейку запрещает: иначе
    «и эмсипи» превратилось бы в термин, а «с фнами» — в «Финам» с лишней «с».
    """
    if _fold(left) in forms or _fold(right) in forms:
        return False
    if left.casefold() in FUNCTION_WORDS:
        return False
    # Короткая приставка перед искажённым словом: «ре» + «балансланировка».
    if len(left) <= 3:
        return True
    return not is_known_word(left) and not is_known_word(right)


def _match_case(source: str, replacement: str) -> str:
    """Сохранить заглавную букву начала предложения при замене.

    Канон с латиницей в верхнем регистре («MCP», «API») не меняется: иначе
    получилось бы «Mcp». Замена строчными словами капитализируется, если
    исходное слово стояло с заглавной.
    """
    if not source or not replacement or not source[0].isupper():
        return replacement
    if any(char.isupper() for char in replacement):
        return replacement
    return replacement[0].upper() + replacement[1:]


# Короткие окончания существительных: канон + такое окончание — это словоформа
# канона («Финаме», «Финама»), а не искажение ASR. Замена сломала бы падеж.
_INFLECTION_TAILS = frozenset({"а", "я", "у", "ю", "е", "и", "ы", "о", "ом", "ем", "ой", "ей", "ах", "ях", "ам", "ям"})


def _is_word_form(surface: str, canonical: str) -> bool:
    """Является ли слово словоформой канона (а не искажением).

    Так отсекаются падежные формы канона, если оператор добавил их в текст:
    нечёткая замена не должна превращать «Финама» в «Финам».
    """
    folded_surface = _fold(surface)
    folded_canonical = _fold(canonical)
    if not folded_surface.startswith(folded_canonical) or folded_surface == folded_canonical:
        return False
    return folded_surface[len(folded_canonical) :] in _INFLECTION_TAILS


def _inflect_like(canonical: str, surface: str) -> str:
    """Поставить канон в ту же форму, что и заменяемое слово.

    Согласование допускается только для обычных существительных: канон
    оканчивается на гласную или «ь», части речи и род совпадают. Бренды и
    латиница не склоняются («Финам» разбирается как форма слова «фина», и
    «в Финаме» превратилось бы в «в Фине»). При любом сомнении возвращается
    канон как есть.
    """
    if " " in canonical or not canonical:
        return canonical
    if canonical[-1].casefold() not in "аяоуеиыюь":
        return canonical
    analyzer = morph_analyzer()
    surface_parse = analyzer.parse(surface)[0]
    canonical_parse = analyzer.parse(canonical)[0]
    if surface_parse.tag.POS != "NOUN" or canonical_parse.tag.POS != "NOUN":
        return canonical
    for grammeme in ("femn", "masc", "neut"):
        if surface_parse.tag.grammemes.issuperset({grammeme}) != canonical_parse.tag.grammemes.issuperset({grammeme}):
            return canonical
    target = surface_parse.tag.grammemes & {"sing", "plur", "nomn", "gent", "datv", "accs", "ablt", "loct"}
    if not target:
        return canonical
    inflected = canonical_parse.inflect(target)
    return inflected.word if inflected is not None else canonical


def apply_fuzzy_lexicon(
    text: str,
    entries: list[LexiconEntry],
    config: FuzzyConfig | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
    """Заменить близкие к лексикону слова канонической формой.

    Сравнение идёт по нормализованному расстоянию (rapidfuzz): регистр и «ё»
    не различаются. Заменяются только слова, неизвестные словарю pymorphy3.
    Склейка соседних слов — отдельное правило `apply_glue`.
    """
    cfg = config or FuzzyConfig()
    forms = lexicon_forms(entries)
    if not forms:
        return text, [], {}
    matches = list(_WORD_TOKEN_RE.finditer(text))
    if not matches:
        return text, [], {}

    changes: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    pieces: list[str] = []
    cursor = 0
    for match in matches:
        token = match.group(0)
        if not _fuzzy_allowed(token, forms):
            continue
        best = _best_match(_fold(token), forms, _fuzzy_threshold(_fold(token), cfg))
        if best is None or _is_word_form(token, best[0]):
            continue
        replacement = _inflect_like(best[0], token)
        pieces.append(text[cursor : match.start()])
        pieces.append(_match_case(token, replacement))
        cursor = match.end()
        changes.append(
            {
                "type": "term",
                "from": token,
                "to": replacement,
                "fuzzy": True,
            }
        )
        counts[replacement] = counts.get(replacement, 0) + 1
    if not changes:
        return text, [], {}
    pieces.append(text[cursor:])
    return "".join(pieces), changes, counts


def apply_glue(
    text: str,
    entries: list[LexiconEntry],
    config: FuzzyConfig | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
    """Склеить соседние слова, если вместе они дают термин лексикона.

    Расшифровка иногда разделяет термин пробелом: «ре балансланировка»,
    «транзак коннектор». Пара заменяется каноном, только если склеенная форма
    близка к форме лексикона и левое слово не служебное. Это отдельное правило:
    в журнале запись помечается `glued=true`, чтобы отличать склейку от замены
    одного слова.
    """
    cfg = config or FuzzyConfig()
    forms = lexicon_forms(entries)
    if not forms:
        return text, [], {}
    compact = _compact_forms(forms)
    matches = list(_WORD_TOKEN_RE.finditer(text))
    if len(matches) < 2:
        return text, [], {}

    changes: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    pieces: list[str] = []
    cursor = 0
    index = 0
    while index + 1 < len(matches):
        first, second = matches[index], matches[index + 1]
        if text[first.end() : second.start()] != " ":
            index += 1
            continue
        left, right = first.group(0), second.group(0)
        if not _pair_allowed(left, right, forms):
            index += 1
            continue
        joined = _fold(left) + _fold(right)
        best = _best_match(joined, compact, _fuzzy_threshold(joined, cfg))
        if best is None:
            index += 1
            continue
        start = first.start()
        # Пара уже записана каноном точно: склейка только продублировала бы
        # запись. Регистр канона при этом не трогаем.
        if text[start : second.end()] == best[0]:
            index += 2
            continue
        # Форму задаёт второе слово: «ре балансланировку» -> «ребалансировку».
        replacement = _match_case(left, _inflect_like(best[0], right))
        pieces.append(text[cursor:start])
        pieces.append(replacement)
        cursor = second.end()
        changes.append(
            {
                "type": "term",
                "from": text[start:cursor],
                "to": replacement,
                "glued": True,
            }
        )
        counts[best[0]] = counts.get(best[0], 0) + 1
        index += 2
    if not changes:
        return text, [], {}
    pieces.append(text[cursor:])
    return "".join(pieces), changes, counts


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


def normalize_deterministic(
    text: str,
    entries: list[LexiconEntry],
    fuzzy: FuzzyConfig | None = None,
) -> DeterministicResult:
    """Выполнить полный детерминированный проход по одной реплике.

    Нечёткое сопоставление с лексиконом идёт сразу после точных замен: точные
    варианты уже заменены, поэтому нечёткий проход видит только оставшиеся
    искажения. Это не вызов модели, поэтому правило работает и на коротких
    репликах, которые в модель не отправляются.
    """
    changes: list[dict[str, Any]] = []

    result, term_changes, term_counts = apply_lexicon(text, entries)
    changes.extend(term_changes)

    # Склейка идёт до замены слов: иначе «ре» + «балансланировка» сначала
    # превратилось бы в «ре» + «ребалансировка».
    result, glue_changes, glue_counts = apply_glue(result, entries, fuzzy)
    changes.extend(glue_changes)
    for canonical, count in glue_counts.items():
        term_counts[canonical] = term_counts.get(canonical, 0) + count

    result, fuzzy_changes, fuzzy_counts = apply_fuzzy_lexicon(result, entries, fuzzy)
    changes.extend(fuzzy_changes)
    for canonical, count in fuzzy_counts.items():
        term_counts[canonical] = term_counts.get(canonical, 0) + count

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
        fuzzy_replacements=sum(fuzzy_counts.values()),
        glued_replacements=sum(glue_counts.values()),
    )
