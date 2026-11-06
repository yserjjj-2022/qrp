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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qa.llm.base import LlmProvider
from qa.llm.errors import LlmError, LlmValidationError
from qa.pipeline.normalize_rules import LexiconEntry, normalize_deterministic

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
                        "enum": ["term", "filler", "stutter", "order", "punctuation", "garbled"],
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

_NEGATIONS = ("не", "нет", "ни", "без")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_WORD_RE = re.compile(r"[а-яёa-z]+(?:-[а-яёa-z]+)?", re.IGNORECASE)
_CAPITAL_RE = re.compile(r"(?<![.!?…]\s)(?<!^)\b([А-ЯЁA-Z][а-яёa-z\-]+)")
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
    return (
        template.replace("{previous_question}", previous_question or "(нет)")
        .replace("{lexicon}", lexicon_text)
        .replace("{markers}", marker_text)
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
    должны встречаться во входе или быть канонической формой лексикона.
    """
    allowed = {entry.canonical.casefold() for entry in lexicon}
    for entry in lexicon:
        allowed.update(variant.casefold() for variant in entry.variants)
    source_folded = source.casefold()
    for word in _capitalized_words(target):
        folded = word.casefold()
        if folded in allowed or folded in source_folded:
            continue
        return CheckFailure("names", f"слово {word!r} отсутствует во входе и в лексиконе")
    return None


def run_checks(
    source: str,
    target: str,
    markers: list[str],
    lexicon: list[LexiconEntry],
) -> list[CheckFailure]:
    """Выполнить все автопроверки вывода модели."""
    failures: list[CheckFailure] = []
    for failure in (
        check_numbers(source, target),
        check_negations(source, target),
        check_uncertainty(source, target, markers),
        check_names(source, target, lexicon),
    ):
        if failure is not None:
            failures.append(failure)
    return failures


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
) -> NormalizeOutcome:
    """Нормализовать одну реплику.

    Args:
        utterance_id: идентификатор реплики L1.
        role: роль говорящего.
        is_backchannel: короткая вставка-подтверждение.
        text: исходный текст L1.
        previous_text: текст предыдущей реплики интервьюера.
        lexicon: термины с вариантами.
        markers: маркеры неуверенности.
        examples: синтетические примеры для промпта.
        provider: провайдер LLM; None — только детерминированный проход.
        min_words: минимальная длина реплики для вызова модели.
    """
    deterministic = normalize_deterministic(text, lexicon)
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

    prompt = build_prompt(deterministic.text, previous_question, lexicon, markers, examples)
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


def _apply_model_output(
    outcome: NormalizeOutcome,
    deterministic_text: str,
    data: dict[str, Any],
    markers: list[str],
    lexicon: list[LexiconEntry],
) -> NormalizeOutcome:
    """Применить вывод модели, если он прошёл автопроверки."""
    normalized = str(data.get("normalized_text") or "").strip()
    quality = str(data.get("asr_quality") or "ok")
    if quality not in ASR_QUALITIES:
        quality = "doubtful"
    outcome.asr_quality = quality
    outcome.uncertain_spans = [str(item) for item in data.get("uncertain_spans") or []]
    outcome.new_terms = [str(item).strip() for item in data.get("new_terms") or [] if str(item).strip()]

    failures = run_checks(deterministic_text, normalized, markers, lexicon)
    if not normalized:
        failures.append(CheckFailure("empty", "модель вернула пустой текст"))
    if failures:
        outcome.status = "rejected"
        outcome.failures = failures
        outcome.changes = [*outcome.changes, *[{"type": "rejected", "check": item.check} for item in failures]]
        logger.info(
            "LLM: реплика %d отклонена автопроверкой (%s)",
            outcome.utterance_id,
            ",".join(item.check for item in failures),
        )
        return outcome

    model_changes = data.get("changes")
    if isinstance(model_changes, list):
        outcome.changes = [*outcome.changes, *[item for item in model_changes if isinstance(item, dict)]]
    outcome.text = normalized
    return outcome


def _numbers(text: str) -> list[str]:
    """Числа в тексте в нормализованном виде."""
    return [value.replace(",", ".") for value in _NUMBER_RE.findall(text)]


def _tokens(text: str) -> list[str]:
    """Слова в нижнем регистре."""
    return [value.casefold() for value in _WORD_RE.findall(text)]


def _capitalized_words(text: str) -> list[str]:
    """Слова с заглавной буквы не в начале предложения, плюс латиница."""
    found = list(_CAPITAL_RE.findall(text))
    for latin in _LATIN_RE.findall(text):
        if latin not in found:
            found.append(latin)
    return found


def summarize(outcomes: list[NormalizeOutcome]) -> NormalizeStats:
    """Собрать счётчики по результатам обработки."""
    quality: dict[str, int] = {}
    rejections: dict[str, int] = {}
    terms: dict[str, int] = {}
    for outcome in outcomes:
        quality[outcome.asr_quality] = quality.get(outcome.asr_quality, 0) + 1
        for failure in outcome.failures:
            rejections[failure.check] = rejections.get(failure.check, 0) + 1
        for change in outcome.changes:
            if change.get("type") == "term" and isinstance(change.get("to"), str):
                terms[change["to"]] = terms.get(change["to"], 0) + 1
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
        new_terms=sum(len(item.new_terms) for item in outcomes),
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
