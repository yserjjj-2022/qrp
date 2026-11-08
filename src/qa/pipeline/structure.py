"""Стадия structure: привязка реплик к вопросам гайда и разбиение L2 на фрагменты.

Вход — уже загруженное интервью (M1a) и записанный слой L2 (M1b). Выход —
заполненные `utterance.question_id`, `utterance.is_probe`,
`utterance.prior_interviewer_turn_id` и строки `fragment`.

Границы фрагментов предлагает модель, но в БД они попадают только после
проверки кодом: каждый фрагмент обязан дословно совпадать со срезом текста L2,
фрагменты одной реплики не пересекаются и вместе покрывают текст. Не прошедший
проверку ответ модели отбрасывается целиком для этой реплики, и применяется
детерминированное деление по предложениям.

Реплика вне гайда получает `question_id = NULL`: «ближайший вопрос» не
подставляется.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from qa.db.workbook_repo import (
    FragmentRow,
    GuideQuestion,
    InterviewRepository,
    RunRepository,
    StructureRepository,
    TextUnitRepository,
    UtteranceWithL2,
)
from qa.llm.base import LlmProvider
from qa.llm.errors import LlmError, LlmValidationError
from qa.settings import Settings

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"

BIND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "block": {"type": "integer"},
                    "question": {"type": ["string", "null"]},
                    "is_probe": {"type": "boolean"},
                },
                "required": ["block", "question", "is_probe"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["blocks"],
    "additionalProperties": False,
}

SEGMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "fragments": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["fragments"],
    "additionalProperties": False,
}

# Кандидаты в промпте привязки: сколько ближайших вопросов показывать модели.
CANDIDATES_PER_BLOCK = 3
# Целевая длина фрагмента при детерминированном делении по предложениям.
# Это не предел: жёсткая граница задана `m1c.max_fragment_chars` и проверяется
# в `validate_spans`. Смешивать их нельзя — fallback группировал бы предложения
# до 1200 символов и почти не резал реплики.
# Значение откалибровано по прогону run 13 (R01, L2 v8): при 140 получается
# 216 фрагментов против 217 у модели и средняя длина 111.8 против 111.3.
_FALLBACK_TARGET_CHARS = 140
_SENTENCE_END_RE = re.compile(r"[.!?…]+(?=\s|$)")


class StructureError(Exception):
    """Привязка невозможна: нет интервью, реплик или L2."""


@dataclass(frozen=True)
class FragmentSpan:
    """Границы фрагмента в тексте L2."""

    start: int
    end: int

    def text_of(self, text: str) -> str:
        """Текст фрагмента (срез L2)."""
        return text[self.start : self.end]


@dataclass(frozen=True)
class Block:
    """Блок «вопрос(ы) интервьюера -> ответ респондента».

    Единица привязки — не отдельная реплика: интервьюер часто задаёт вопрос
    двумя-тремя репликами («Угу. А почему?»), и только вместе с ответом
    понятно, какой вопрос гайда прозвучал. Чистые подтверждения («Угу»)
    своего блока не образуют: они примыкают к соседнему вопросу.
    """

    index: int
    interviewer_ids: tuple[int, ...]
    respondent_ids: tuple[int, ...]
    question_text: str
    answer_text: str
    question_is_backchannel: bool = False
    answer_is_backchannel: bool = True


def build_blocks(utterances: list[UtteranceWithL2]) -> list[Block]:
    """Собрать блоки «вопрос(ы) -> ответ» из реплик интервью.

    Правила:
    1. Подряд идущие реплики интервьюера — один вопрос. Две реплики подряд
       («Угу. А почему решили подключить?») не делятся на два вопроса.
    2. Блок из одних подтверждений («Угу») вопросом не считается: он
       примыкает к следующему блоку интервьюера, а в конце интервью — к
       предыдущему.
    3. Ответ респондента закрепляет накопленные реплики интервьюера как блок.
    4. Хвост интервьюера без ответа остаётся блоком с пустым ответом: это
       может быть вопрос гайда, на который не ответили.
    """
    blocks: list[Block] = []
    pending: list[UtteranceWithL2] = []

    def flush(respondent: list[UtteranceWithL2]) -> None:
        if not pending and not respondent:
            return
        blocks.append(
            Block(
                index=len(blocks) + 1,
                interviewer_ids=tuple(item.id for item in pending),
                respondent_ids=tuple(item.id for item in respondent),
                question_text=" ".join(item.l2_text or item.text for item in pending).strip(),
                answer_text=" ".join(item.l2_text or item.text for item in respondent).strip(),
                question_is_backchannel=bool(pending) and all(item.is_backchannel for item in pending),
                answer_is_backchannel=all(item.is_backchannel for item in respondent),
            )
        )
        pending.clear()

    for item in utterances:
        if item.speaker_role == "INTERVIEWER":
            pending.append(item)
            continue
        if item.speaker_role == "RESPONDENT":
            if item.is_backchannel and not pending:
                # Подтверждение респондента без вопроса: примыкает к прошлому блоку.
                if blocks:
                    blocks[-1] = _add_respondent(blocks[-1], item)
                continue
            flush([item])
            continue
        # OBSERVER и прочие роли блоков не образуют.
    flush([])

    # Блоки из одних подтверждений склеиваем со следующим вопросом.
    merged: list[Block] = []
    carry: Block | None = None
    for block in blocks:
        if _is_backchannel_only(block):
            carry = block if carry is None else _concat(carry, block)
            continue
        if carry is not None:
            block = _concat(carry, block)
            carry = None
        merged.append(block)
    if carry is not None:
        if merged:
            merged[-1] = _concat(merged[-1], carry)
        else:
            merged.append(carry)

    # Перенумеровать после склейки: модель работает с номерами блоков.
    return [
        Block(
            index=position,
            interviewer_ids=block.interviewer_ids,
            respondent_ids=block.respondent_ids,
            question_text=block.question_text,
            answer_text=block.answer_text,
            question_is_backchannel=block.question_is_backchannel,
            answer_is_backchannel=block.answer_is_backchannel,
        )
        for position, block in enumerate(merged, start=1)
    ]


def _is_backchannel_only(block: Block) -> bool:
    """Блок без содержательного вопроса: только подтверждения интервьюера."""
    return block.question_is_backchannel and block.answer_is_backchannel


def _add_respondent(block: Block, item: UtteranceWithL2) -> Block:
    """Добавить реплику респондента к блоку (подтверждение после ответа)."""
    answer_text = " ".join(part for part in (block.answer_text, item.l2_text or item.text) if part).strip()
    return Block(
        index=block.index,
        interviewer_ids=block.interviewer_ids,
        respondent_ids=(*block.respondent_ids, item.id),
        question_text=block.question_text,
        answer_text=answer_text,
        question_is_backchannel=block.question_is_backchannel,
        answer_is_backchannel=block.answer_is_backchannel and item.is_backchannel,
    )


def _concat(left: Block, right: Block) -> Block:
    """Склеить два блока в один (подтверждения примыкают к вопросу)."""
    return Block(
        index=left.index,
        interviewer_ids=(*left.interviewer_ids, *right.interviewer_ids),
        respondent_ids=(*left.respondent_ids, *right.respondent_ids),
        question_text=" ".join(part for part in (left.question_text, right.question_text) if part),
        answer_text=" ".join(part for part in (left.answer_text, right.answer_text) if part),
        question_is_backchannel=left.question_is_backchannel and right.question_is_backchannel,
        answer_is_backchannel=left.answer_is_backchannel and right.answer_is_backchannel,
    )


@dataclass
class StructureStats:
    """Счётчики прогона привязки и сегментации."""

    interviewer: int = 0
    respondent: int = 0
    blocks: int = 0
    in_guide: int = 0
    out_of_guide: int = 0
    probes: int = 0
    fragments: int = 0
    model_calls: int = 0
    model_bindings: int = 0
    unknown_questions: int = 0
    unbound_blocks: int = 0
    fallback_segments: int = 0
    rejected_segments: int = 0
    per_question: dict[str | None, int] = field(default_factory=dict)
    similarity_source: str = ""
    model_used_hint: str = ""


@dataclass
class StructureResult:
    """Итог стадии structure."""

    interview_code: str
    interview_db_id: int
    l2_version: int
    stats: StructureStats
    skipped: bool = False
    run_id: int | None = None
    model_used: str = ""


def load_prompt(name: str) -> str:
    """Прочитать файл промпта из `prompts/`."""
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def prompt_hash(*texts: str) -> str:
    """Хеш промптов для записи в run."""
    joined = "\n\n".join(texts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _normalize_for_similarity(text: str) -> str:
    """Текст для сравнения: нижний регистр, ё -> е, схлопнутые пробелы."""
    return " ".join(text.casefold().replace("ё", "е").split())


def lexical_similarity(left: str, right: str) -> float:
    """Сходство двух текстов по словам, 0-1.

    Используется как запасной путь, когда эмбеддинги недоступны, и как
    подсказка для промпта: модель всё равно проверяет кандидатов.
    """
    return fuzz.token_set_ratio(_normalize_for_similarity(left), _normalize_for_similarity(right)) / 100


def _cosine(left: list[float], right: list[float]) -> float:
    """Косинусное сходство двух векторов; нулевые векторы дают 0."""
    if len(left) != len(right) or not left:
        return 0.0
    dot: float = sum(a * b for a, b in zip(left, right, strict=True))
    norm_left: float = sum(a * a for a in left) ** 0.5
    norm_right: float = sum(b * b for b in right) ** 0.5
    if norm_left == 0 or norm_right == 0:
        return 0.0
    scale = norm_left * norm_right
    if scale == 0.0:
        return 0.0
    return float(dot / scale)


def question_scores(
    blocks: list[Block],
    questions: list[GuideQuestion],
    provider: LlmProvider | None,
) -> tuple[dict[int, dict[int, float]], str]:
    """Сходство каждой пары «вопрос + ответ» с каждым вопросом гайда.

    Сначала пробуем эмбеддинги: они сравнивают смысл, а не слова. Если провайдер
    не умеет эмбеддинги или вернул нулевые векторы (заглушка), считается
    лексическое сходство — оно грубее, но работает без сети.

    Returns:
        (карта «номер блока -> {id вопроса: сходство}», источник сходства).
    """
    pair_texts = {
        block.index: " ".join(part for part in (block.question_text, block.answer_text) if part) for block in blocks
    }
    if provider is not None and pair_texts and questions:
        try:
            texts = list(pair_texts.values())
            wording_vectors = provider.embed([question.wording for question in questions])
            pair_vectors = provider.embed(texts)
            if any(any(value != 0.0 for value in vector) for vector in pair_vectors):
                scores: dict[int, dict[int, float]] = {}
                for (index, _text), vector in zip(pair_texts.items(), pair_vectors, strict=True):
                    scores[index] = {
                        question.id: _cosine(vector, wording)
                        for question, wording in zip(questions, wording_vectors, strict=True)
                    }
                return scores, "embeddings"
        except LlmError as exc:
            logger.info("Эмбеддинги недоступны (%s), сравниваю по словам", exc)

    scores = {}
    for index, text in pair_texts.items():
        scores[index] = {question.id: lexical_similarity(text, question.wording) for question in questions}
    return scores, "lexical"


def build_bind_prompt(
    questions: list[GuideQuestion],
    candidates: dict[int, list[tuple[str, float]]],
    blocks: list[Block],
) -> str:
    """Собрать промпт привязки: гайд, кандидаты по сходству, блоки вопрос-ответ."""
    template = load_prompt("bind_questions.md")
    guide_text = "\n".join(
        f"- {question.identifier} | {question.group or '—'} | {question.wording}" for question in questions
    )
    candidate_lines = []
    for block in blocks:
        items = candidates.get(block.index, [])
        if not items:
            candidate_lines.append(f"- блок {block.index}: кандидатов нет")
            continue
        joined = "; ".join(f"{identifier} ({score:.2f})" for identifier, score in items)
        candidate_lines.append(f"- блок {block.index}: {joined}")
    block_lines = []
    for block in blocks:
        question = block.question_text or "(нет реплик интервьюера)"
        answer = block.answer_text or "(ответа нет)"
        block_lines.append(f"{block.index} | ВОПРОС: {question} | ОТВЕТ: {answer}")
    return (
        template.replace("{guide}", guide_text or "- (гайд пуст)")
        .replace("{candidates}", "\n".join(candidate_lines) or "- (кандидатов нет)")
        .replace("{blocks}", "\n".join(block_lines))
    )


def build_segment_prompt(
    utterance_id: int,
    text: str,
    question_wording: str | None,
    hints: list[str],
) -> str:
    """Собрать промпт сегментации одной реплики."""
    template = load_prompt("segment.md")
    hint_text = "\n".join(f"- {hint}" for hint in hints) or "- (подсказок нет)"
    return (
        template.replace("{utterance}", str(utterance_id))
        .replace("{question}", question_wording or "(нет)")
        .replace("{text}", text)
        .replace("{hints}", hint_text)
    )


def segmentation_hints(utterance: UtteranceWithL2, changes: list[dict[str, Any]]) -> list[str]:
    """Подсказки для сегментации из журнала изменений и оценки качества.

    Код не задаёт границы, а показывает модели места, где меняется тема:
    замены терминов, исправления ASR и пометку неразборчивого места.
    """
    hints: list[str] = []
    for change in changes:
        kind = change.get("type")
        if kind == "term":
            hints.append(f"термин: {change.get('from')!s} -> {change.get('to')!s}")
        elif kind == "asr_fix":
            hints.append(f"исправление ASR: {change.get('from')!s} -> {change.get('to')!s}")
        elif kind == "garbled":
            hints.append("неразборчивое место")
    if utterance.asr_quality == "garbled":
        hints.append("реплика помечена как garbled")
    return hints


def validate_spans(spans: list[FragmentSpan], text_length: int, max_chars: int) -> str | None:
    """Проверить границы фрагментов; вернуть причину отказа или None.

    Требования: границы внутри текста, порядок по возрастанию, отсутствие
    пересечений и пропусков, полное покрытие текста, непустой текст фрагмента.
    """
    if not spans:
        return "нет фрагментов"
    if spans[0].start != 0:
        return "первый фрагмент начинается не с начала текста"
    if spans[-1].end != text_length:
        return "последний фрагмент не доходит до конца текста"
    previous_end = 0
    for span in spans:
        if span.start < 0 or span.end > text_length:
            return "границы выходят за текст"
        if span.end <= span.start:
            return "пустой или обратный фрагмент"
        if span.start != previous_end:
            return "фрагменты пересекаются или оставляют пропуск"
        if span.end - span.start > max_chars:
            return "фрагмент длиннее предела"
        previous_end = span.end
    return None


def spans_from_payload(payload: list[Any], text: str) -> list[FragmentSpan] | None:
    """Найти позиции фрагментов в тексте последовательным поиском.

    Модель возвращает дословные куски текста, а не числовые границы: смещения
    она считает ненадёжно (на реальном интервью последний `end` систематически
    не доходил до конца текста). Позиции ищутся кодом.

    Поиск идёт от конца предыдущего фрагмента, поэтому фрагменты не
    пересекаются и не переставляются. Пробелы между кусками относятся к
    предыдущему фрагменту: так текст покрывается целиком без дыр.

    Returns:
        Границы или None, если кусок не найден или порядок нарушен.
    """
    spans: list[FragmentSpan] = []
    cursor = 0
    for item in payload:
        if not isinstance(item, str):
            return None
        chunk = item.strip()
        if not chunk:
            return None
        position = text.find(chunk, cursor)
        if position < 0:
            return None
        spans.append(FragmentSpan(position, position + len(chunk)))
        cursor = position + len(chunk)
    return spans


def _absorb_gaps(spans: list[FragmentSpan], text: str) -> list[FragmentSpan]:
    """Расширить фрагменты на промежутки между кусками модели.

    Модель возвращает смысловые куски без служебных пробелов на стыках. Пробел
    между куском и следующим относится к предыдущему фрагменту, поэтому
    фрагменты снова покрывают текст целиком без дыр. Первый фрагмент
    начинается с нуля, последний доходит до конца текста.
    """
    if not spans:
        return spans
    result = [FragmentSpan(0, spans[0].end)]
    for span in spans[1:]:
        result.append(FragmentSpan(result[-1].end, span.end))
    result[-1] = FragmentSpan(result[-1].start, len(text))
    return result


def fallback_spans(text: str, target_chars: int = _FALLBACK_TARGET_CHARS) -> list[FragmentSpan]:
    """Детерминированное деление по предложениям.

    Границы ставятся после конца предложения; предложения группируются, пока
    длина группы не превысит `target_chars`. Одно предложение длиннее предела
    режется по последнему пробелу перед пределом: иначе фрагмент нарушил бы
    `max_fragment_chars` и не попал бы в БД. Пустой текст даёт один фрагмент.
    """
    if not text:
        return [FragmentSpan(0, 0)]
    limit = max(1, target_chars)
    ends = [match.end() for match in _SENTENCE_END_RE.finditer(text)]
    if not ends or ends[-1] != len(text):
        ends.append(len(text))
    spans: list[FragmentSpan] = []
    start = 0
    for end in ends:
        while end - start > limit:
            cut = text.rfind(" ", start + 1, start + limit + 1)
            if cut <= start:
                cut = start + limit
            spans.append(FragmentSpan(start, cut))
            start = cut
        if end - start >= limit:
            spans.append(FragmentSpan(start, end))
            start = end
    if start < len(text):
        spans.append(FragmentSpan(start, len(text)))
    return spans


def _chunk_text(text: str, limit: int = 120) -> str:
    """Короткая выдержка текста для отчёта (без служебных данных)."""
    return text if len(text) <= limit else text[:limit] + "…"


def _resolve_question(identifier: Any, by_identifier: dict[str, GuideQuestion]) -> GuideQuestion | None:
    """Вопрос по идентификатору из ответа модели; неизвестный даёт None."""
    if identifier is None:
        return None
    key = str(identifier).strip()
    return by_identifier.get(key)


def bind_blocks(
    blocks: list[Block],
    questions: list[GuideQuestion],
    provider: LlmProvider | None,
    settings: Settings,
    stats: StructureStats,
) -> dict[int, tuple[int | None, bool]]:
    """Привязать блоки «вопрос -> ответ» к вопросам гайда.

    Сходство считается по паре «вопрос + ответ»: один вопрос интервьюера часто
    короткий и общий («А почему?»), а ответ респондента содержит тему. Модель
    получает весь список блоков и видит последовательность интервью.

    Returns:
        карта «номер блока -> (id вопроса или None, is_probe)».
    """
    if not blocks or not questions:
        stats.unbound_blocks = len(blocks)
        return {block.index: (None, False) for block in blocks}

    by_identifier = {question.identifier: question for question in questions}
    scores, source = question_scores(blocks, questions, provider)
    stats.similarity_source = source

    candidates: dict[int, list[tuple[str, float]]] = {}
    by_id = {question.id: question for question in questions}
    threshold = settings.m1c.min_similarity
    for block in blocks:
        per_question = scores.get(block.index, {})
        ordered = sorted(per_question.items(), key=lambda pair: pair[1], reverse=True)[:CANDIDATES_PER_BLOCK]
        candidates[block.index] = [
            (by_id[question_id].identifier, score) for question_id, score in ordered if score >= threshold
        ]

    bindings: dict[int, tuple[int | None, bool]] = {block.index: (None, False) for block in blocks}
    if provider is None:
        # Без модели единственный источник — сходство пары «вопрос + ответ».
        for index, items in candidates.items():
            if not items:
                continue
            question = by_identifier.get(items[0][0])
            if question is not None:
                bindings[index] = (question.id, False)
        stats.unbound_blocks = sum(1 for question_id, _probe in bindings.values() if question_id is None)
        return bindings

    prompt = build_bind_prompt(questions, candidates, blocks)
    try:
        result = provider.complete_json_detailed(prompt, BIND_SCHEMA)
    except LlmValidationError as exc:
        logger.warning("LLM: привязка не прошла валидацию JSON: %s", exc)
        stats.unbound_blocks = len(blocks)
        return bindings
    except LlmError as exc:
        logger.warning("LLM: вызов привязки не удался: %s", exc)
        stats.unbound_blocks = len(blocks)
        return bindings

    stats.model_calls += 1
    stats.model_used_hint = result.model_used
    for item in result.data.get("blocks") or []:
        if not isinstance(item, dict):
            continue
        raw_block = item.get("block")
        if not isinstance(raw_block, int) or raw_block not in bindings:
            continue
        question = _resolve_question(item.get("question"), by_identifier)
        if item.get("question") is not None and question is None:
            stats.unknown_questions += 1
        bindings[raw_block] = (None if question is None else question.id, bool(item.get("is_probe")))
        stats.model_bindings += 1

    # Модель могла пропустить блок: он остаётся вне гайда, а не получает
    # «ближайший» вопрос по сходству.
    stats.unbound_blocks = sum(1 for question_id, _probe in bindings.values() if question_id is None)
    return bindings


def segment_utterance(
    utterance: UtteranceWithL2,
    question_wording: str | None,
    changes: list[dict[str, Any]],
    provider: LlmProvider | None,
    settings: Settings,
    stats: StructureStats,
) -> list[FragmentSpan]:
    """Получить границы фрагментов реплики: модель, иначе детерминированное деление.

    Модель возвращает список дословных кусков текста. Код находит их позиции
    последовательным поиском и проверяет покрытие: кусок, которого нет в тексте
    или который нарушает порядок, отклоняет ответ целиком, и применяется
    деление по предложениям.
    """
    text = utterance.l2_text or ""
    if not text.strip():
        return []
    max_chars = settings.m1c.max_fragment_chars
    if provider is not None:
        prompt = build_segment_prompt(utterance.id, text, question_wording, segmentation_hints(utterance, changes))
        try:
            result = provider.complete_json_detailed(prompt, SEGMENT_SCHEMA)
        except (LlmValidationError, LlmError) as exc:
            logger.info("LLM: сегментация реплики %d не удалась (%s), делю по предложениям", utterance.id, exc)
            result = None
        if result is not None:
            stats.model_calls += 1
            spans = spans_from_payload(result.data.get("fragments") or [], text)
            if spans is not None:
                spans = _absorb_gaps(spans, text)
                reason = validate_spans(spans, len(text), max_chars)
                if reason is None:
                    return spans
                logger.info("LLM: границы реплики %d отклонены (%s)", utterance.id, reason)
                stats.rejected_segments += 1
            else:
                logger.info("LLM: кусок реплики %d не найден в тексте, делю по предложениям", utterance.id)
                stats.rejected_segments += 1

    stats.fallback_segments += 1
    spans = fallback_spans(text)
    reason = validate_spans(spans, len(text), max_chars)
    if reason is not None:
        # Последний рубеж: жёсткая нарезка по пределу. Один фрагмент длиннее
        # предела записать нельзя — ограничение задано схемой.
        logger.warning("Сегментация реплики %d: %s, режу по пределу длины", utterance.id, reason)
        spans = [FragmentSpan(start, min(start + max_chars, len(text))) for start in range(0, len(text), max_chars)]
    return spans


def structure_interview(
    conn: sqlite3.Connection,
    interview_code: str,
    settings: Settings,
    provider: LlmProvider | None,
    *,
    l2_version: int | None = None,
    code_version: str = "",
    rebuild: bool = False,
) -> StructureResult:
    """Привязать реплики интервью к вопросам и разбить L2 на фрагменты.

    Args:
        conn: соединение с БД.
        interview_code: внешний код интервью (R01).
        settings: настройки проекта.
        provider: провайдер LLM; None — только детерминированные правила.
        l2_version: версия L2; None — последняя записанная.
        code_version: версия кода для записи в run.
        rebuild: пересобрать привязку и фрагменты, даже если они уже есть.
    """
    interview_db_id = InterviewRepository(conn).interview_db_id_by_code(interview_code)
    if interview_db_id is None:
        msg = f"Интервью {interview_code} не найдено в БД: сначала загрузите транскрипт"
        raise StructureError(msg)

    text_units = TextUnitRepository(conn)
    version = l2_version if l2_version is not None else text_units.max_version_for_interview(interview_db_id)
    if version <= 0:
        msg = f"У интервью {interview_code} нет слоя единиц L2: сначала выполните normalize"
        raise StructureError(msg)
    if text_units.count_for_interview(interview_db_id, version) == 0:
        msg = f"У интервью {interview_code} нет единиц L2 версии {version}: выберите существующую версию"
        raise StructureError(msg)

    structure = StructureRepository(conn)
    existing = structure.fragments_of_interview(interview_db_id)
    if existing and not rebuild:
        # В обработку идёт только одна версия L2 — выбранная. Если фрагменты
        # построены по другой (например, до перезапуска normalize), они
        # пересобираются: смешивать срезы разных версий нельзя.
        built_from = structure.fragment_l2_versions(interview_db_id)
        if built_from == {version}:
            logger.info("Фрагменты %s уже записаны по L2 v%d, пропуск", interview_code, version)
            return StructureResult(
                interview_code=interview_code,
                interview_db_id=interview_db_id,
                l2_version=version,
                stats=StructureStats(),
                skipped=True,
            )
        logger.info(
            "Фрагменты %s построены по L2 %s, пересобираю по версии %d",
            interview_code,
            ",".join(f"v{item}" for item in sorted(built_from)) or "—",
            version,
        )
        rebuild = True

    utterances = structure.utterances_with_l2(interview_db_id, version)
    if not utterances:
        msg = f"У интервью {interview_code} нет реплик: сначала загрузите транскрипт"
        raise StructureError(msg)

    questions = structure.guide_questions()
    if not questions:
        logger.warning("Гайд не загружен: все реплики останутся вне гайда")

    stats = StructureStats(
        interviewer=sum(1 for item in utterances if item.speaker_role == "INTERVIEWER"),
        respondent=sum(1 for item in utterances if item.speaker_role == "RESPONDENT"),
    )
    blocks = build_blocks(utterances)
    stats.blocks = len(blocks)
    bindings = bind_blocks(blocks, questions, provider, settings, stats)
    by_question_id = {question.id: question for question in questions}

    # Привязка блока распространяется на все его реплики: вопрос интервьюера и
    # ответ респондента — один контекст. `prior_interviewer_turn_id` — id
    # последней записи транскрипта у предыдущей реплики интервьюера.
    utterance_binding: dict[int, tuple[int | None, bool, int | None]] = {}
    previous_turn_id: int | None = None
    for block in blocks:
        question_id, is_probe = bindings.get(block.index, (None, False))
        for utterance_id in block.interviewer_ids:
            # У реплики интервьюера «предшествующий вопрос» не заполняется:
            # признак нужен только ответу респондента.
            utterance_binding[utterance_id] = (question_id, is_probe, None)
            turn_id = structure.last_turn_id(utterance_id)
            if turn_id is not None:
                previous_turn_id = turn_id
        for utterance_id in block.respondent_ids:
            utterance_binding[utterance_id] = (question_id, False, previous_turn_id)

    fragments: list[FragmentRow] = []
    changes_by_unit = _changes_by_text_unit(conn, interview_db_id, version)
    for item in utterances:
        if item.speaker_role != "RESPONDENT" or item.is_backchannel or item.text_unit_id is None:
            continue
        question_id, _probe, prior_turn_id = utterance_binding.get(item.id, (None, False, None))
        wording = None if question_id is None else by_question_id.get(question_id)
        spans = segment_utterance(
            item,
            None if wording is None else wording.wording,
            changes_by_unit.get(item.text_unit_id, []),
            provider,
            settings,
            stats,
        )
        text = item.l2_text or ""
        for span in spans:
            fragments.append(
                FragmentRow(
                    utterance_id=item.id,
                    text_unit_id=item.text_unit_id,
                    start=span.start,
                    end=span.end,
                    text=span.text_of(text),
                    question_id=question_id,
                    prior_interviewer_turn_id=prior_turn_id,
                    asr_quality=item.asr_quality or "ok",
                )
            )

    run_prompt_hash = prompt_hash(load_prompt("bind_questions.md"), load_prompt("segment.md"))
    model_used = _model_label(provider, settings, stats)
    params = {
        "l2_version": version,
        "min_similarity": settings.m1c.min_similarity,
        "max_fragment_chars": settings.m1c.max_fragment_chars,
        "similarity_source": stats.similarity_source,
        "blocks": stats.blocks,
        "model_calls": stats.model_calls,
    }

    conn.execute("SAVEPOINT structure")
    try:
        run_id = RunRepository(conn).start(
            model=model_used,
            prompt_hash=run_prompt_hash,
            codebook_version_id=RunRepository(conn).latest_codebook_version_id(),
            code_version=code_version,
            params_json=json.dumps(params, ensure_ascii=False),
        )
        if rebuild:
            structure.clear_fragments(interview_db_id)
        for utterance_id, (question_id, is_probe, prior_turn_id) in utterance_binding.items():
            structure.set_utterance_binding(utterance_id, question_id, is_probe, prior_turn_id)
        for row in fragments:
            structure.insert_fragment(row)
    except BaseException:
        conn.execute("ROLLBACK TO SAVEPOINT structure")
        conn.execute("RELEASE SAVEPOINT structure")
        raise
    conn.execute("RELEASE SAVEPOINT structure")
    conn.commit()

    stats.fragments = len(fragments)
    stats.in_guide = sum(1 for question_id, _probe in bindings.values() if question_id is not None)
    stats.out_of_guide = sum(1 for question_id, _probe in bindings.values() if question_id is None)
    stats.probes = sum(1 for _question_id, is_probe in bindings.values() if is_probe)
    stats.per_question = structure.utterances_per_question(interview_db_id)
    stats.similarity_source = stats.similarity_source or "—"

    logger.info(
        "structure %s (L2 v%d): блоков=%d, в гайде=%d, вне гайда=%d, фрагментов=%d, вызовов=%d",
        interview_code,
        version,
        stats.blocks,
        stats.in_guide,
        stats.out_of_guide,
        stats.fragments,
        stats.model_calls,
    )
    return StructureResult(
        interview_code=interview_code,
        interview_db_id=interview_db_id,
        l2_version=version,
        stats=stats,
        run_id=run_id,
        model_used=model_used,
    )


def _changes_by_text_unit(
    conn: sqlite3.Connection,
    interview_db_id: int,
    version: int,
) -> dict[int, list[dict[str, Any]]]:
    """Журнал изменений L2 по единицам интервью: id единицы -> список изменений."""
    rows = conn.execute(
        "SELECT t.id, t.changes_json FROM text_unit t JOIN utterance u ON t.utterance_id = u.id"
        " WHERE u.interview_id = ? AND t.version = ?",
        (interview_db_id, version),
    ).fetchall()
    result: dict[int, list[dict[str, Any]]] = {}
    for unit_id, raw in rows:
        try:
            value = json.loads(str(raw))
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            result[int(unit_id)] = [item for item in value if isinstance(item, dict)]
    return result


def _model_label(provider: LlmProvider | None, settings: Settings, stats: StructureStats) -> str:
    """Модель для записи в run: фактическая из ответа, иначе запрошенная."""
    if provider is None:
        return "deterministic"
    return stats.model_used_hint or settings.llm.model or ""


def format_report(result: StructureResult) -> list[str]:
    """Строки отчёта для CLI (без текста реплик)."""
    stats = result.stats
    lines = [
        f"structure {result.interview_code} (L2 v{result.l2_version}): "
        f"реплик интервьюера={stats.interviewer}, респондента={stats.respondent}, "
        f"блоков вопрос-ответ={stats.blocks}, фрагментов={stats.fragments}",
        f"  привязка: в гайде={stats.in_guide}, вне гайда={stats.out_of_guide}, "
        f"уточнений={stats.probes}, неизвестных вопросов={stats.unknown_questions}, "
        f"без ответа модели={stats.unbound_blocks}, сходство={stats.similarity_source}",
        f"  сегментация: отказов модели={stats.rejected_segments}, "
        f"делений по предложениям={stats.fallback_segments}, вызовов={stats.model_calls}",
        "  распределение реплик респондента по вопросам:",
    ]
    for identifier, count in sorted(stats.per_question.items(), key=lambda pair: (pair[0] is None, pair[0] or "")):
        lines.append(f"    {identifier or 'вне гайда'}: {count}")
    return lines


__all__ = [
    "BIND_SCHEMA",
    "SEGMENT_SCHEMA",
    "Block",
    "FragmentSpan",
    "StructureError",
    "StructureResult",
    "StructureStats",
    "bind_blocks",
    "build_bind_prompt",
    "build_blocks",
    "build_segment_prompt",
    "fallback_spans",
    "format_report",
    "lexical_similarity",
    "load_prompt",
    "prompt_hash",
    "question_scores",
    "segment_utterance",
    "spans_from_payload",
    "structure_interview",
    "validate_spans",
]
