"""Стадия coding (M2): присвоение фрагментам L2 кодов текущего кодбука.

Единица вызова модели — реплика респондента: в промпт идёт её L2 целиком со
списком фрагментов, ответ описывает все фрагменты. Проверка цитаты остаётся
внутри фрагмента: цитата обязана быть срезом своего `fragment`, а не всей
реплики.

Что проверяется кодом перед записью (SPEC-M2, «Проверка результата кодом»):
цитата равна срезу L2 и лежит внутри фрагмента, код есть в текущей версии
кодбука, обязательные атрибуты заполнены, у `OTH` есть описание, `reliability`
в диапазоне 1–3. Нарушение — отказ аннотации с причиной в
`fragment_coding.note`, а не молчаливая запись.

Повторный прогон не перезаписывает принятое и исправленное: удаляются только
аннотации в статусе `proposed`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qa.db.workbook_repo import (
    AnnotationRow,
    CodeRow,
    CodingRepository,
    FragmentCodingRow,
    InterviewRepository,
    LexiconRepository,
    RunRepository,
    StructureRepository,
)
from qa.llm.base import LlmProvider
from qa.llm.errors import LlmError
from qa.pipeline.fragment_quality import fragment_asr_quality, tokens_of
from qa.pipeline.normalize_rules import LexiconEntry
from qa.settings import Settings

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"

CODING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "fragments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fragment_id": {"type": "integer"},
                    "codes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string"},
                                "quote": {"type": "string"},
                                "rationale": {"type": "string"},
                                "order": {"type": ["string", "null"]},
                                "scenario": {"type": ["string", "null"]},
                                "reliability": {"type": ["integer", "null"]},
                                "reliability_note": {"type": ["string", "null"]},
                                "oth_description": {"type": ["string", "null"]},
                            },
                            "required": ["code", "quote", "rationale", "order", "scenario", "reliability"],
                            "additionalProperties": False,
                        },
                    },
                    "interpretation": {"type": ["string", "null"]},
                    "interpretation_only": {"type": "boolean"},
                },
                "required": ["fragment_id", "codes", "interpretation", "interpretation_only"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["fragments"],
    "additionalProperties": False,
}

# Атрибут книги, без которого кодирование не начинается: имя задаёт оператор.
RELIABILITY_ATTRIBUTE = "НАДЁЖНОСТЬ_КОДА"
ORDER_VALUES = frozenset({"spontaneous", "prompted"})
SCENARIO_VALUES = frozenset({"real", "imaginary"})
SCENARIO_QUESTION = "14Т-б"


class CodingError(Exception):
    """Кодирование невозможно: нет интервью, фрагментов, кодбука или атрибута."""


@dataclass(frozen=True)
class FragmentForCoding:
    """Фрагмент реплики в объёме, нужном для промпта и проверок."""

    id: int
    text_unit_id: int
    start: int
    end: int
    text: str
    unit_text: str
    question_id: int | None
    asr_quality: str

    def quote_positions(self, quote: str) -> tuple[int, int] | None:
        """Позиции цитаты в тексте единицы, если цитата лежит внутри фрагмента."""
        position = self.unit_text.find(quote, self.start, self.end)
        if position < 0:
            return None
        return position, position + len(quote)


@dataclass(frozen=True)
class UtteranceForCoding:
    """Реплика респондента со своими фрагментами."""

    id: int
    question_id: int | None
    question_identifier: str | None
    question_wording: str | None
    prior_turns: str
    fragments: list[FragmentForCoding]


@dataclass
class CodingStats:
    """Счётчики прогона кодирования."""

    utterances: int = 0
    fragments: int = 0
    model_calls: int = 0
    coded: int = 0
    unmarked: int = 0
    failed: int = 0
    skipped_garbled: int = 0
    skipped_protected: int = 0
    annotations: int = 0
    oth: int = 0
    interpretation_only: int = 0
    doubtful_with_interpretation: int = 0
    rejected_annotations: int = 0
    by_quality: dict[str, int] = field(default_factory=dict)
    rejections: dict[str, int] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_used_hint: str = ""


@dataclass
class CodingResult:
    """Итог стадии кодирования."""

    interview_code: str
    interview_db_id: int
    run_id: int | None
    stats: CodingStats
    skipped: bool = False
    model_used: str = ""


@dataclass(frozen=True)
class DryRunCodingReport:
    """Оценка объёма работы без вызовов модели."""

    interview_code: str
    utterances: int
    fragments: int
    garbled: int
    doubtful: int
    model_calls: int
    prompt_tokens: int


def load_prompt() -> str:
    """Прочитать файл промпта кодирования."""
    return (PROMPTS_DIR / "code_utterance.md").read_text(encoding="utf-8")


def prompt_hash(text: str) -> str:
    """Хеш промпта для записи в run."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _question_text(identifier: str | None, wording: str | None) -> str:
    """Строка вопроса для промпта."""
    if identifier is None and not wording:
        return "(вне гайда: вопрос не задавался)"
    return f"{identifier or '—'} | {wording or '—'}"


def _codes_text(codes: list[CodeRow]) -> str:
    """Список кодов для промпта."""
    lines: list[str] = []
    for code in codes:
        parts = [
            code.name,
            code.family or "—",
            code.title or "—",
            code.definition or "—",
            code.include or "—",
            code.exclude or "—",
            code.tiebreak_rule or "—",
        ]
        lines.append(" | ".join(parts))
    return "\n".join(lines) or "- (кодбук пуст)"


def build_coding_prompt(
    codes: list[CodeRow],
    utterance: UtteranceForCoding,
) -> str:
    """Собрать промпт кодирования одной реплики."""
    template = load_prompt()
    fragment_lines = "\n".join(
        f"{fragment.id} | {fragment.asr_quality} | {fragment.text}" for fragment in utterance.fragments
    )
    return (
        template.replace("{codes}", _codes_text(codes))
        .replace("{question}", _question_text(utterance.question_identifier, utterance.question_wording))
        .replace("{prior_turns}", utterance.prior_turns or "(реплик интервьюера не было)")
        .replace("{fragments}", fragment_lines or "(фрагментов нет)")
    )


def load_codes(conn: sqlite3.Connection) -> tuple[int | None, list[CodeRow]]:
    """Коды текущей версии кодбука и её id."""
    repository = CodingRepository(conn)
    version_id = repository.latest_codebook_version_id()
    if version_id is None:
        return None, []
    return version_id, repository.codes(version_id)


def require_reliability_attribute(conn: sqlite3.Connection) -> None:
    """Проверить, что в загруженной версии кодбука есть атрибут НАДЁЖНОСТЬ_КОДА.

    Имя атрибута задаёт оператор в книге; в коде оно не подменяется, иначе
    проверка обязательных атрибутов тихо проверяла бы не то поле.

    Проверяется версия из БД, а не файл книги: кодирование работает с кодами
    загруженной версии. Если книгу переименовали, но не перезагрузили, версия
    в БД осталась старой — об этом и говорится в ошибке, иначе сообщение
    отправляло бы править уже исправленную книгу.

    Raises:
        CodingError: атрибута нет в текущей версии кодбука.
    """
    repository = CodingRepository(conn)
    version_id = repository.latest_codebook_version_id()
    if version_id is None:
        msg = "Кодбук не загружен: сначала выполните load-codebook"
        raise CodingError(msg)
    names = repository.attribute_names(version_id)
    if RELIABILITY_ATTRIBUTE not in names:
        msg = f"В листе «Атрибуты» нет поля {RELIABILITY_ATTRIBUTE}: переименуйте атрибут в книге и выполните load-codebook"
        raise CodingError(msg)


def _lexicon_entries(conn: sqlite3.Connection) -> list[LexiconEntry]:
    """Термины лексикона без отклонённых."""
    entries: list[LexiconEntry] = []
    for canonical, data in LexiconRepository(conn).all_terms().items():
        if data.get("status") == "rejected":
            continue
        variants = tuple(str(item) for item in data.get("variants") or [])
        entries.append(LexiconEntry(canonical=canonical, variants=variants))
    return entries


def _asr_fix_words_by_unit(conn: sqlite3.Connection, interview_db_id: int, version: int) -> dict[int, set[str]]:
    """Слова, заменённые по `asr_fix`, по единицам L2 интервью."""
    rows = conn.execute(
        "SELECT t.id, t.changes_json FROM text_unit t JOIN utterance u ON t.utterance_id = u.id"
        " WHERE u.interview_id = ? AND t.version = ?",
        (interview_db_id, version),
    ).fetchall()
    result: dict[int, set[str]] = {}
    for unit_id, raw in rows:
        words: set[str] = set()
        try:
            changes = json.loads(str(raw))
        except json.JSONDecodeError:
            changes = []
        for change in changes if isinstance(changes, list) else []:
            if isinstance(change, dict) and change.get("type") == "asr_fix" and change.get("to"):
                words |= tokens_of(str(change["to"]))
        result[int(unit_id)] = words
    return result


def _new_term_words(conn: sqlite3.Connection, interview_db_id: int) -> dict[int, set[str]]:
    """Формы кандидатов `new_term` по репликам интервью.

    Отклонённые кандидаты не считаются искажением: статус `rejected` означает,
    что форма признана выдумкой ASR или модели, а не новым термином.
    """
    rows = conn.execute(
        "SELECT context_utterance_id, surface_form FROM new_term WHERE interview_id = ? AND status <> 'rejected'",
        (interview_db_id,),
    ).fetchall()
    result: dict[int, set[str]] = {}
    for utterance_id, form in rows:
        if utterance_id is None:
            continue
        result.setdefault(int(utterance_id), set()).update(tokens_of(str(form)))
    return result


def _prior_turns(conn: sqlite3.Connection, interview_db_id: int, utterance_id: int, limit: int) -> str:
    """Реплики интервьюера до реплики, последние `limit` штук.

    Ограничение нужно, чтобы промпт не рос вместе с интервью. Признак ПОРЯДОК
    опирается на эти реплики, поэтому обрезка отмечается в `reliability`.
    """
    rows = conn.execute(
        "SELECT text FROM utterance WHERE interview_id = ? AND id < ? AND speaker_role = 'INTERVIEWER'"
        " AND is_backchannel = 0 ORDER BY id DESC LIMIT ?",
        (interview_db_id, utterance_id, limit),
    ).fetchall()
    texts = [str(row[0]) for row in reversed(rows)]
    return "\n".join(f"- {text}" for text in texts)


def _load_utterances(conn: sqlite3.Connection, interview_db_id: int) -> tuple[int, list[UtteranceForCoding], int]:
    """Реплики респондента с фрагментами и версия L2, на которую они ссылаются."""
    fragments = StructureRepository(conn).fragments_with_text(interview_db_id)
    if not fragments:
        return 0, [], 0
    versions = StructureRepository(conn).fragment_l2_versions(interview_db_id)
    version = max(versions) if versions else 0

    question_rows = conn.execute(
        "SELECT f.question_id, q.identifier, q.wording FROM fragment f"
        " JOIN utterance u ON f.utterance_id = u.id"
        " LEFT JOIN question q ON f.question_id = q.id"
        " WHERE u.interview_id = ? GROUP BY f.question_id",
        (interview_db_id,),
    ).fetchall()
    questions = {
        None if qid is None else int(qid): (
            None if identifier is None else str(identifier),
            None if wording is None else str(wording),
        )
        for qid, identifier, wording in question_rows
    }

    grouped: dict[int, list[FragmentForCoding]] = {}
    for fid, uid, _tuid, unit_text, start, end, frag_text, qid, quality in fragments:
        grouped.setdefault(uid, []).append(
            FragmentForCoding(
                id=fid,
                text_unit_id=_tuid,
                start=start,
                end=end,
                text=frag_text,
                unit_text=unit_text,
                question_id=qid,
                asr_quality=quality,
            )
        )

    utterances: list[UtteranceForCoding] = []
    for utterance_id in sorted(grouped):
        items = grouped[utterance_id]
        question_id = items[0].question_id
        identifier, wording = questions.get(question_id, (None, None))
        utterances.append(
            UtteranceForCoding(
                id=utterance_id,
                question_id=question_id,
                question_identifier=identifier,
                question_wording=wording,
                prior_turns="",
                fragments=items,
            )
        )
    return version, utterances, len(fragments)


def _utterance_ids_for_coding(utterances: list[UtteranceForCoding]) -> list[int]:
    """Реплики, у которых есть хотя бы один фрагмент к кодированию."""
    return [utterance.id for utterance in utterances if utterance.fragments]


def _recalculate_quality(
    conn: sqlite3.Connection,
    interview_db_id: int,
    version: int,
    utterances: list[UtteranceForCoding],
) -> list[UtteranceForCoding]:
    """Пересчитать `asr_quality` по тексту каждого фрагмента.

    Значения в `fragment` могли быть перенесены с единицы L2, поэтому и
    dry-run, и прогон считают качество заново одним и тем же правилом:
    иначе оценка объёма не совпала бы с результатом.
    """
    lexicon = _lexicon_entries(conn)
    fixes = _asr_fix_words_by_unit(conn, interview_db_id, version)
    new_terms = _new_term_words(conn, interview_db_id)
    result: list[UtteranceForCoding] = []
    for utterance in utterances:
        words = new_terms.get(utterance.id, set())
        fragments: list[FragmentForCoding] = []
        for fragment in utterance.fragments:
            quality = fragment_asr_quality(
                fragment.text,
                asr_fix_words=fixes.get(fragment.text_unit_id, set()),
                new_term_words=words,
                lexicon=lexicon,
            )
            fragments.append(
                FragmentForCoding(
                    id=fragment.id,
                    text_unit_id=fragment.text_unit_id,
                    start=fragment.start,
                    end=fragment.end,
                    text=fragment.text,
                    unit_text=fragment.unit_text,
                    question_id=fragment.question_id,
                    asr_quality=quality,
                )
            )
        result.append(
            UtteranceForCoding(
                id=utterance.id,
                question_id=utterance.question_id,
                question_identifier=utterance.question_identifier,
                question_wording=utterance.question_wording,
                prior_turns=utterance.prior_turns,
                fragments=fragments,
            )
        )
    return result


def plan_dry_run(
    conn: sqlite3.Connection,
    interview_code: str,
    settings: Settings,
) -> DryRunCodingReport:
    """Посчитать объём работы без обращения к модели."""
    interview_db_id = InterviewRepository(conn).interview_db_id_by_code(interview_code)
    if interview_db_id is None:
        msg = f"Интервью {interview_code} не найдено в БД: сначала загрузите транскрипт"
        raise CodingError(msg)
    version, utterances, total = _load_utterances(conn, interview_db_id)
    if not utterances:
        msg = f"У интервью {interview_code} нет фрагментов: сначала выполните structure"
        raise CodingError(msg)
    require_reliability_attribute(conn)

    utterances = _recalculate_quality(conn, interview_db_id, version, utterances)
    garbled = sum(1 for item in utterances for fragment in item.fragments if fragment.asr_quality == "garbled")
    doubtful = sum(1 for item in utterances for fragment in item.fragments if fragment.asr_quality == "doubtful")
    codes: list[CodeRow] = []
    version_id = CodingRepository(conn).latest_codebook_version_id()
    if version_id is not None:
        codes = CodingRepository(conn).codes(version_id)

    # Токены считаются по собранному промпту, а не по тексту фрагментов: в
    # каждом вызове заново уходят шаблон, весь кодбук и история интервьюера,
    # поэтому оценка «только фрагменты» занижала объём на порядок. Принятые и
    # исправленные фрагменты прогон не трогает, поэтому в оценку они не идут.
    protected = CodingRepository(conn).protected_fragments(interview_db_id)
    calls = 0
    prompt_tokens = 0
    prior_limit = settings.m2.prior_turns_limit
    for item in utterances:
        to_code = [
            fragment
            for fragment in item.fragments
            if fragment.asr_quality != "garbled" and fragment.id not in protected
        ]
        if not to_code:
            continue
        calls += 1
        prior = _prior_turns(conn, interview_db_id, item.id, prior_limit)
        prompt = build_coding_prompt(codes, _with_prior(item, prior))
        prompt_tokens += max(1, len(prompt) // 4)
    logger.debug("coding dry-run: L2 v%d, фрагментов=%d", version, total)
    return DryRunCodingReport(
        interview_code=interview_code,
        utterances=calls,
        fragments=total,
        garbled=garbled,
        doubtful=doubtful,
        model_calls=calls,
        prompt_tokens=prompt_tokens,
    )


def _check_code(code: str, by_name: dict[str, CodeRow]) -> CodeRow | None:
    """Код по имени из ответа модели."""
    return by_name.get(code.strip())


def _normalize_order(value: Any) -> str | None:  # noqa: ANN401
    """Признак ПОРЯДОК из ответа модели; неизвестное значение — None."""
    text = "" if value is None else str(value).strip()
    return text if text in ORDER_VALUES else None


def _normalize_scenario(value: Any, question_identifier: str | None) -> str | None:  # noqa: ANN401
    """Атрибут СЦЕНАРИЙ: только для вопроса 14Т-б."""
    if question_identifier != SCENARIO_QUESTION:
        return None
    text = "" if value is None else str(value).strip()
    return text if text in SCENARIO_VALUES else None


def _normalize_reliability(value: Any) -> int | None:  # noqa: ANN401
    """Надёжность кода 1–3; вне диапазона — None."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if 1 <= number <= 3 else None


def _reliability_cap(
    *,
    interpretation_only: bool,
    quality: str,
    interpretation: str | None,
    truncated_context: bool,
) -> int:
    """Верхняя граница надёжности по правилам SPEC-M2."""
    cap = 3
    if truncated_context:
        cap = min(cap, 2)
    if quality == "doubtful" and interpretation:
        cap = min(cap, 2)
    if interpretation_only:
        cap = min(cap, 1)
    return cap


def _build_annotation(
    *,
    fragment: FragmentForCoding,
    code: CodeRow,
    payload: dict[str, Any],
    run_id: int,
    coding_id: int,
    interpretation: str | None,
    interpretation_only: bool,
    question_identifier: str | None,
    truncated_context: bool,
) -> tuple[AnnotationRow | None, str | None]:
    """Собрать аннотацию после проверок; вернуть (аннотация, причина отказа)."""
    quote = str(payload.get("quote") or "")
    if not quote.strip():
        return None, "пустая цитата"
    positions = fragment.quote_positions(quote)
    if positions is None:
        return None, "цитата не найдена внутри фрагмента"
    start, end = positions

    reliability = _normalize_reliability(payload.get("reliability"))
    cap = _reliability_cap(
        interpretation_only=interpretation_only,
        quality=fragment.asr_quality,
        interpretation=interpretation,
        truncated_context=truncated_context,
    )
    if reliability is None:
        reliability = cap
    reliability = min(reliability, cap)

    order = _normalize_order(payload.get("order"))
    if order is None:
        return None, "не указан признак ПОРЯДОК"
    scenario = _normalize_scenario(payload.get("scenario"), question_identifier)

    oth_description = payload.get("oth_description")
    oth_text = None if oth_description is None else str(oth_description).strip()
    if code.name == "OTH" and not oth_text:
        return None, "у кода OTH нет описания"

    rationale = payload.get("rationale")
    note = payload.get("reliability_note")
    return (
        AnnotationRow(
            fragment_id=fragment.id,
            code_id=code.id,
            run_id=run_id,
            coding_id=coding_id,
            rationale=None if rationale is None else str(rationale),
            quote=quote,
            quote_start=start,
            quote_end=end,
            interpretation=interpretation,
            interpretation_only=interpretation_only,
            order_kind=order,
            scenario=scenario,
            reliability=reliability,
            reliability_note=None if note is None else str(note),
            oth_description=oth_text,
        ),
        None,
    )


def _record_fragment(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    fragment: FragmentForCoding,
    payload: dict[str, Any] | None,
    by_name: dict[str, CodeRow],
    question_identifier: str | None,
    truncated_context: bool,
    stats: CodingStats,
) -> None:
    """Записать итог одного фрагмента: коды, unmarked или failed."""
    repository = CodingRepository(conn)
    if fragment.asr_quality == "garbled":
        repository.insert_coding(
            FragmentCodingRow(
                fragment_id=fragment.id,
                run_id=run_id,
                status="skipped_garbled",
                note="искажённый текст: фрагмент не кодируется",
            )
        )
        stats.skipped_garbled += 1
        return

    if payload is None:
        repository.insert_coding(
            FragmentCodingRow(
                fragment_id=fragment.id,
                run_id=run_id,
                status="failed",
                note="модель не вернула фрагмент в ответе",
            )
        )
        stats.failed += 1
        return

    interpretation_raw = payload.get("interpretation")
    interpretation = None if interpretation_raw is None else str(interpretation_raw).strip() or None
    interpretation_only = bool(payload.get("interpretation_only"))
    codes_payload = [item for item in payload.get("codes") or [] if isinstance(item, dict)]

    if not codes_payload:
        repository.insert_coding(
            FragmentCodingRow(
                fragment_id=fragment.id,
                run_id=run_id,
                status="unmarked",
                note="значимого содержания нет",
            )
        )
        stats.unmarked += 1
        return

    coding_id = repository.insert_coding(
        FragmentCodingRow(fragment_id=fragment.id, run_id=run_id, status="coded", note=None)
    )
    written = 0
    reasons: list[str] = []
    for item in codes_payload:
        code = _check_code(str(item.get("code") or ""), by_name)
        if code is None:
            reasons.append("код вне кодбука")
            continue
        annotation, reason = _build_annotation(
            fragment=fragment,
            code=code,
            payload=item,
            run_id=run_id,
            coding_id=coding_id,
            interpretation=interpretation,
            interpretation_only=interpretation_only,
            question_identifier=question_identifier,
            truncated_context=truncated_context,
        )
        if annotation is None:
            reasons.append(reason or "аннотация отклонена")
            continue
        repository.insert_annotation(annotation)
        written += 1
        stats.annotations += 1
        if code.name == "OTH":
            stats.oth += 1
        if interpretation_only:
            stats.interpretation_only += 1
        if fragment.asr_quality == "doubtful" and interpretation:
            stats.doubtful_with_interpretation += 1

    if written == 0:
        repository.note_failure(coding_id, "; ".join(dict.fromkeys(reasons)) or "аннотации отклонены")
        stats.failed += 1
        stats.rejected_annotations += len(codes_payload)
        for reason in dict.fromkeys(reasons):
            stats.rejections[reason] = stats.rejections.get(reason, 0) + 1
        return
    stats.coded += 1
    if reasons:
        stats.rejected_annotations += len(reasons)
        for reason in dict.fromkeys(reasons):
            stats.rejections[reason] = stats.rejections.get(reason, 0) + 1


def code_interview(
    conn: sqlite3.Connection,
    interview_code: str,
    settings: Settings,
    provider: LlmProvider | None,
    *,
    code_version: str = "",
) -> CodingResult:
    """Закодировать фрагменты интервью кодами текущего кодбука.

    Args:
        conn: соединение с БД.
        interview_code: внешний код интервью (R01).
        settings: настройки проекта.
        provider: провайдер LLM; None — прогон без модели (все фрагменты failed).
        code_version: версия кода для записи в run.
    """
    interview_db_id = InterviewRepository(conn).interview_db_id_by_code(interview_code)
    if interview_db_id is None:
        msg = f"Интервью {interview_code} не найдено в БД: сначала загрузите транскрипт"
        raise CodingError(msg)

    require_reliability_attribute(conn)
    version_id, codes = load_codes(conn)
    if not codes:
        msg = "В кодбуке нет активных кодов: сначала выполните load-codebook"
        raise CodingError(msg)
    by_name = {code.name: code for code in codes}

    version, utterances, total_fragments = _load_utterances(conn, interview_db_id)
    if not utterances:
        msg = f"У интервью {interview_code} нет фрагментов: сначала выполните structure"
        raise CodingError(msg)

    stats = CodingStats(utterances=len(utterances), fragments=total_fragments)

    # Качество фрагмента пересчитывается по его собственному тексту: значения,
    # перенесённые с единицы L2, показывали искажение там, где его нет.
    recalculated = _recalculate_quality(conn, interview_db_id, version, utterances)
    structure = StructureRepository(conn)
    for before, after in zip(utterances, recalculated, strict=True):
        for old, new in zip(before.fragments, after.fragments, strict=True):
            stats.by_quality[new.asr_quality] = stats.by_quality.get(new.asr_quality, 0) + 1
            if new.asr_quality != old.asr_quality:
                structure.set_fragment_quality(new.id, new.asr_quality)
    utterances = recalculated

    prompt_text = load_prompt()
    params = {
        "temperature": 0,
        "codebook_version_id": version_id,
        "l2_version": version,
        "utterances": stats.utterances,
        "fragments": stats.fragments,
        "quality": stats.by_quality,
    }
    model_used = settings.llm.model or ""

    conn.execute("SAVEPOINT coding")
    try:
        run_id = RunRepository(conn).start(
            model=model_used,
            prompt_hash=prompt_hash(prompt_text),
            codebook_version_id=version_id,
            code_version=code_version,
            params_json=json.dumps(params, ensure_ascii=False),
        )
        repository = CodingRepository(conn)
        repository.clear_proposed(interview_db_id, run_id)
        # Принятое и исправленное не кодируется заново: иначе рядом с решением
        # человека появилось бы второе предложение модели по тому же фрагменту.
        protected = repository.protected_fragments(interview_db_id)

        prior_limit = settings.m2.prior_turns_limit
        for utterance in utterances:
            to_code = [
                fragment
                for fragment in utterance.fragments
                if fragment.asr_quality != "garbled" and fragment.id not in protected
            ]
            stats.skipped_protected += sum(1 for fragment in utterance.fragments if fragment.id in protected)
            for fragment in utterance.fragments:
                if fragment.id in protected:
                    continue
                if fragment.asr_quality == "garbled":
                    _record_fragment(
                        conn,
                        run_id=run_id,
                        fragment=fragment,
                        payload=None,
                        by_name=by_name,
                        question_identifier=utterance.question_identifier,
                        truncated_context=False,
                        stats=stats,
                    )
            if not to_code:
                continue

            prior = _prior_turns(conn, interview_db_id, utterance.id, prior_limit)
            truncated = _prior_truncated(conn, interview_db_id, utterance.id, prior_limit)
            prompt = build_coding_prompt(codes, _with_prior(utterance, prior))
            payload_by_fragment: dict[int, dict[str, Any]] = {}
            if provider is None:
                for fragment in to_code:
                    _record_fragment(
                        conn,
                        run_id=run_id,
                        fragment=fragment,
                        payload=None,
                        by_name=by_name,
                        question_identifier=utterance.question_identifier,
                        truncated_context=truncated,
                        stats=stats,
                    )
                continue

            stats.model_calls += 1
            try:
                result = provider.complete_json_detailed(prompt, CODING_SCHEMA)
            except LlmError as exc:
                logger.warning("LLM: реплика %d не закодирована: %s", utterance.id, exc)
                for fragment in to_code:
                    _record_fragment(
                        conn,
                        run_id=run_id,
                        fragment=fragment,
                        payload=None,
                        by_name=by_name,
                        question_identifier=utterance.question_identifier,
                        truncated_context=truncated,
                        stats=stats,
                    )
                continue
            stats.prompt_tokens += result.prompt_tokens
            stats.completion_tokens += result.completion_tokens
            if result.model_used:
                stats.model_used_hint = result.model_used
            for item in result.data.get("fragments") or []:
                if isinstance(item, dict) and isinstance(item.get("fragment_id"), int):
                    payload_by_fragment[int(item["fragment_id"])] = item
            for fragment in to_code:
                _record_fragment(
                    conn,
                    run_id=run_id,
                    fragment=fragment,
                    payload=payload_by_fragment.get(fragment.id),
                    by_name=by_name,
                    question_identifier=utterance.question_identifier,
                    truncated_context=truncated,
                    stats=stats,
                )
    except BaseException:
        conn.execute("ROLLBACK TO SAVEPOINT coding")
        conn.execute("RELEASE SAVEPOINT coding")
        raise
    # Модель известна только из ответа API, а run создаётся до вызовов.
    if stats.model_used_hint:
        RunRepository(conn).set_model(run_id, stats.model_used_hint)
    conn.execute("RELEASE SAVEPOINT coding")
    conn.commit()

    logger.info(
        "coding %s: реплик=%d, фрагментов=%d, coded=%d, unmarked=%d, failed=%d, skipped_garbled=%d, вызовов=%d",
        interview_code,
        stats.utterances,
        stats.fragments,
        stats.coded,
        stats.unmarked,
        stats.failed,
        stats.skipped_garbled,
        stats.model_calls,
    )
    return CodingResult(
        interview_code=interview_code,
        interview_db_id=interview_db_id,
        run_id=run_id,
        stats=stats,
        model_used=stats.model_used_hint or model_used,
    )


def _with_prior(utterance: UtteranceForCoding, prior: str) -> UtteranceForCoding:
    """Копия реплики с контекстом интервьюера."""
    return UtteranceForCoding(
        id=utterance.id,
        question_id=utterance.question_id,
        question_identifier=utterance.question_identifier,
        question_wording=utterance.question_wording,
        prior_turns=prior,
        fragments=utterance.fragments,
    )


def _prior_truncated(conn: sqlite3.Connection, interview_db_id: int, utterance_id: int, limit: int) -> bool:
    """Была ли история интервьюера обрезана лимитом."""
    row = conn.execute(
        "SELECT COUNT(*) FROM utterance WHERE interview_id = ? AND id < ? AND speaker_role = 'INTERVIEWER'"
        " AND is_backchannel = 0",
        (interview_db_id, utterance_id),
    ).fetchone()
    return int(row[0]) > limit if row is not None else False


def format_report(result: CodingResult) -> list[str]:
    """Строки отчёта для CLI (без текста реплик)."""
    stats = result.stats
    quality = ", ".join(f"{name}={value}" for name, value in sorted(stats.by_quality.items())) or "—"
    rejections = ", ".join(f"{name}={value}" for name, value in sorted(stats.rejections.items())) or "нет"
    lines = [
        f"coding {result.interview_code}: реплик={stats.utterances}, фрагментов={stats.fragments}, "
        f"coded={stats.coded}, unmarked={stats.unmarked}, failed={stats.failed}, "
        f"skipped_garbled={stats.skipped_garbled}, вызовов={stats.model_calls}",
        f"  пропущено принятых/исправленных={stats.skipped_protected}",
        f"  аннотаций={stats.annotations}, OTH={stats.oth}, "
        f"только на интерпретации={stats.interpretation_only}, "
        f"doubtful с интерпретацией={stats.doubtful_with_interpretation}",
        f"  asr_quality по фрагментам: {quality}",
        f"  отказы аннотаций: {rejections}",
    ]
    return lines


__all__ = [
    "CODING_SCHEMA",
    "RELIABILITY_ATTRIBUTE",
    "CodingError",
    "CodingResult",
    "CodingStats",
    "DryRunCodingReport",
    "build_coding_prompt",
    "code_interview",
    "format_report",
    "load_codes",
    "load_prompt",
    "plan_dry_run",
    "prompt_hash",
    "require_reliability_attribute",
]
