"""Загрузка необязательных листов проекта: лексикон, маркеры, интервью, золотой набор, факты.

Отсутствующий лист не ошибка: используются умолчания из config.yaml.
Решения, принятые в review-terms, имеют приоритет над листом и не перезаписываются.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qa.db.workbook_repo import (
    CodebookRepository,
    GoldenRepository,
    InterviewRepository,
    LexiconRepository,
    MarkerRepository,
    ReferenceFactRepository,
    WorkbookRepository,
)
from qa.pipeline.load_transcript import ROLES
from qa.pipeline.speaker_roles import SpeakerRolesError, apply_speaker_roles, backfill_speaker_labels
from qa.pipeline.workbook import WorkbookData, read_workbook
from qa.settings import GroupsSettings, PipelineSettings, WorkbookSettings, normalize_lexicon_status

logger = logging.getLogger(__name__)

SHEET_LEXICON = "Лексикон"
SHEET_MARKERS = "Маркеры_неуверенности"
SHEET_INTERVIEWS = "Интервью"
SHEET_SPEAKERS = "Роли_говорящих"
SHEET_GOLDEN = "Золотой_набор"


@dataclass
class LoadProjectResult:
    """Итог загрузки необязательных листов."""

    workbook_version_id: int
    counts: dict[str, int] = field(default_factory=dict)
    discrepancies: list[str] = field(default_factory=list)
    missing_sheets: list[str] = field(default_factory=list)


def _text(row: dict[str, Any], key: str) -> str | None:
    """Строковое значение поля строки."""
    value = row.get(key)
    return None if value is None else str(value)


def _role_value(value: str, interview_id: str, result: LoadProjectResult) -> str:
    """Роль говорящего в каноническом виде; неизвестная роль не пишется."""
    role = value.strip().upper()
    if role not in ROLES:
        result.discrepancies.append(
            f"Роли_говорящих: интервью '{interview_id}', роль {value!r} не распознана (допустимы: {sorted(ROLES)})"
        )
        return value
    return role


def _list(row: dict[str, Any], key: str) -> list[str]:
    """Список значений поля строки."""
    value = row.get(key)
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _load_lexicon(
    workbook: WorkbookData,
    repo: LexiconRepository,
    status_map: dict[str, str],
    result: LoadProjectResult,
) -> None:
    """Загрузить лексикон; расхождения с review-terms вывести в отчёт.

    Статус приводится к proposed/approved/rejected по lexicon_status_map;
    неизвестное значение даёт предупреждение и 'proposed'.
    """
    sheet = workbook.sheet(SHEET_LEXICON)
    if sheet is None:
        result.missing_sheets.append(SHEET_LEXICON)
        return
    for row, row_number in zip(sheet.rows, sheet.row_numbers, strict=True):
        canonical = _text(row, "canonical")
        if canonical is None:
            continue
        variants = _list(row, "variants")
        definition = _text(row, "definition")
        raw_status = _text(row, "status")
        status, recognized = normalize_lexicon_status(raw_status, status_map)
        if not recognized:
            result.discrepancies.append(
                f"Лексикон, строка {row_number}: неизвестный статус {raw_status!r} у '{canonical}', принят 'proposed'"
            )
        applied = repo.upsert_from_workbook(canonical, variants, definition, status)
        if not applied:
            stored = repo.all_terms().get(canonical, {})
            stored_variants = stored.get("variants") or []
            stored_status = stored.get("status")
            if stored_variants != variants or stored_status != status:
                result.discrepancies.append(
                    f"Лексикон: '{canonical}' сохранено решение review-terms"
                    f" (статус {stored_status!r}, вариантов {len(stored_variants)}),"
                    f" лист предлагает {status!r} и {len(variants)} вариантов — лист не перезаписан"
                )
        result.counts["lexicon"] = result.counts.get("lexicon", 0) + 1


def _load_markers(
    workbook: WorkbookData,
    repo: MarkerRepository,
    workbook_settings: WorkbookSettings,
    result: LoadProjectResult,
) -> None:
    """Загрузить маркеры неуверенности; без листа — умолчания из config.yaml."""
    sheet = workbook.sheet(SHEET_MARKERS)
    if sheet is not None and not sheet.is_empty:
        for row in sheet.rows:
            expression = _text(row, "expression")
            if expression is None:
                continue
            repo.upsert(expression, _text(row, "note"), "workbook")
        result.counts["uncertainty_markers"] = len(sheet.rows)
        return
    result.missing_sheets.append(SHEET_MARKERS)
    for expression in workbook_settings.defaults.get("uncertainty_markers", []):
        repo.upsert(expression, None, "default")
    result.counts["uncertainty_markers"] = len(workbook_settings.defaults.get("uncertainty_markers", []))


def _load_interviews(
    workbook: WorkbookData,
    interviews: InterviewRepository,
    speakers: InterviewRepository,
    result: LoadProjectResult,
) -> dict[str, dict[str, str]]:
    """Загрузить метаданные интервью и роли говорящих из книги.

    Returns:
        Карта «id_интервью -> {метка говорящего: роль}» для применения к записям.
    """
    roles_by_interview: dict[str, dict[str, str]] = {}
    sheet = workbook.sheet(SHEET_INTERVIEWS)
    if sheet is not None and not sheet.is_empty:
        for row in sheet.rows:
            interview_id = _text(row, "interview_id")
            if interview_id is None:
                continue
            existing = interviews.find_by_external_id(interview_id)
            respondent_code = _text(row, "respondent_code") or interview_id
            if existing is None:
                interviews.create(
                    interview_id=interview_id,
                    name=None,
                    respondent_code=respondent_code,
                    group=_text(row, "group"),
                    transcript_source=_text(row, "transcript_source"),
                    interview_date=_text(row, "date"),
                )
            else:
                interviews.update_metadata(
                    interview_db_id=int(existing["id"]),
                    name=existing["name"],
                    respondent_code=respondent_code,
                    group=_text(row, "group") or existing["group"],
                    transcript_source=_text(row, "transcript_source") or existing["transcript_source"],
                    interview_date=_text(row, "date") or existing["interview_date"],
                )
            result.counts["interviews"] = result.counts.get("interviews", 0) + 1
    else:
        result.missing_sheets.append(SHEET_INTERVIEWS)

    speaker_sheet = workbook.sheet(SHEET_SPEAKERS)
    if speaker_sheet is not None and not speaker_sheet.is_empty:
        for row in speaker_sheet.rows:
            interview_id = _text(row, "interview_id")
            label = _text(row, "speaker_label")
            role = _text(row, "role")
            if interview_id is None or label is None or role is None:
                continue
            existing = interviews.find_by_external_id(interview_id)
            if existing is None:
                result.discrepancies.append(
                    f"Роли_говорящих: интервью '{interview_id}' нет в листе «Интервью» — строка пропущена"
                )
                continue
            role = _role_value(role, interview_id, result)
            words = row.get("words")
            speakers.add_speaker(int(existing["id"]), label, role, int(words) if isinstance(words, int) else 0)
            roles_by_interview.setdefault(interview_id, {})[label] = role
            result.counts["speakers"] = result.counts.get("speakers", 0) + 1
    else:
        result.missing_sheets.append(SHEET_SPEAKERS)
    return roles_by_interview


def _transcript_path(workbook: WorkbookData, interview_id: str, transcripts_dir: Path | None) -> Path | None:
    """Путь к транскрипту интервью по колонке «файл_транскрипта» листа «Интервью»."""
    if transcripts_dir is None:
        return None
    sheet = workbook.sheet(SHEET_INTERVIEWS)
    for row in sheet.rows if sheet is not None else []:
        if _text(row, "interview_id") != interview_id:
            continue
        file_name = _text(row, "transcript_file")
        if not file_name:
            return None
        return Path(transcripts_dir) / file_name
    return None


def _apply_roles(
    conn: sqlite3.Connection,
    workbook: WorkbookData,
    roles_by_interview: dict[str, dict[str, str]],
    result: LoadProjectResult,
    *,
    pipeline: PipelineSettings,
    transcripts_dir: Path | None,
    groups: GroupsSettings | None,
) -> None:
    """Довести роли из листа до записей turn и реплик utterance."""
    for interview_id, roles in roles_by_interview.items():
        transcript = _transcript_path(workbook, interview_id, transcripts_dir)
        try:
            applied = apply_speaker_roles(
                conn,
                interview_id,
                roles,
                pipeline,
                transcript_path=transcript,
                groups=groups,
            )
        except SpeakerRolesError as exc:
            result.discrepancies.append(str(exc))
            continue
        result.counts["speaker_roles_applied"] = result.counts.get("speaker_roles_applied", 0) + 1
        result.counts["utterances_rebuilt"] = result.counts.get("utterances_rebuilt", 0) + applied.utterances_rebuilt
        result.discrepancies.extend(applied.notes)
        logger.info(
            "Роли применены: %s, записей=%s, реплик=%s",
            interview_id,
            applied.turns_updated,
            applied.utterances_rebuilt,
        )


def _backfill_labels(
    conn: sqlite3.Connection,
    workbook: WorkbookData,
    result: LoadProjectResult,
    *,
    transcripts_dir: Path | None,
    groups: GroupsSettings | None,
) -> None:
    """Заполнить метки говорящих в записях, загруженных до миграции 0004.

    Нужно и при пустом листе «Роли_говорящих»: без меток будущая правка роли
    не сможет дойти до записей, и это осталось бы незамеченным.
    """
    sheet = workbook.sheet(SHEET_INTERVIEWS)
    for row in sheet.rows if sheet is not None else []:
        interview_id = _text(row, "interview_id")
        if interview_id is None:
            continue
        transcript = _transcript_path(workbook, interview_id, transcripts_dir)
        filled, notes = backfill_speaker_labels(conn, interview_id, transcript, groups=groups)
        if filled:
            result.counts["speaker_labels_filled"] = result.counts.get("speaker_labels_filled", 0) + filled
        result.discrepancies.extend(notes)


def _load_golden(workbook: WorkbookData, repo: GoldenRepository, result: LoadProjectResult) -> None:
    """Загрузить золотой набор из листа книги."""
    sheet = workbook.sheet(SHEET_GOLDEN)
    if sheet is None or sheet.is_empty:
        result.missing_sheets.append(SHEET_GOLDEN)
        return
    for row in sheet.rows:
        text_l1 = _text(row, "text_l1")
        if text_l1 is None:
            continue
        repo.upsert(
            interview_code=_text(row, "interview_code"),
            utterance_ref=_text(row, "utterance_ref"),
            text_l1=text_l1,
            text_l2_manual=_text(row, "text_l2_manual"),
            source="workbook",
        )
        result.counts["golden"] = result.counts.get("golden", 0) + 1


def _load_facts(
    workbook: WorkbookData,
    workbook_version_id: int,
    repo: ReferenceFactRepository,
    workbook_settings: WorkbookSettings,
    result: LoadProjectResult,
) -> None:
    """Загрузить листы с префиксом «Факты_»."""
    found = False
    for name, sheet in workbook.sheets.items():
        if not any(name.startswith(prefix) for prefix in workbook_settings.optional_prefixes):
            continue
        found = True
        for row in sheet.rows:
            if _text(row, "external_id") is None:
                continue
            repo.insert(workbook_version_id, name, row)
            result.counts["facts"] = result.counts.get("facts", 0) + 1
    if not found:
        result.missing_sheets.append("Факты_*")


def load_project(
    conn: sqlite3.Connection,
    workbook_path: Path,
    workbook_settings: WorkbookSettings,
    *,
    groups: GroupsSettings | None = None,
    lexicon_status_map: dict[str, str] | None = None,
    pipeline: PipelineSettings | None = None,
    transcripts_dir: Path | None = None,
) -> LoadProjectResult:
    """Загрузить необязательные листы проекта и умолчания.

    Args:
        groups: канонические значения групп; неизвестное сохраняется с предупреждением.
        lexicon_status_map: приведение статусов лексикона к proposed/approved/rejected.
        pipeline: настройки сборки реплик; нужны, чтобы применить роли к записям.
        transcripts_dir: каталог транскриптов; по нему заполняются недостающие метки.

    Raises:
        FileNotFoundError: файл не найден.
        WorkbookError: нет обязательного листа/колонки или книга не загружена.
    """
    workbook = read_workbook(workbook_path, workbook_settings)
    workbook_repo = WorkbookRepository(conn)
    codebook_repo = CodebookRepository(conn, groups)
    # Правка листа «Роли_говорящих» меняет хеш файла, но не кодбук: версию ищем
    # по хешу кодбуковых листов, иначе каждая правка роли требовала бы load-codebook.
    workbook_version_id = workbook_repo.find_by_hash(workbook.file_hash)
    if workbook_version_id is None:
        codebook_version_id = codebook_repo.find_by_codebook_hash(workbook.codebook_hash)
        if codebook_version_id is not None:
            row = conn.execute(
                "SELECT workbook_version_id FROM codebook_version WHERE id = ?",
                (codebook_version_id,),
            ).fetchone()
            if row is not None:
                workbook_version_id = int(row[0])
    if workbook_version_id is None:
        msg = f"Книга {workbook_path} ещё не загружена командой load-codebook; сначала загрузите кодбук, затем проект"
        raise RuntimeError(msg)

    result = LoadProjectResult(workbook_version_id=workbook_version_id)
    _load_lexicon(workbook, LexiconRepository(conn), lexicon_status_map or {}, result)
    _load_markers(workbook, MarkerRepository(conn), workbook_settings, result)
    interviews = InterviewRepository(conn, groups)
    roles_by_interview = _load_interviews(workbook, interviews, interviews, result)
    _backfill_labels(conn, workbook, result, transcripts_dir=transcripts_dir, groups=groups)
    if pipeline is not None and roles_by_interview:
        _apply_roles(
            conn,
            workbook,
            roles_by_interview,
            result,
            pipeline=pipeline,
            transcripts_dir=transcripts_dir,
            groups=groups,
        )
    _load_golden(workbook, GoldenRepository(conn), result)
    _load_facts(workbook, workbook_version_id, ReferenceFactRepository(conn), workbook_settings, result)
    conn.commit()

    logger.info("Проектные листы загружены: %s", result.counts)
    return result
