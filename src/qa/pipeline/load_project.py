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
    GoldenRepository,
    InterviewRepository,
    LexiconRepository,
    MarkerRepository,
    ReferenceFactRepository,
    WorkbookRepository,
)
from qa.pipeline.workbook import WorkbookData, read_workbook
from qa.settings import GroupsSettings, WorkbookSettings, normalize_lexicon_status

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
) -> None:
    """Загрузить метаданные интервью и роли говорящих из книги."""
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
            words = row.get("words")
            speakers.add_speaker(int(existing["id"]), label, role, int(words) if isinstance(words, int) else 0)
            result.counts["speakers"] = result.counts.get("speakers", 0) + 1
    else:
        result.missing_sheets.append(SHEET_SPEAKERS)


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
) -> LoadProjectResult:
    """Загрузить необязательные листы проекта и умолчания.

    Args:
        groups: канонические значения групп; неизвестное сохраняется с предупреждением.
        lexicon_status_map: приведение статусов лексикона к proposed/approved/rejected.

    Raises:
        FileNotFoundError: файл не найден.
        WorkbookError: нет обязательного листа/колонки или книга не загружена.
    """
    workbook = read_workbook(workbook_path, workbook_settings)
    workbook_repo = WorkbookRepository(conn)
    workbook_version_id = workbook_repo.find_by_hash(workbook.file_hash)
    if workbook_version_id is None:
        msg = f"Книга {workbook_path} ещё не загружена командой load-codebook; сначала загрузите кодбук, затем проект"
        raise RuntimeError(msg)

    result = LoadProjectResult(workbook_version_id=workbook_version_id)
    _load_lexicon(workbook, LexiconRepository(conn), lexicon_status_map or {}, result)
    _load_markers(workbook, MarkerRepository(conn), workbook_settings, result)
    interviews = InterviewRepository(conn, groups)
    _load_interviews(workbook, interviews, interviews, result)
    _load_golden(workbook, GoldenRepository(conn), result)
    _load_facts(workbook, workbook_version_id, ReferenceFactRepository(conn), workbook_settings, result)
    conn.commit()

    logger.info("Проектные листы загружены: %s", result.counts)
    return result
