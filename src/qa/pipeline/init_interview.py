"""init-interview: определить говорящих, предложить роли, подтвердить, записать в БД.

Код респондента берётся из имени файла, дата — из заголовка транскрипта.
Если лист «Интервью» уже заполнен, данные берутся оттуда и повторно не спрашиваются.
Таблица «имя -> код» создаётся вне проекта по NAMES_MAP_PATH.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from qa.db.workbook_repo import InterviewRepository
from qa.pipeline.load_transcript import (
    InterviewMeta,
    TranscriptMetaError,
    count_speaker_words,
    ensure_names_map_skeleton,
    format_roles_table,
    load_names_map,
    parse_header_date,
    parse_transcript,
    propose_roles,
    respondent_code_from_filename,
)

logger = logging.getLogger(__name__)

GROUP_TRADING = "ТОРГ"
GROUP_NON_TRADING = "не торгующие"
ALLOWED_GROUPS = (GROUP_TRADING, GROUP_NON_TRADING)


@dataclass
class InterviewPlan:
    """Что будет записано в БД и что нужно подтвердить оператору."""

    interview_id: str
    respondent_code: str
    group: str
    interview_date: str | None
    header: str
    word_counts: dict[str, int]
    roles: dict[str, str]
    from_workbook: bool = False
    from_db: bool = False
    existing: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def needs_confirmation(self) -> bool:
        """Спрашивать ли оператора: роли уже подтверждены в книге или в БД."""
        return not self.from_workbook


def normalize_group(value: str) -> str:
    """Привести группу к допустимому значению.

    Raises:
        TranscriptMetaError: группа вне списка.
    """
    text = value.strip()
    lowered = text.lower()
    if lowered in (GROUP_TRADING.lower(), "торг", "торгующие"):
        return GROUP_TRADING
    if lowered in (GROUP_NON_TRADING.lower(), "не_торгующие", "не торг", "нет"):
        return GROUP_NON_TRADING
    msg = f"Недопустимая группа {value!r}. Допустимы: {GROUP_TRADING!r}, {GROUP_NON_TRADING!r}"
    raise TranscriptMetaError(msg)


def build_plan(
    conn: sqlite3.Connection,
    transcript_path: Path,
    group: str | None,
) -> InterviewPlan:
    """Составить план интервью: код из имени файла, дата из заголовка, роли по словам.

    Если роли уже записаны в БД (лист «Интервью» или прошлый init-interview),
    они берутся оттуда и повторно не спрашиваются.
    """
    header, records = parse_transcript(transcript_path)
    respondent_code = respondent_code_from_filename(transcript_path)
    header_date = parse_header_date(header)

    repo = InterviewRepository(conn)
    stored = repo.find_by_external_id(respondent_code)
    if stored is None:
        if group is None:
            msg = 'Не задана группа респондента. Укажите --group ТОРГ или --group "не торгующие"'
            raise TranscriptMetaError(msg)
        word_counts = count_speaker_words(records)
        return InterviewPlan(
            interview_id=respondent_code,
            respondent_code=respondent_code,
            group=normalize_group(group),
            interview_date=header_date,
            header=header,
            word_counts=word_counts,
            roles=propose_roles(word_counts),
        )

    stored_speakers = repo.speakers(respondent_code)
    stored_date = str(stored["interview_date"]) if stored["interview_date"] else None
    stored_group = str(stored["group"]) if stored["group"] else None
    if stored_speakers:
        return InterviewPlan(
            interview_id=respondent_code,
            respondent_code=str(stored["respondent_code"] or respondent_code),
            group=str(group and normalize_group(group) or stored["group"] or GROUP_NON_TRADING),
            interview_date=stored_date or header_date,
            header=str(stored["name"] or header),
            word_counts={label: words for label, _role, words in stored_speakers},
            roles={label: role for label, role, _words in stored_speakers},
            from_workbook=True,
            from_db=True,
            existing=True,
            notes=["Роли взяты из БД (лист «Интервью» или прошлый init-interview); повторно не спрашиваются"],
        )

    if group is None and stored_group is None:
        msg = 'Не задана группа респондента. Укажите --group ТОРГ или --group "не торгующие"'
        raise TranscriptMetaError(msg)
    word_counts = count_speaker_words(records)
    stored_code = str(stored["respondent_code"] or respondent_code)
    return InterviewPlan(
        interview_id=respondent_code,
        respondent_code=stored_code,
        group=normalize_group(group) if group is not None else str(stored_group),
        interview_date=header_date or stored_date,
        header=str(stored["name"] or header),
        word_counts=word_counts,
        roles=propose_roles(word_counts),
        from_db=True,
        existing=True,
        notes=["Группа и дата взяты из БД; роли предложены по числу слов и требуют подтверждения"],
    )


def apply_plan(conn: sqlite3.Connection, plan: InterviewPlan) -> int:
    """Записать интервью и роли в БД (идемпотентно). Возвращает id интервью."""
    repo = InterviewRepository(conn)
    existing = repo.find_by_external_id(plan.interview_id)
    if existing is None:
        interview_db_id = repo.create(
            interview_id=plan.interview_id,
            name=plan.header,
            respondent_code=plan.respondent_code,
            group=plan.group,
            transcript_source=None,
            interview_date=plan.interview_date,
        )
    else:
        interview_db_id = int(existing["id"])
        repo.update_metadata(
            interview_db_id=interview_db_id,
            name=existing["name"] or plan.header,
            respondent_code=plan.respondent_code,
            group=plan.group,
            transcript_source=existing["transcript_source"],
            interview_date=plan.interview_date or existing["interview_date"],
        )
    for label, role in plan.roles.items():
        repo.add_speaker(interview_db_id, label, role, plan.word_counts.get(label, 0))
    conn.commit()
    return interview_db_id


def create_names_map_skeleton(plan: InterviewPlan) -> Path | None:
    """Создать заготовку «имя -> код» вне проекта, если задан NAMES_MAP_PATH."""
    import os

    env_path = os.environ.get("NAMES_MAP_PATH")
    if not env_path:
        return None
    return ensure_names_map_skeleton(env_path, list(plan.roles))


def plan_to_meta(plan: InterviewPlan, names_map: dict[str, str] | None = None) -> InterviewMeta:
    """Метаданные для загрузки транскрипта из плана интервью."""
    return InterviewMeta(
        interview_id=plan.interview_id,
        respondent_code=plan.respondent_code,
        group=plan.group,
        roles=dict(plan.roles),
        names_to_codes=dict(names_map or {}),
        date=plan.interview_date,
        transcript_source=None,
    )


def resolve_names_map() -> dict[str, str]:
    """Прочитать таблицу «имя -> код» по NAMES_MAP_PATH (если задана)."""
    import os

    env_path = os.environ.get("NAMES_MAP_PATH")
    if not env_path:
        return {}
    return load_names_map(env_path)


def render_plan(plan: InterviewPlan) -> str:
    """Текст для показа оператору перед записью."""
    lines = [
        f"Интервью: {plan.interview_id}",
        f"Код респондента: {plan.respondent_code}",
        f"Группа: {plan.group}",
        f"Дата: {plan.interview_date or 'не найдена в заголовке'}",
        f"Заголовок: {plan.header}",
        format_roles_table(plan.word_counts, plan.roles),
    ]
    lines.extend(f"  ! {note}" for note in plan.notes)
    return "\n".join(lines)
