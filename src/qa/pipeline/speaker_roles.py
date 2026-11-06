"""Применение ролей говорящих из листа «Роли_говорящих» к записям и репликам.

Лист «Роли_говорящих» — интерфейс оператора: человек правит роль в книге и запускает
`load-project`. Раньше правка доезжала только до `interview_speaker`, а анализ читает
`turn.speaker_role` и `utterance.speaker_role`, поэтому цифры не менялись молча.

Здесь роль доводится до конца: `interview_speaker` -> `turn` -> `utterance`.
Реплики пересобираются, потому что их границы зависят от того, кто говорит.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from qa.db.workbook_repo import InterviewRepository
from qa.pipeline.load_transcript import (
    ROLES,
    build_utterances,
    parse_transcript,
)
from qa.settings import GroupsSettings, PipelineSettings

logger = logging.getLogger(__name__)


class SpeakerRolesError(Exception):
    """Роли нельзя применить без потери данных."""


@dataclass
class SpeakerRolesResult:
    """Что изменилось при применении ролей."""

    interview_id: str
    labels_filled: int = 0
    speakers_updated: int = 0
    turns_updated: int = 0
    utterances_rebuilt: int = 0
    notes: list[str] = field(default_factory=list)


def _labels_from_transcript(transcript_path: Path) -> dict[int, str]:
    """Карта «номер строки -> метка говорящего» из файла транскрипта."""
    _header, records = parse_transcript(transcript_path)
    return {record.line_no: record.speaker_label for record in records}


def _fill_missing_labels(
    repo: InterviewRepository,
    interview_db_id: int,
    transcript_path: Path | None,
    result: SpeakerRolesResult,
) -> None:
    """Заполнить метки в записях, загруженных до появления колонки turn.speaker_label."""
    stored = repo.turn_line_labels(interview_db_id)
    missing = [line_no for line_no, label in stored if label is None]
    if not missing:
        return
    if transcript_path is None or not transcript_path.exists():
        result.notes.append(
            f"Интервью {result.interview_id}: у {len(missing)} записей нет метки говорящего,"
            " транскрипт не найден — роли к ним не применимы"
        )
        return
    labels_by_line = _labels_from_transcript(transcript_path)
    filled = {line_no: labels_by_line[line_no] for line_no in missing if line_no in labels_by_line}
    if filled:
        result.labels_filled = repo.fill_turn_labels(interview_db_id, filled)
    unknown = [line_no for line_no in missing if line_no not in labels_by_line]
    if unknown:
        result.notes.append(
            f"Интервью {result.interview_id}: строки {unknown[:5]} отсутствуют в транскрипте — метка не заполнена"
        )


def _check_rebuild_allowed(repo: InterviewRepository, interview_db_id: int, interview_id: str) -> None:
    """Пересборка реплик меняет их id, а на id могут ссылаться кандидаты новых терминов."""
    linked = repo.new_terms_on_utterances(interview_db_id)
    if linked:
        msg = (
            f"Интервью {interview_id}: на реплики ссылается кандидатов новых терминов — {linked}."
            " Пересборка реплик потеряет эти ссылки. Сначала закройте ревью (review-terms)"
        )
        raise SpeakerRolesError(msg)


def backfill_speaker_labels(
    conn: sqlite3.Connection,
    interview_id: str,
    transcript_path: Path | None,
    *,
    groups: GroupsSettings | None = None,
) -> tuple[int, list[str]]:
    """Заполнить метки говорящих в записях, загруженных до миграции 0004.

    Returns:
        (сколько записей заполнено, заметки).
    """
    repo = InterviewRepository(conn, groups)
    interview = repo.find_by_external_id(interview_id)
    if interview is None:
        return 0, []
    result = SpeakerRolesResult(interview_id=interview_id)
    _fill_missing_labels(repo, int(interview["id"]), transcript_path, result)
    return result.labels_filled, result.notes


def apply_speaker_roles(
    conn: sqlite3.Connection,
    interview_id: str,
    roles: dict[str, str],
    pipeline: PipelineSettings,
    *,
    transcript_path: Path | None = None,
    groups: GroupsSettings | None = None,
) -> SpeakerRolesResult:
    """Применить роли из листа «Роли_говорящих» к записям и репликам интервью.

    Args:
        conn: подключение к SQLite (миграции уже применены).
        interview_id: внешний идентификатор интервью (`id_интервью`).
        roles: карта «метка говорящего -> роль».
        pipeline: настройки сборки реплик (слова-подтверждения, лимит слов).
        transcript_path: файл транскрипта для заполнения недостающих меток.
        groups: канонические значения групп.

    Raises:
        SpeakerRolesError: ссылки new_term мешают пересборке реплик.
        ValueError: неизвестная роль.
    """
    invalid = {label: role for label, role in roles.items() if role not in ROLES}
    if invalid:
        msg = f"Недопустимые роли: {invalid}. Допустимы: {sorted(ROLES)}"
        raise ValueError(msg)

    repo = InterviewRepository(conn, groups)
    interview = repo.find_by_external_id(interview_id)
    if interview is None:
        msg = f"Интервью {interview_id} нет в БД: сначала загрузите транскрипт"
        raise SpeakerRolesError(msg)
    interview_db_id = int(interview["id"])
    result = SpeakerRolesResult(interview_id=interview_id)

    _fill_missing_labels(repo, interview_db_id, transcript_path, result)

    if not roles:
        result.notes.append(f"Интервью {interview_id}: в листе «Роли_говорящих» нет строк, роли не менялись")
        return result

    _check_rebuild_allowed(repo, interview_db_id, interview_id)

    conn.execute("SAVEPOINT apply_speaker_roles")
    try:
        for label, role in roles.items():
            result.speakers_updated += repo.update_speaker_role(interview_db_id, label, role)
            result.turns_updated += repo.update_turn_roles(interview_db_id, label, role)

        turns = repo.turns_for_utterances(interview_db_id)
        drafts = build_utterances(turns, set(pipeline.backchannel_words), pipeline.backchannel_max_words)
        repo.clear_utterances(interview_db_id)
        for draft in drafts:
            repo.insert_utterance(
                interview_db_id,
                draft.speaker_role,
                draft.text,
                draft.turn_ids,
                draft.is_backchannel,
            )
        result.utterances_rebuilt = len(drafts)
    except BaseException:
        conn.execute("ROLLBACK TO SAVEPOINT apply_speaker_roles")
        conn.execute("RELEASE SAVEPOINT apply_speaker_roles")
        raise
    conn.execute("RELEASE SAVEPOINT apply_speaker_roles")
    return result
