"""Чтение состояния конвейера для команды `run` (M8).

Весь SQL, нужный оркестрации, живёт здесь: `qa.pipeline.orchestrate` получает
готовые числа и не знает о таблицах (CONSTITUTION §4). Отдельной таблицы
статусов нет — состояние вычисляется по фактическим данным, поэтому повторный
запуск ничего не портит.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass


def _params(raw: str | None) -> dict[str, object]:
    """Разобрать `params_json`; пустое или битое значение даёт пустой словарь."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


@dataclass(frozen=True)
class TermCandidateRow:
    """Кандидат `new_term` вместе с репликой-контекстом."""

    term_id: int
    surface_form: str
    context_utterance_id: int | None


@dataclass(frozen=True)
class UtteranceRow:
    """Реплика интервью с тайм-кодом первой записи.

    `timecode` — `MIN(turn.start_sec)` по записям реплики: тайм-код реплики
    (SPEC-M8, «Контракты данных»). `None`, если записей не осталось.
    """

    id: int
    speaker_role: str
    text: str
    is_backchannel: bool
    timecode: int | None


@dataclass(frozen=True)
class ManualEditsRow:
    """Разовые правки, применённые последней нормализацией интервью.

    `count` и `file_hash` берутся из `run.params_json` того прогона, который
    записал актуальную версию L2: файл правок мог измениться после прогона, и
    тогда в отчёте видно именно то, что попало в L2.
    """

    count: int
    file_hash: str


@dataclass(frozen=True)
class QualitySampleRow:
    """Реплика для выборки качества: L1 и L2 рядом."""

    utterance_id: int
    timecode: int | None
    text_l1: str
    text_l2: str
    asr_quality: str


class PipelineRepository:
    """Состояние стадий конвейера и данные для выгрузок M8."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def codebook_versions(self) -> int:
        """Сколько версий кодбука загружено."""
        row = self._conn.execute("SELECT COUNT(*) FROM codebook_version").fetchone()
        return int(row[0]) if row is not None else 0

    def lexicon_terms(self) -> int:
        """Сколько терминов в лексиконе."""
        row = self._conn.execute("SELECT COUNT(*) FROM lexicon").fetchone()
        return int(row[0]) if row is not None else 0

    def interview_db_id(self, interview_code: str) -> int | None:
        """id интервью по внешнему коду."""
        row = self._conn.execute("SELECT id FROM interview WHERE id_external = ?", (interview_code,)).fetchone()
        return int(row[0]) if row is not None else None

    def turn_count(self, interview_db_id: int) -> int:
        """Сколько записей транскрипта загружено."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM turn WHERE interview_id = ?",
            (interview_db_id,),
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def proposed_term_count(self, interview_db_id: int) -> int:
        """Сколько кандидатов терминов ждёт решения (ворота 1)."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM new_term WHERE interview_id = ? AND status = 'proposed'",
            (interview_db_id,),
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def text_unit_count(self, interview_db_id: int) -> int:
        """Сколько единиц L2 записано (любой версии)."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM text_unit t JOIN utterance u ON u.id = t.utterance_id WHERE u.interview_id = ?",
            (interview_db_id,),
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def fragment_count(self, interview_db_id: int) -> int:
        """Сколько фрагментов записано."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM fragment f JOIN utterance u ON u.id = f.utterance_id WHERE u.interview_id = ?",
            (interview_db_id,),
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def annotation_count(self, interview_db_id: int) -> int:
        """Сколько аннотаций записано."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM annotation a"
            " JOIN fragment f ON f.id = a.fragment_id"
            " JOIN utterance u ON u.id = f.utterance_id"
            " WHERE u.interview_id = ?",
            (interview_db_id,),
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def max_l2_version(self, interview_db_id: int) -> int:
        """Наибольшая версия L2 интервью (0 — единиц нет)."""
        row = self._conn.execute(
            "SELECT MAX(t.version) FROM text_unit t JOIN utterance u ON u.id = t.utterance_id WHERE u.interview_id = ?",
            (interview_db_id,),
        ).fetchone()
        return int(row[0]) if row is not None and row[0] is not None else 0

    def latest_run_started_at(self) -> str | None:
        """Время последнего прогона в формате SQLite (UTC) или None."""
        row = self._conn.execute("SELECT MAX(started_at) FROM run").fetchone()
        return None if row is None or row[0] is None else str(row[0])

    def latest_manual_edits(self, interview_db_id: int) -> ManualEditsRow | None:
        """Разовые правки, применённые прогоном актуальной версии L2.

        Считаются по `params_json` прогона, записавшего единицы L2: если файл
        правок менялся после прогона, в L2 всё равно лежит прежний результат.
        Прогон без ключей правок даёт None: правок не применялось.
        """
        row = self._conn.execute(
            "SELECT r.params_json FROM run r WHERE r.id = ("
            "   SELECT MAX(t.run_id) FROM text_unit t"
            "   JOIN utterance u ON u.id = t.utterance_id"
            "   WHERE u.interview_id = ? AND t.run_id IS NOT NULL)",
            (interview_db_id,),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        params = _params(str(row[0]))
        count = params.get("manual_edits")
        if not isinstance(count, int) or count <= 0:
            return None
        return ManualEditsRow(count=count, file_hash=str(params.get("manual_edits_hash") or ""))

    def proposed_terms(self, interview_db_id: int) -> list[TermCandidateRow]:
        """Кандидаты интервью в статусе `proposed` в порядке появления."""
        rows = self._conn.execute(
            "SELECT id, surface_form, context_utterance_id FROM new_term"
            " WHERE interview_id = ? AND status = 'proposed' ORDER BY id",
            (interview_db_id,),
        ).fetchall()
        return [
            TermCandidateRow(
                term_id=int(term_id),
                surface_form=str(form),
                context_utterance_id=None if context is None else int(context),
            )
            for term_id, form, context in rows
        ]

    def utterances_with_timecodes(self, interview_db_id: int) -> list[UtteranceRow]:
        """Реплики интервью с тайм-кодом первой записи, в порядке транскрипта."""
        rows = self._conn.execute(
            """
            SELECT u.id, u.speaker_role, u.text, u.is_backchannel,
                   (SELECT MIN(t.start_sec) FROM turn t, json_each(u.turn_ids) j
                     WHERE t.id = j.value)
              FROM utterance u
             WHERE u.interview_id = ?
             ORDER BY u.id
            """,
            (interview_db_id,),
        ).fetchall()
        return [
            UtteranceRow(
                id=int(utterance_id),
                speaker_role=str(role),
                text=str(text),
                is_backchannel=bool(flag),
                timecode=None if timecode is None else int(timecode),
            )
            for utterance_id, role, text, flag, timecode in rows
        ]

    def quality_sample_rows(self, interview_db_id: int, version: int) -> list[QualitySampleRow]:
        """Реплики респондента с содержимым и их L2 заданной версии."""
        rows = self._conn.execute(
            """
            SELECT u.id, u.text, t.text, t.asr_quality,
                   (SELECT MIN(turn.start_sec) FROM turn, json_each(u.turn_ids) j
                     WHERE turn.id = j.value)
              FROM utterance u
              JOIN text_unit t ON t.utterance_id = u.id AND t.version = ?
             WHERE u.interview_id = ? AND u.speaker_role = 'RESPONDENT' AND u.is_backchannel = 0
             ORDER BY u.id
            """,
            (version, interview_db_id),
        ).fetchall()
        return [
            QualitySampleRow(
                utterance_id=int(utterance_id),
                timecode=None if timecode is None else int(timecode),
                text_l1=str(text_l1),
                text_l2=str(text_l2),
                asr_quality=str(quality),
            )
            for utterance_id, text_l1, text_l2, quality, timecode in rows
        ]

    def utterance_ids(self, interview_db_id: int) -> set[int]:
        """Идентификаторы реплик интервью."""
        rows = self._conn.execute("SELECT id FROM utterance WHERE interview_id = ?", (interview_db_id,)).fetchall()
        return {int(row[0]) for row in rows}

    def turn_ids_of_utterance(self, utterance_id: int) -> list[int]:
        """id записей транскрипта, из которых собрана реплика."""
        row = self._conn.execute("SELECT turn_ids FROM utterance WHERE id = ?", (utterance_id,)).fetchone()
        if row is None or row[0] is None:
            return []
        try:
            values = json.loads(str(row[0]))
        except json.JSONDecodeError:
            return []
        if not isinstance(values, list):
            return []
        return [int(item) for item in values if isinstance(item, int)]
