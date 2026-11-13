"""Чтение данных для агрегации покрытия (M4/M5).

Весь SQL, нужный дашборду и отчёту, живёт здесь: агрегация и рендер получают
готовые строки и не знают о таблицах (SPEC-M6, «Разделение слоёв»).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InterviewRow:
    """Интервью в объёме, нужном для шапки артефакта."""

    id: int
    code: str
    group: str | None
    transcript_source: str | None
    interview_date: str | None


@dataclass(frozen=True)
class RunRow:
    """Прогон для блока происхождения отчёта."""

    id: int
    started_at: str | None
    model: str | None
    prompt_hash: str | None
    codebook_version_id: int | None
    code_version: str | None
    stage: str


@dataclass(frozen=True)
class CodingRunRow:
    """Прогон кодирования: версии кодбука и L2 и происхождение."""

    id: int
    model: str | None
    prompt_hash: str | None
    codebook_version_id: int | None
    code_version: str | None
    started_at: str | None
    l2_version: int | None
    stage: str


@dataclass(frozen=True)
class ThemeRow:
    """Узел или тема дерева."""

    id: int
    name: str
    parent_id: int | None
    level: int


@dataclass(frozen=True)
class CodeRow:
    """Код текущей версии кодбука."""

    id: int
    name: str
    family: str | None


@dataclass(frozen=True)
class QuestionRow:
    """Вопрос гайда с группой применимости."""

    id: int
    identifier: str
    group: str | None


@dataclass(frozen=True)
class QuestionCodeRow:
    """Связь вопроса гайда с кодом."""

    question_id: int
    code_id: int


@dataclass(frozen=True)
class AnnotationRow:
    """Аннотация с контекстом цитаты и фрагмента.

    `verified` — признак ручной проверки: статус `accepted` или `edited`.
    Непроверенные (`proposed`) входят в агрегаты наравне: выборочный контроль
    не является фильтром, иначе числа описывают подвыборку, а не разметку.

    `timecode` — тайм-код реплики в секундах (`MIN(turn.start_sec)` по её
    записям), `l1_text` — исходный текст реплики L1. Оба нужны отчёту, чтобы
    рядом с цитатой стоял проверяемый оригинал (SPEC-M8, C.4).
    """

    id: int
    code_id: int
    fragment_id: int
    utterance_id: int
    start: int
    quote: str
    quote_start: int | None
    quote_end: int | None
    reliability: int | None
    fragment_quality: str
    l2_text: str
    question_identifier: str | None
    verified: bool
    interpretation_only: bool
    timecode: int | None = None
    l1_text: str = ""


def run_stage(params: dict[str, Any]) -> str:
    """Стадия прогона по `params_json`: в таблице `run` отдельной колонки нет.

    Кодирование пишет `fragments`, структура — `blocks`, нормализация — `version`
    или `l2_version`. Проверка `fragments` идёт первой: прогон кодирования
    содержит и номер версии L2.
    """
    if "fragments" in params:
        return "coding"
    if "blocks" in params:
        return "structure"
    if "l2_version" in params or "version" in params:
        return "normalize"
    return "unknown"


def _parse_params(raw: str | None) -> dict[str, Any]:
    """Разобрать `params_json`; пустое или битое значение даёт пустой словарь."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class AnalyticsRepository:
    """Запросы покрытия дерева тем, кодбука и качества данных."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def interview_by_code(self, interview_code: str) -> InterviewRow | None:
        """Интервью по внешнему коду; None, если интервью нет в БД."""
        row = self._conn.execute(
            'SELECT id, id_external, "group", transcript_source, interview_date FROM interview WHERE id_external = ?',
            (interview_code,),
        ).fetchone()
        if row is None:
            return None
        return InterviewRow(
            id=int(row[0]),
            code=str(row[1]),
            group=None if row[2] is None else str(row[2]),
            transcript_source=None if row[3] is None else str(row[3]),
            interview_date=None if row[4] is None else str(row[4]),
        )

    def latest_coding_run(self, interview_db_id: int) -> CodingRunRow | None:
        """Последний прогон, по которому есть итоги кодирования фрагментов."""
        row = self._conn.execute(
            "SELECT r.id, r.model, r.prompt_hash, r.codebook_version_id, r.code_version, r.started_at,"
            " r.params_json"
            " FROM run r WHERE r.id = ("
            "   SELECT MAX(fc.run_id) FROM fragment_coding fc"
            "   JOIN fragment f ON f.id = fc.fragment_id"
            "   JOIN utterance u ON u.id = f.utterance_id"
            "   WHERE u.interview_id = ?)",
            (interview_db_id,),
        ).fetchone()
        if row is None:
            return None
        params = _parse_params(None if row[6] is None else str(row[6]))
        raw_l2 = params.get("l2_version")
        return CodingRunRow(
            id=int(row[0]),
            model=None if row[1] is None else str(row[1]),
            prompt_hash=None if row[2] is None else str(row[2]),
            codebook_version_id=None if row[3] is None else int(row[3]),
            code_version=None if row[4] is None else str(row[4]),
            started_at=None if row[5] is None else str(row[5]),
            l2_version=raw_l2 if isinstance(raw_l2, int) else None,
            stage=run_stage(params),
        )

    def interview_runs(self, interview_id: int) -> list[RunRow]:
        """Прогоны, которые затрагивали это интервью, в порядке запуска.

        Прогон относится к интервью, если оставил след в его данных: запись
        фрагментов (`fragment`), статусы фрагментов (`fragment_coding`) или
        аннотации. Отбор по данным, а не по времени: глобальный прогон другого
        интервью в блок происхождения не попадает.
        """
        rows = self._conn.execute(
            """
            SELECT r.id, r.started_at, r.model, r.prompt_hash, r.codebook_version_id,
                   r.code_version, r.params_json
              FROM run r
             WHERE r.id IN (
                     SELECT fc.run_id FROM fragment_coding fc
                       JOIN fragment f ON f.id = fc.fragment_id
                       JOIN utterance u ON u.id = f.utterance_id
                      WHERE u.interview_id = :iid
                     UNION
                     SELECT a.run_id FROM annotation a
                       JOIN fragment f ON f.id = a.fragment_id
                       JOIN utterance u ON u.id = f.utterance_id
                      WHERE u.interview_id = :iid AND a.run_id IS NOT NULL
             )
             ORDER BY r.id
            """,
            {"iid": interview_id},
        ).fetchall()
        return [
            RunRow(
                id=int(row[0]),
                started_at=None if row[1] is None else str(row[1]),
                model=None if row[2] is None else str(row[2]),
                prompt_hash=None if row[3] is None else str(row[3]),
                codebook_version_id=None if row[4] is None else int(row[4]),
                code_version=None if row[5] is None else str(row[5]),
                stage=run_stage(_parse_params(None if row[6] is None else str(row[6]))),
            )
            for row in rows
        ]

    def latest_codebook_version_id(self) -> int | None:
        """Последняя загруженная версия кодбука."""
        row = self._conn.execute("SELECT id FROM codebook_version ORDER BY id DESC LIMIT 1").fetchone()
        return int(row[0]) if row is not None else None

    def themes(self) -> list[ThemeRow]:
        """Все узлы и темы дерева в порядке загрузки."""
        rows = self._conn.execute("SELECT id, name, parent_id, level FROM theme ORDER BY level, id").fetchall()
        return [
            ThemeRow(
                id=int(row[0]),
                name=str(row[1]),
                parent_id=None if row[2] is None else int(row[2]),
                level=int(row[3]),
            )
            for row in rows
        ]

    def theme_codes(self) -> list[tuple[int, int]]:
        """Связи «тема — код» как пары (theme_id, code_id)."""
        rows = self._conn.execute("SELECT theme_id, code_id FROM theme_code ORDER BY theme_id, code_id").fetchall()
        return [(int(theme_id), int(code_id)) for theme_id, code_id in rows]

    def codes(self, codebook_version_id: int) -> list[CodeRow]:
        """Коды версии кодбука."""
        rows = self._conn.execute(
            "SELECT id, name, family FROM code WHERE codebook_version_id = ? ORDER BY name",
            (codebook_version_id,),
        ).fetchall()
        return [
            CodeRow(id=int(row[0]), name=str(row[1]), family=None if row[2] is None else str(row[2])) for row in rows
        ]

    def questions(self, codebook_version_id: int) -> list[QuestionRow]:
        """Вопросы гайда версии кодбука."""
        rows = self._conn.execute(
            'SELECT id, identifier, "group" FROM question WHERE codebook_version_id = ? ORDER BY id',
            (codebook_version_id,),
        ).fetchall()
        return [
            QuestionRow(
                id=int(row[0]),
                identifier=str(row[1]),
                group=None if row[2] is None else str(row[2]),
            )
            for row in rows
        ]

    def question_codes(self, codebook_version_id: int) -> list[QuestionCodeRow]:
        """Связи «вопрос — код» версии кодбука."""
        rows = self._conn.execute(
            "SELECT qc.question_id, qc.code_id FROM question_code qc"
            " JOIN question q ON q.id = qc.question_id"
            " WHERE q.codebook_version_id = ?",
            (codebook_version_id,),
        ).fetchall()
        return [QuestionCodeRow(question_id=int(row[0]), code_id=int(row[1])) for row in rows]

    def annotations(self, interview_db_id: int) -> list[AnnotationRow]:
        """Все аннотации интервью, кроме отклонённых, по порядку реплик.

        База агрегатов — вся разметка модели без `rejected`: выборочный контроль
        не отбрасывает непроверенное. Признак проверки выводится отдельной
        колонкой, а не фильтром.

        Текст цитаты берётся из L2 той версии, на которой построены фрагменты:
        позиции `quote_start`/`quote_end` относятся к ней, а не к L1.
        """
        rows = self._conn.execute(
            'SELECT a.id, a.code_id, a.fragment_id, f.utterance_id, f."start", a.quote,'
            " a.quote_start, a.quote_end, a.reliability, f.asr_quality, t.text, q.identifier,"
            " a.status, a.interpretation_only, u.text,"
            " (SELECT MIN(turn.start_sec) FROM turn, json_each(u.turn_ids) j WHERE turn.id = j.value)"
            " FROM annotation a"
            " JOIN fragment f ON f.id = a.fragment_id"
            " JOIN utterance u ON u.id = f.utterance_id"
            " JOIN text_unit t ON t.id = f.text_unit_id"
            " LEFT JOIN question q ON q.id = u.question_id"
            " WHERE u.interview_id = ? AND a.status <> 'rejected'"
            ' ORDER BY f.utterance_id, f."start", a.id',
            (interview_db_id,),
        ).fetchall()
        return [
            AnnotationRow(
                id=int(row[0]),
                code_id=int(row[1]),
                fragment_id=int(row[2]),
                utterance_id=int(row[3]),
                start=int(row[4]),
                quote=str(row[5]),
                quote_start=None if row[6] is None else int(row[6]),
                quote_end=None if row[7] is None else int(row[7]),
                reliability=None if row[8] is None else int(row[8]),
                fragment_quality=str(row[9]),
                l2_text=str(row[10]),
                question_identifier=None if row[11] is None else str(row[11]),
                verified=str(row[12]) in ("accepted", "edited"),
                interpretation_only=bool(row[13]),
                l1_text=str(row[14]),
                timecode=None if row[15] is None else int(row[15]),
            )
            for row in rows
        ]

    def annotation_status_counts(self, interview_db_id: int) -> dict[str, int]:
        """Число аннотаций по статусам для интервью."""
        rows = self._conn.execute(
            "SELECT a.status, COUNT(*) FROM annotation a"
            " JOIN fragment f ON f.id = a.fragment_id"
            " JOIN utterance u ON u.id = f.utterance_id"
            " WHERE u.interview_id = ? GROUP BY a.status",
            (interview_db_id,),
        ).fetchall()
        return {str(status): int(count) for status, count in rows}

    def fragment_coding_counts(self, interview_db_id: int, run_id: int) -> dict[str, int]:
        """Итоги кодирования фрагментов интервью по статусам для прогона."""
        rows = self._conn.execute(
            "SELECT fc.status, COUNT(*) FROM fragment_coding fc"
            " JOIN fragment f ON f.id = fc.fragment_id"
            " JOIN utterance u ON u.id = f.utterance_id"
            " WHERE u.interview_id = ? AND fc.run_id = ? GROUP BY fc.status",
            (interview_db_id, run_id),
        ).fetchall()
        return {str(status): int(count) for status, count in rows}

    def fragment_quality_counts(self, interview_db_id: int) -> dict[str, int]:
        """Число фрагментов интервью по оценке качества расшифровки."""
        rows = self._conn.execute(
            "SELECT f.asr_quality, COUNT(*) FROM fragment f"
            " JOIN utterance u ON u.id = f.utterance_id"
            " WHERE u.interview_id = ? GROUP BY f.asr_quality",
            (interview_db_id,),
        ).fetchall()
        return {str(quality): int(count) for quality, count in rows}

    def fragments_total(self, interview_db_id: int) -> int:
        """Всего фрагментов интервью."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM fragment f JOIN utterance u ON u.id = f.utterance_id WHERE u.interview_id = ?",
            (interview_db_id,),
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def answered_question_ids(self, interview_db_id: int) -> set[int]:
        """Вопросы, на которые есть хотя бы одна реплика респондента."""
        rows = self._conn.execute(
            "SELECT DISTINCT question_id FROM utterance"
            " WHERE interview_id = ? AND speaker_role = 'RESPONDENT' AND question_id IS NOT NULL",
            (interview_db_id,),
        ).fetchall()
        return {int(row[0]) for row in rows}

    def l2_version_of_fragments(self, interview_db_id: int) -> int | None:
        """Версия L2 по записанным фрагментам — запасной источник, если её нет в прогоне."""
        row = self._conn.execute(
            "SELECT MAX(t.version) FROM fragment f"
            " JOIN utterance u ON u.id = f.utterance_id"
            " JOIN text_unit t ON t.id = f.text_unit_id"
            " WHERE u.interview_id = ?",
            (interview_db_id,),
        ).fetchone()
        return None if row is None or row[0] is None else int(row[0])
