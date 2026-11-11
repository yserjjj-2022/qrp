"""Синтетическая БД для тестов агрегации покрытия (M4/M5).

Собирает минимальный набор строк, достаточный для `analytics_repo` и
`aggregate`: дерево тем, кодбук с вопросами, интервью с фрагментами и
аннотациями. Тексты синтетические, реальные данные не используются.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from qa.db.connection import get_connection
from qa.db.migrations import migrate

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations"


@dataclass
class CodeSpec:
    """Код кодбука и вопросы, к которым он привязан."""

    name: str
    family: str
    questions: tuple[str, ...] = ()


@dataclass
class QuestionSpec:
    """Вопрос гайда."""

    identifier: str
    group: str


@dataclass
class AnnotationSpec:
    """Принятая аннотация поверх фрагмента."""

    fragment_index: int
    code_name: str
    quote: str
    status: str = "accepted"
    reliability: int | None = 2


@dataclass
class InterviewSpec:
    """Интервью: фрагменты, аннотации и итоги кодирования."""

    code: str = "R01"
    group: str = "Не торгующие"
    fragments: list[str] = field(default_factory=list)
    # Вопрос гайда, к которому привязана реплика: индекс фрагмента -> идентификатор.
    # Реплики без записи считаются репликами вне гайда.
    questions: dict[int, str] = field(default_factory=dict)
    annotations: list[AnnotationSpec] = field(default_factory=list)
    unmarked_fragments: tuple[int, ...] = ()
    doubtful_fragments: tuple[int, ...] = ()
    proposed: int = 0


def build_analytics_db(
    path: Path,
    *,
    themes: list[tuple[str, str, list[str]]],
    codes: list[CodeSpec],
    questions: list[QuestionSpec],
    interview: InterviewSpec,
    codebook_version_id: int = 1,
) -> sqlite3.Connection:
    """Создать БД с деревом тем, кодбуком и одним интервью."""
    db_path = path / "analytics.db"
    conn = get_connection(db_path)
    migrate(conn, db_path, _MIGRATIONS_DIR, path / "backups")

    code_ids: dict[str, int] = {}
    cursor = conn.execute(
        "INSERT INTO codebook_version (file_hash, codebook_hash) VALUES (?, ?)",
        ("test-hash", "test-hash"),
    )
    codebook_version_id = int(cursor.lastrowid or 0)
    for code in codes:
        cursor = conn.execute(
            "INSERT INTO code (codebook_version_id, family, name) VALUES (?, ?, ?)",
            (codebook_version_id, code.family, code.name),
        )
        code_ids[code.name] = int(cursor.lastrowid or 0)

    question_ids: dict[str, int] = {}
    for question in questions:
        cursor = conn.execute(
            'INSERT INTO question (codebook_version_id, identifier, "group", wording) VALUES (?, ?, ?, ?)',
            (codebook_version_id, question.identifier, question.group, f"Вопрос {question.identifier}"),
        )
        question_ids[question.identifier] = int(cursor.lastrowid or 0)

    for code in codes:
        for identifier in code.questions:
            conn.execute(
                "INSERT INTO question_code (question_id, code_id) VALUES (?, ?)",
                (question_ids[identifier], code_ids[code.name]),
            )

    theme_ids: dict[str, int] = {}
    for node, theme, theme_codes in themes:
        if node not in theme_ids:
            cursor = conn.execute("INSERT INTO theme (name, parent_id, level) VALUES (?, NULL, 0)", (node,))
            theme_ids[node] = int(cursor.lastrowid or 0)
        cursor = conn.execute(
            "INSERT INTO theme (name, parent_id, level) VALUES (?, ?, 1)",
            (theme, theme_ids[node]),
        )
        theme_id = int(cursor.lastrowid or 0)
        for code_name in theme_codes:
            conn.execute(
                "INSERT INTO theme_code (theme_id, code_id) VALUES (?, ?)",
                (theme_id, code_ids[code_name]),
            )

    cursor = conn.execute(
        'INSERT INTO interview (id_external, "group", transcript_source, interview_date) VALUES (?, ?, ?, ?)',
        (interview.code, interview.group, "asr", "2026-11-02"),
    )
    interview_id = int(cursor.lastrowid or 0)

    cursor = conn.execute(
        "INSERT INTO run (model, prompt_hash, codebook_version_id, code_version, params_json) VALUES (?, ?, ?, ?, ?)",
        (
            "fake-model",
            "hash",
            codebook_version_id,
            "0.0.0",
            '{"l2_version": 1, "fragments": ' + str(len(interview.fragments)) + "}",
        ),
    )
    run_id = int(cursor.lastrowid or 0)

    for index, text in enumerate(interview.fragments):
        cursor = conn.execute(
            "INSERT INTO utterance (interview_id, speaker_role, text, turn_ids, question_id)"
            " VALUES (?, 'RESPONDENT', ?, '[]', ?)",
            (interview_id, text, question_ids.get(interview.questions.get(index, ""))),
        )
        utterance_id = int(cursor.lastrowid or 0)
        cursor = conn.execute(
            "INSERT INTO text_unit (utterance_id, version, text, status) VALUES (?, 1, ?, 'proposed')",
            (utterance_id, text),
        )
        text_unit_id = int(cursor.lastrowid or 0)
        quality = "doubtful" if index in interview.doubtful_fragments else "ok"
        cursor = conn.execute(
            'INSERT INTO fragment (utterance_id, text_unit_id, "start", "end", text, asr_quality)'
            " VALUES (?, ?, 0, ?, ?, ?)",
            (utterance_id, text_unit_id, len(text), text, quality),
        )
        fragment_id = int(cursor.lastrowid or 0)
        status = "unmarked" if index in interview.unmarked_fragments else "coded"
        conn.execute(
            "INSERT INTO fragment_coding (fragment_id, run_id, status) VALUES (?, ?, ?)",
            (fragment_id, run_id, status),
        )

        for annotation in interview.annotations:
            if annotation.fragment_index != index:
                continue
            position = text.find(annotation.quote)
            conn.execute(
                "INSERT INTO annotation (fragment_id, code_id, quote, quote_start, quote_end, status, reliability)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    fragment_id,
                    code_ids[annotation.code_name],
                    annotation.quote,
                    position if position >= 0 else None,
                    position + len(annotation.quote) if position >= 0 else None,
                    annotation.status,
                    annotation.reliability,
                ),
            )

    for _ in range(interview.proposed):
        conn.execute(
            "INSERT INTO annotation (fragment_id, code_id, quote, status) VALUES (1, ?, 'x', 'proposed')",
            (next(iter(code_ids.values())),),
        )

    conn.commit()
    return conn
