"""Синтетическая БД для тестов M8 (оркестрация, термины, цитаты, отчёт).

Никаких реальных текстов: короткие нейтральные фразы. Метка говорящего
заполнена вымышленным именем — тесты проверяют, что она не попадает в
выгрузки (CONSTITUTION §5).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from qa.db.connection import get_connection
from qa.db.migrations import migrate

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations"

# Метка говорящего в тестовых данных: настоящие имена в выгрузки не идут.
FAKE_SPEAKER_LABEL = "Тестовое Имя"

CODES: list[tuple[str, str]] = [
    ("C1", "Контроль"),
    ("C2", "Контроль"),
    ("C4", "Контроль"),
    ("B3", "Основание"),
    ("OTH", "Индуктивный"),
]

QUESTIONS: list[tuple[str, str]] = [
    ("5", "Что ИИ мог бы делать сам?"),
    ("17", "Что бы вы посоветовали новичку?"),
    ("18", "При каких условиях вы бы согласились?"),
    ("20", "Что ещё доверили бы ИИ?"),
    ("22", "Что нужно, чтобы попробовать?"),
]

THEMES: list[tuple[str, str, list[str]]] = [
    ("Ответственность", "Контроль", ["C1", "C2", "C4"]),
    ("Ответственность", "Основание", ["B3"]),
    ("Индуктивные", "Индуктивный", ["OTH"]),
]


@dataclass
class UtteranceSpec:
    """Реплика интервью: L1, L2, тайм-код и вопрос гайда."""

    role: str
    text_l1: str
    text_l2: str
    timecode: int
    question: str | None = None
    backchannel: bool = False
    quality: str = "ok"


@dataclass
class AnnotationSpec:
    """Аннотация поверх реплики; `quote` — подстрока её L2."""

    utterance_index: int
    code: str
    quote: str
    status: str = "accepted"
    reliability: int | None = 2
    interpretation_only: bool = False


@dataclass
class TermSpec:
    """Кандидат `new_term`: форма и реплика-контекст."""

    surface_form: str
    utterance_index: int | None = None


@dataclass
class M8Spec:
    """Состав синтетического интервью."""

    utterances: list[UtteranceSpec] = field(default_factory=list)
    annotations: list[AnnotationSpec] = field(default_factory=list)
    terms: list[TermSpec] = field(default_factory=list)
    lexicon: tuple[str, ...] = ("MCP",)
    run_started_at: str = "2020-01-01 00:00:00"


def build_m8_db(path: Path, spec: M8Spec) -> sqlite3.Connection:
    """Создать БД с кодбуком, деревом тем, интервью и аннотациями."""
    db_path = path / "m8.db"
    conn = get_connection(db_path)
    migrate(conn, db_path, _MIGRATIONS_DIR, path / "backups")

    cursor = conn.execute("INSERT INTO codebook_version (file_hash, codebook_hash) VALUES ('m8-hash', 'm8-hash')")
    version_id = int(cursor.lastrowid or 0)
    code_ids: dict[str, int] = {}
    for name, family in CODES:
        cursor = conn.execute(
            "INSERT INTO code (codebook_version_id, family, name) VALUES (?, ?, ?)",
            (version_id, family, name),
        )
        code_ids[name] = int(cursor.lastrowid or 0)

    question_ids: dict[str, int] = {}
    for identifier, wording in QUESTIONS:
        cursor = conn.execute(
            'INSERT INTO question (codebook_version_id, identifier, "group", wording) VALUES (?, ?, ?, ?)',
            (version_id, identifier, "Все", wording),
        )
        question_ids[identifier] = int(cursor.lastrowid or 0)

    node_ids: dict[str, int] = {}
    for node, theme, theme_codes in THEMES:
        if node not in node_ids:
            cursor = conn.execute("INSERT INTO theme (name, parent_id, level) VALUES (?, NULL, 0)", (node,))
            node_ids[node] = int(cursor.lastrowid or 0)
        cursor = conn.execute(
            "INSERT INTO theme (name, parent_id, level) VALUES (?, ?, 1)",
            (theme, node_ids[node]),
        )
        theme_id = int(cursor.lastrowid or 0)
        for code_name in theme_codes:
            conn.execute(
                "INSERT INTO theme_code (theme_id, code_id) VALUES (?, ?)",
                (theme_id, code_ids[code_name]),
            )

    for canonical in spec.lexicon:
        conn.execute(
            "INSERT INTO lexicon (canonical, variants_json, status, source) VALUES (?, '[]', 'approved', 'workbook')",
            (canonical,),
        )

    cursor = conn.execute(
        'INSERT INTO interview (id_external, "group", transcript_source, interview_date)'
        " VALUES ('R01', 'Не торгующие', 'asr', '2026-11-02')"
    )
    interview_id = int(cursor.lastrowid or 0)

    conn.execute(
        "INSERT INTO run (started_at, model, prompt_hash, codebook_version_id, code_version, params_json)"
        " VALUES (?, 'fake-model', 'hash', ?, '0.0.0', ?)",
        (spec.run_started_at, version_id, '{"l2_version": 1, "fragments": ' + str(len(spec.utterances)) + "}"),
    )
    run_id = int(conn.execute("SELECT MAX(id) FROM run").fetchone()[0])

    utterance_ids: list[int] = []
    fragment_ids: list[int] = []
    for index, item in enumerate(spec.utterances):
        cursor = conn.execute(
            "INSERT INTO turn (interview_id, line_no, start_sec, speaker_role, text, speaker_label)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (interview_id, index + 1, item.timecode, item.role, item.text_l1, FAKE_SPEAKER_LABEL),
        )
        turn_id = int(cursor.lastrowid or 0)
        cursor = conn.execute(
            "INSERT INTO utterance (interview_id, speaker_role, text, turn_ids, is_backchannel, question_id)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                interview_id,
                item.role,
                item.text_l1,
                f"[{turn_id}]",
                1 if item.backchannel else 0,
                question_ids.get(item.question or ""),
            ),
        )
        utterance_id = int(cursor.lastrowid or 0)
        utterance_ids.append(utterance_id)
        cursor = conn.execute(
            "INSERT INTO text_unit (utterance_id, version, text, status, asr_quality) VALUES (?, 1, ?, 'proposed', ?)",
            (utterance_id, item.text_l2, item.quality),
        )
        unit_id = int(cursor.lastrowid or 0)
        cursor = conn.execute(
            'INSERT INTO fragment (utterance_id, text_unit_id, "start", "end", text, question_id, asr_quality)'
            " VALUES (?, ?, 0, ?, ?, ?, ?)",
            (
                utterance_id,
                unit_id,
                len(item.text_l2),
                item.text_l2,
                question_ids.get(item.question or ""),
                item.quality,
            ),
        )
        fragment_id = int(cursor.lastrowid or 0)
        fragment_ids.append(fragment_id)
        conn.execute(
            "INSERT INTO fragment_coding (fragment_id, run_id, status) VALUES (?, ?, 'coded')",
            (fragment_id, run_id),
        )

    for annotation in spec.annotations:
        text = spec.utterances[annotation.utterance_index].text_l2
        position = text.find(annotation.quote)
        conn.execute(
            "INSERT INTO annotation (fragment_id, code_id, run_id, quote, quote_start, quote_end,"
            " status, reliability, interpretation_only)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                fragment_ids[annotation.utterance_index],
                code_ids[annotation.code],
                run_id,
                annotation.quote,
                position if position >= 0 else None,
                position + len(annotation.quote) if position >= 0 else None,
                annotation.status,
                annotation.reliability,
                1 if annotation.interpretation_only else 0,
            ),
        )

    for term in spec.terms:
        # UNIQUE(surface_form, context_utterance_id): одна форма из двух разных
        # реплик — это две строки `new_term`, как в живой базе.
        conn.execute(
            "INSERT OR IGNORE INTO new_term (interview_id, surface_form, context_utterance_id, status)"
            " VALUES (?, ?, ?, 'proposed')",
            (
                interview_id,
                term.surface_form,
                None if term.utterance_index is None else utterance_ids[term.utterance_index],
            ),
        )

    conn.commit()
    return conn


__all__ = [
    "CODES",
    "FAKE_SPEAKER_LABEL",
    "QUESTIONS",
    "THEMES",
    "AnnotationSpec",
    "M8Spec",
    "TermSpec",
    "UtteranceSpec",
    "build_m8_db",
]
