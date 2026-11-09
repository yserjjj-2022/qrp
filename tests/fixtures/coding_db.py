"""Синтетическая БД для тестов кодирования (M2).

Никаких реальных текстов: только короткие нейтральные фразы. Модель
подменяется заглушкой, сети нет.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qa.db.connection import get_connection
from qa.db.migrations import migrate
from qa.llm.base import LlmProvider, LlmResult
from qa.llm.errors import LlmError

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations"

CODES = [
    ("R1", "Риски", "Ошибка модели", "Модель ошибается", "неверный ответ", "сбой (R7)", "R1"),
    ("V1", "Проверка", "Сверяю", "Сверяю с источником", "сверка", "—", "—"),
    ("OTH", "Индуктивный", "Другое", "Не подошло ни под один код", "—", "—", "—"),
]

QUESTIONS = [("8", "Что может пойти не так?"), ("14Т-б", "Кто отвечал бы за потери?")]

# (роль, текст, is_backchannel)
TRANSCRIPT = [
    ("INTERVIEWER", "Что может пойти не так, когда ИИ торгует?", 0),
    ("RESPONDENT", "Он может ошибиться и купить не то. Я потом сверяю по счёту.", 0),
    ("INTERVIEWER", "А кто отвечал бы за потери?", 0),
    ("RESPONDENT", "Наверное, я сам и отвечаю.", 0),
]


@dataclass
class Seeded:
    """Подготовленная БД с интервью, кодбуком и фрагментами."""

    conn: sqlite3.Connection
    interview_id: int
    fragment_ids: list[int]


def make_db(tmp_path: Path) -> sqlite3.Connection:
    """Пустая БД со всеми миграциями."""
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    migrate(conn, db_path, _MIGRATIONS_DIR, tmp_path / "backups")
    return conn


def seed(
    conn: sqlite3.Connection,
    *,
    fragments: list[tuple[int, int, int]] | None = None,
    attributes: list[str] | None = None,
    codebook: bool = True,
) -> Seeded:
    """Заполнить интервью, кодбук и фрагменты.

    Args:
        fragments: (id реплики, start, end) в тексте L2; по умолчанию вся
            реплика — один фрагмент.
        attributes: имена атрибутов; по умолчанию только НАДЁЖНОСТЬ_КОДА.
        codebook: создавать ли версию кодбука с кодами и вопросами.
    """
    existing = conn.execute("SELECT id FROM interview WHERE id_external = 'R01'").fetchone()
    if existing is not None:
        return _reseed_fragments(conn, int(existing[0]), fragments)

    cursor = conn.execute(
        "INSERT INTO interview (id_external, respondent_code, \"group\") VALUES ('R01', 'R01', 'Все')"
    )
    interview_id = int(cursor.lastrowid or 0)

    utterance_ids: list[int] = []
    for index, (role, text, flag) in enumerate(TRANSCRIPT, start=1):
        cursor = conn.execute(
            "INSERT INTO turn (interview_id, line_no, start_sec, speaker_role, text) VALUES (?, ?, ?, ?, ?)",
            (interview_id, index, index * 10, role, text),
        )
        turn_id = int(cursor.lastrowid or 0)
        cursor = conn.execute(
            "INSERT INTO utterance (interview_id, speaker_role, text, turn_ids, is_backchannel) VALUES (?, ?, ?, ?, ?)",
            (interview_id, role, text, f"[{turn_id}]", flag),
        )
        utterance_ids.append(int(cursor.lastrowid or 0))

    if codebook:
        cursor = conn.execute("INSERT INTO codebook_version (file_hash) VALUES ('h1')")
        version_id = int(cursor.lastrowid or 0)
        for name, family, title, definition, include, exclude, tiebreak in CODES:
            conn.execute(
                "INSERT INTO code (codebook_version_id, family, name, title, definition, include, exclude,"
                " tiebreak_rule) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (version_id, family, name, title, definition, include, exclude, tiebreak),
            )
        for identifier, wording in QUESTIONS:
            conn.execute(
                'INSERT INTO question (codebook_version_id, identifier, "group", wording) VALUES (?, ?, ?, ?)',
                (version_id, identifier, "Все", wording),
            )
        for name in attributes if attributes is not None else ["НАДЁЖНОСТЬ_КОДА"]:
            conn.execute(
                "INSERT INTO attribute_definition (codebook_version_id, name, kind, description)"
                " VALUES (?, ?, 'оценка', '1–3')",
                (version_id, name),
            )

    fragment_ids: list[int] = []
    question_ids = [int(row[0]) for row in conn.execute("SELECT id FROM question ORDER BY id").fetchall()]
    for index, utterance_id in enumerate(utterance_ids):
        row = conn.execute("SELECT text, speaker_role FROM utterance WHERE id = ?", (utterance_id,)).fetchone()
        text, role = str(row[0]), str(row[1])
        conn.execute(
            "INSERT INTO text_unit (utterance_id, version, text) VALUES (?, 1, ?)",
            (utterance_id, text),
        )
        unit_id = int(conn.execute("SELECT id FROM text_unit WHERE utterance_id = ?", (utterance_id,)).fetchone()[0])
        if role == "INTERVIEWER":
            continue
        # Вопрос привязывается по порядку: первый ответ — к первому вопросу.
        question_id = question_ids[min(index // 2, len(question_ids) - 1)] if question_ids else None
        conn.execute(
            "UPDATE utterance SET question_id = ? WHERE id = ?",
            (question_id, utterance_id),
        )
        if fragments is None:
            spans = [(utterance_id, 0, len(text))]
        else:
            spans = [item for item in fragments if item[0] == utterance_id]
        for target, start, end in spans:
            if target != utterance_id:
                continue
            cursor = conn.execute(
                'INSERT INTO fragment (utterance_id, text_unit_id, "start", "end", text, question_id)'
                " VALUES (?, ?, ?, ?, ?, ?)",
                (utterance_id, unit_id, start, end, text[start:end], question_id),
            )
            fragment_ids.append(int(cursor.lastrowid or 0))
    conn.commit()
    return Seeded(conn=conn, interview_id=interview_id, fragment_ids=fragment_ids)


def _reseed_fragments(
    conn: sqlite3.Connection,
    interview_id: int,
    fragments: list[tuple[int, int, int]] | None,
) -> Seeded:
    """Пересобрать фрагменты уже засеянного интервью с другим делением."""
    if fragments is None:
        msg = "Повторный seed без fragments не поддерживается"
        raise ValueError(msg)
    conn.execute("DELETE FROM annotation")
    conn.execute("DELETE FROM fragment_coding")
    conn.execute("DELETE FROM fragment")
    fragment_ids: list[int] = []
    for utterance_id, start, end in fragments:
        unit_row = conn.execute("SELECT id, text FROM text_unit WHERE utterance_id = ?", (utterance_id,)).fetchone()
        if unit_row is None:
            msg = f"У реплики {utterance_id} нет L2"
            raise ValueError(msg)
        unit_id, text = int(unit_row[0]), str(unit_row[1])
        question_row = conn.execute("SELECT question_id FROM utterance WHERE id = ?", (utterance_id,)).fetchone()
        question_id = None if question_row is None or question_row[0] is None else int(question_row[0])
        cursor = conn.execute(
            'INSERT INTO fragment (utterance_id, text_unit_id, "start", "end", text, question_id)'
            " VALUES (?, ?, ?, ?, ?, ?)",
            (utterance_id, unit_id, start, end, text[start:end], question_id),
        )
        fragment_ids.append(int(cursor.lastrowid or 0))
    conn.commit()
    return Seeded(conn=conn, interview_id=interview_id, fragment_ids=fragment_ids)


def fragment_texts(conn: sqlite3.Connection) -> dict[int, str]:
    """Тексты фрагментов: id -> текст."""
    rows = conn.execute("SELECT id, text FROM fragment ORDER BY id").fetchall()
    return {int(fid): str(text) for fid, text in rows}


class ScriptedProvider(LlmProvider):
    """Заглушка: отдаёт заранее заданные ответы по очереди."""

    def __init__(self, responses: list[dict[str, Any]], *, model: str = "fake-model") -> None:
        self._responses = list(responses)
        self._model = model
        self.calls = 0
        self.prompts: list[str] = []

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        return self.complete_json_detailed(prompt, schema).data

    def complete_json_detailed(self, prompt: str, schema: dict[str, Any]) -> LlmResult:  # noqa: ARG002
        self.calls += 1
        self.prompts.append(prompt)
        if not self._responses:
            msg = "заглушка исчерпала ответы"
            raise LlmError(msg)
        return LlmResult(data=self._responses.pop(0), model_used=self._model, prompt_tokens=10, completion_tokens=5)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Нулевые векторы: эмбеддинги в кодировании не нужны."""
        return [[0.0] * 4 for _ in texts]


def answer_for_fragments(
    texts: dict[int, str],
    *,
    code: str = "R1",
    order: str | None = "spontaneous",
    reliability: int | None = 3,
    interpretation: str | None = None,
    interpretation_only: bool = False,
    quote: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ответ модели на реплику: по коду на каждый фрагмент."""
    items = []
    for fragment_id, text in texts.items():
        payload: dict[str, Any] = {
            "code": code,
            "quote": text if quote is None else quote,
            "rationale": "подходит по определению",
            "order": order,
            "scenario": None,
            "reliability": reliability,
            "reliability_note": "прямо следует из текста",
        }
        payload.update(extra or {})
        items.append(
            {
                "fragment_id": fragment_id,
                "codes": [payload],
                "interpretation": interpretation,
                "interpretation_only": interpretation_only,
            }
        )
    return {"fragments": items}
