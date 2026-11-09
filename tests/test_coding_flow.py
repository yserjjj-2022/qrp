"""Сквозной тест M1a -> M1b -> M1c -> M2 на синтетическом интервью.

Проверяется стык этапов: транскрипт загружается, L2 пишется, фрагменты
привязываются к вопросу гайда, кодирование делает один вызов на реплику и
оставляет проверяемые аннотации. Все тексты синтетические, модель — заглушка.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from qa.db.connection import get_connection
from qa.db.migrations import migrate
from qa.llm.base import LlmProvider, LlmResult
from qa.llm.errors import LlmError
from qa.pipeline.coding import code_interview, plan_dry_run
from qa.pipeline.init_interview import build_plan, plan_to_meta
from qa.pipeline.load_codebook import load_codebook
from qa.pipeline.load_transcript import load_transcript
from qa.pipeline.normalize import normalize_interview
from qa.pipeline.structure import structure_interview
from qa.settings import Settings
from tests.fixtures.workbook_factory import build_workbook

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations"

_TRANSCRIPT = "\n".join(
    [
        "Интервью R01, 2026-11-05",
        "00:00:01\tИнтервьюер\tКак давно вы инвестируете?",
        "00:00:10\tРеспондент\tГода три уже. Начинал с облигаций.",
    ]
)

_ROLES = {"Интервьюер": "INTERVIEWER", "Респондент": "RESPONDENT"}


class FakeProvider(LlmProvider):
    """Заглушка: отдаёт заданные ответы по очереди, исчерпание — LlmError."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
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
        return LlmResult(
            data=self._responses.pop(0),
            model_used="fake-model",
            prompt_tokens=10,
            completion_tokens=5,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Нулевые векторы: эмбеддинги на этих этапах не нужны."""
        return [[0.0] * 4 for _ in texts]


@pytest.fixture()
def settings() -> Settings:
    return Settings.load(_PROJECT_ROOT / "config.yaml")


@pytest.fixture()
def db(tmp_path: Path) -> sqlite3.Connection:
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    migrate(conn, db_path, _MIGRATIONS_DIR, tmp_path / "backups")
    yield conn
    conn.close()


def _normalize_answer(text: str) -> dict[str, Any]:
    return {
        "normalized_text": text,
        "changes": [],
        "uncertain_spans": [],
        "new_terms": [],
        "asr_quality": "ok",
    }


def test_pipeline_end_to_end(db: sqlite3.Connection, tmp_path: Path, settings: Settings) -> None:
    """M1a -> M1b -> M1c -> M2: один вызов на реплику, аннотации и run записаны."""
    # M1a: книга и транскрипт.
    book = build_workbook(tmp_path / "book.xlsx")
    load_codebook(db, book, settings.workbook)
    transcript = tmp_path / "R01.txt"
    transcript.write_text(_TRANSCRIPT, encoding="utf-8")
    plan = build_plan(db, transcript, "не торгующие", settings.groups)
    loaded = load_transcript(db, transcript, plan_to_meta(plan), {}, settings.pipeline, groups=settings.groups)
    assert loaded.n_turns == 2
    assert loaded.skipped is False

    # M1b: L2 реплики респондента пишется, интервьюер модель не вызывает.
    normalize_provider = FakeProvider([_normalize_answer("Года три уже. Начинал с облигаций.")])
    normalized = normalize_interview(db, "R01", settings, normalize_provider)
    assert normalized.skipped is False
    assert normalize_provider.calls == 1

    # M1c: привязка к вопросу гайда и деление на два фрагмента.
    structure_provider = FakeProvider(
        [
            {"blocks": [{"block": 1, "question": "1", "is_probe": False}]},
            {"fragments": ["Года три уже.", "Начинал с облигаций."]},
        ]
    )
    structured = structure_interview(db, "R01", settings, structure_provider, code_version="test")
    assert structured.stats.fragments == 2
    rows = db.execute(
        "SELECT f.id, f.text, f.question_id FROM fragment f ORDER BY f.id",
    ).fetchall()
    assert [str(row[1]).strip() for row in rows] == ["Года три уже.", "Начинал с облигаций."]
    assert all(row[2] is not None for row in rows)

    # M2 dry-run: одна реплика, два фрагмента, без сети.
    report = plan_dry_run(db, "R01", settings)
    assert report.utterances == 1
    assert report.fragments == 2
    assert report.model_calls == 1

    # M2: ответ модели по реплике — по коду на фрагмент.
    fragments = {int(row[0]): str(row[1]) for row in rows}
    coding_provider = FakeProvider(
        [
            {
                "fragments": [
                    {
                        "fragment_id": fragment_id,
                        "codes": [
                            {
                                "code": "R1",
                                "quote": text,
                                "rationale": "назван риск",
                                "order": "spontaneous",
                                "scenario": None,
                                "reliability": 3,
                                "reliability_note": "категория названа прямо",
                            }
                        ],
                        "interpretation": None,
                        "interpretation_only": False,
                    }
                    for fragment_id, text in fragments.items()
                ]
            }
        ]
    )
    coded = code_interview(db, "R01", settings, coding_provider, code_version="test")

    assert coding_provider.calls == 1, "ожидался один вызов на реплику"
    assert coded.stats.fragments == 2
    assert coded.stats.coded == 2
    assert coded.stats.annotations == 2

    statuses = [str(row[0]) for row in db.execute("SELECT status FROM fragment_coding ORDER BY id")]
    assert statuses == ["coded", "coded"]

    # Цитата дословна: она равна срезу L2 по сохранённым позициям.
    bad = db.execute(
        "SELECT a.id FROM annotation a"
        " JOIN fragment f ON a.fragment_id = f.id"
        " JOIN text_unit t ON f.text_unit_id = t.id"
        " WHERE substr(t.text, a.quote_start + 1, a.quote_end - a.quote_start) <> a.quote"
    ).fetchall()
    assert bad == []

    # run записан с моделью, хешем промпта, версией кодбука и версией кода.
    run = db.execute(
        "SELECT model, prompt_hash, codebook_version_id, code_version FROM run WHERE id = ?",
        (coded.run_id,),
    ).fetchone()
    assert run is not None
    assert run[0] == "fake-model"
    assert run[1]
    assert run[2] is not None
    assert run[3] == "test"
