"""Тесты признака ПОРЯДОК: категория, а не семья кодов; обрезка контекста."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from qa.pipeline.coding import code_interview
from qa.settings import Settings
from tests.fixtures.coding_db import (
    ScriptedProvider,
    Seeded,
    answer_for_fragments,
    fragment_texts,
    make_db,
    seed,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _settings(**overrides: object) -> Settings:
    settings = Settings.load(_PROJECT_ROOT / "config.yaml")
    if overrides:
        data = settings.model_dump()
        data.setdefault("m2", {}).update(overrides)
        return Settings(**data)
    return settings


@pytest.fixture()
def db(tmp_path: Path) -> sqlite3.Connection:
    conn = make_db(tmp_path)
    yield conn
    conn.close()


@pytest.fixture()
def seeded(db: sqlite3.Connection) -> Seeded:
    return seed(db)


def _orders(conn: sqlite3.Connection) -> list[str | None]:
    return [row[0] for row in conn.execute("SELECT order_kind FROM annotation ORDER BY id").fetchall()]


class TestOrderPrompt:
    """Интервьюер назвал категорию раньше — модель видит это в промпте."""

    def test_prior_interviewer_turn_in_prompt(self, seeded: Seeded) -> None:
        texts = fragment_texts(seeded.conn)
        provider = ScriptedProvider([answer_for_fragments(texts), answer_for_fragments(texts)])
        code_interview(seeded.conn, "R01", _settings(), provider, code_version="test")

        # Первый ответ идёт после вопроса интервьюера о рисках.
        assert "- Что может пойти не так, когда ИИ торгует?" in provider.prompts[0]
        # Второй ответ — после реплики про ответственность.
        assert "А кто отвечал бы за потери?" in provider.prompts[1]

    def test_prompted_is_stored(self, seeded: Seeded) -> None:
        texts = fragment_texts(seeded.conn)
        provider = ScriptedProvider(
            [
                answer_for_fragments(texts, order="prompted"),
                answer_for_fragments(texts, order="spontaneous"),
            ]
        )
        code_interview(seeded.conn, "R01", _settings(), provider, code_version="test")
        assert _orders(seeded.conn) == ["prompted", "spontaneous"]

    def test_unknown_order_is_rejected(self, seeded: Seeded) -> None:
        texts = fragment_texts(seeded.conn)
        provider = ScriptedProvider([answer_for_fragments(texts, order="maybe"), answer_for_fragments(texts)])
        result = code_interview(seeded.conn, "R01", _settings(), provider, code_version="test")
        assert result.stats.rejections.get("не указан признак ПОРЯДОК", 0) >= 1

    def test_category_is_content_not_family(self, seeded: Seeded) -> None:
        """Одна реплика: код «риск» подсказан, код «проверка» — нет.

        Слово интервьюера «риски» не делает подсказкой все коды семьи: признак
        ставится на каждый код отдельно, по его собственному содержанию.
        """
        conn = seeded.conn
        first, second = seeded.fragment_ids
        text = fragment_texts(conn)[first]
        response = {
            "fragments": [
                {
                    "fragment_id": first,
                    "codes": [
                        {
                            "code": "R1",
                            "quote": text,
                            "rationale": "интервьюер спрашивал про риски",
                            "order": "prompted",
                            "scenario": None,
                            "reliability": 3,
                        },
                        {
                            "code": "V1",
                            "quote": text,
                            "rationale": "про проверку интервьюер не спрашивал",
                            "order": "spontaneous",
                            "scenario": None,
                            "reliability": 3,
                        },
                    ],
                    "interpretation": None,
                    "interpretation_only": False,
                }
            ]
        }
        provider = ScriptedProvider([response, answer_for_fragments({second: fragment_texts(conn)[second]})])
        code_interview(conn, "R01", _settings(), provider, code_version="test")

        rows = conn.execute(
            "SELECT c.name, a.order_kind FROM annotation a JOIN code c ON a.code_id = c.id ORDER BY a.id"
        ).fetchall()
        assert (rows[0][0], rows[0][1]) == ("R1", "prompted")
        assert (rows[1][0], rows[1][1]) == ("V1", "spontaneous")


class TestTruncatedContext:
    """Обрезанная история интервьюера понижает надёжность до 2."""

    def test_truncation_caps_reliability(self, seeded: Seeded) -> None:
        texts = fragment_texts(seeded.conn)
        provider = ScriptedProvider(
            [answer_for_fragments(texts, reliability=3), answer_for_fragments(texts, reliability=3)]
        )
        settings = _settings(prior_turns_limit=1)
        result = code_interview(seeded.conn, "R01", settings, provider, code_version="test")

        assert result.stats.annotations >= 1
        reliabilities = [
            row[0] for row in seeded.conn.execute("SELECT reliability FROM annotation ORDER BY id").fetchall()
        ]
        # У первой реплики одна реплика интервьюера (лимит 1 — не обрезано),
        # у второй — две, история обрезана, поэтому надёжность не выше 2.
        assert reliabilities == [3, 2]
