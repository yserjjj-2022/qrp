"""Тесты очереди ручной проверки и повторного прогона."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from openpyxl import load_workbook

from qa.pipeline.coding import code_interview
from qa.pipeline.coding_review import COLUMNS, export_coding_review
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


def _settings() -> Settings:
    return Settings.load(_PROJECT_ROOT / "config.yaml")


@pytest.fixture()
def db(tmp_path: Path) -> sqlite3.Connection:
    conn = make_db(tmp_path)
    yield conn
    conn.close()


@pytest.fixture()
def seeded(db: sqlite3.Connection) -> Seeded:
    return seed(db)


def _multi_codes(texts: dict[int, str], fragment_id: int) -> dict[str, Any]:
    """Ответ с OTH, ненадёжным кодом и обычным кодом на одном фрагменте."""
    text = texts[fragment_id]
    base = {
        "quote": text,
        "rationale": "подходит",
        "order": "spontaneous",
        "scenario": None,
        "reliability": 3,
    }
    return {
        "fragments": [
            {
                "fragment_id": fragment_id,
                "codes": [
                    {**base, "code": "R1"},
                    {**base, "code": "OTH", "oth_description": "новый смысл"},
                    {**base, "code": "V1", "reliability": 1, "reliability_note": "держится на догадке"},
                ],
                "interpretation": None,
                "interpretation_only": False,
            }
        ]
    }


class TestReviewQueue:
    """Очередь = все OTH и ненадёжные + выборочная доля остальных."""

    def test_required_cases_included(self, seeded: Seeded) -> None:
        conn = seeded.conn
        texts = fragment_texts(conn)
        first, second = seeded.fragment_ids
        provider = ScriptedProvider([_multi_codes(texts, first), answer_for_fragments({second: texts[second]})])
        result = code_interview(conn, "R01", _settings(), provider, code_version="test")
        assert result.run_id is not None

        output = Path(conn.execute("SELECT 1").fetchone() and Path.cwd()) / "review.xlsx"
        review = export_coding_review(conn, "R01", result.run_id, output, sample_share=0.0)
        assert review.required == 2  # OTH + reliability=1
        assert review.sampled == 0
        assert review.n_rows == 2

        workbook = load_workbook(output)
        sheet = workbook["Проверка_кодов"]
        rows = list(sheet.iter_rows(values_only=True))
        assert rows[0] == tuple(COLUMNS)
        reasons = {row[COLUMNS.index("причина_проверки")] for row in rows[1:]}
        assert reasons == {"oth", "unreliable"}
        output.unlink()

    def test_sample_share(self, seeded: Seeded) -> None:
        conn = seeded.conn
        texts = fragment_texts(conn)
        provider = ScriptedProvider([answer_for_fragments(texts, code="R1"), answer_for_fragments(texts, code="R1")])
        result = code_interview(conn, "R01", _settings(), provider, code_version="test")
        assert result.run_id is not None
        output = Path.cwd() / "review_sample.xlsx"
        review = export_coding_review(conn, "R01", result.run_id, output, sample_share=1.0)
        assert review.required == 0
        assert review.sampled == review.n_rows == 2
        output.unlink()

    def test_pii_fragment_included(self, seeded: Seeded) -> None:
        conn = seeded.conn
        conn.execute("UPDATE turn SET pii_flag = 1")
        conn.commit()
        texts = fragment_texts(conn)
        provider = ScriptedProvider([answer_for_fragments(texts, code="R1"), answer_for_fragments(texts, code="R1")])
        result = code_interview(conn, "R01", _settings(), provider, code_version="test")
        assert result.run_id is not None
        output = Path.cwd() / "review_pii.xlsx"
        review = export_coding_review(conn, "R01", result.run_id, output, sample_share=0.0)
        assert review.required == 2
        output.unlink()

    def test_no_annotations_raises(self, seeded: Seeded) -> None:
        with pytest.raises(ValueError):
            export_coding_review(seeded.conn, "R01", 999, Path.cwd() / "empty.xlsx")


class TestRerun:
    """Повторный прогон не перезаписывает принятое и исправленное."""

    def test_accepted_annotations_survive(self, seeded: Seeded) -> None:
        conn = seeded.conn
        texts = fragment_texts(conn)
        first, second = seeded.fragment_ids
        provider = ScriptedProvider([answer_for_fragments(texts, code="R1"), answer_for_fragments(texts, code="R1")])
        code_interview(conn, "R01", _settings(), provider, code_version="test")

        conn.execute("UPDATE annotation SET status = 'accepted' WHERE fragment_id = ?", (first,))
        conn.execute("UPDATE annotation SET status = 'edited' WHERE fragment_id = ?", (second,))
        conn.commit()
        before = conn.execute("SELECT COUNT(*) FROM annotation").fetchone()[0]
        assert before == 2

        provider2 = ScriptedProvider([answer_for_fragments(texts, code="V1"), answer_for_fragments(texts, code="V1")])
        code_interview(conn, "R01", _settings(), provider2, code_version="test")

        # Принятое и исправленное на месте, новые предложенные аннотации не дублируют их.
        statuses = dict(conn.execute("SELECT fragment_id, status FROM annotation").fetchall())
        assert statuses == {first: "accepted", second: "edited"}
