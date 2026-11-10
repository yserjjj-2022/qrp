"""Тесты очереди ручной проверки и повторного прогона."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from openpyxl import load_workbook

from qa.pipeline.coding import code_interview
from qa.pipeline.coding_review import COLUMNS, export_coding_review, import_coding_review
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


class TestReviewExportColumns:
    """Выгрузка содержит контекст интервьюера, определение кода и пустые колонки решения."""

    def test_new_columns_filled(self, seeded: Seeded) -> None:
        conn = seeded.conn
        texts = fragment_texts(conn)
        provider = ScriptedProvider([answer_for_fragments(texts, code="R1"), answer_for_fragments(texts, code="R1")])
        result = code_interview(conn, "R01", _settings(), provider, code_version="test")
        assert result.run_id is not None
        output = Path.cwd() / "review_columns.xlsx"
        export_coding_review(conn, "R01", result.run_id, output, sample_share=1.0)

        workbook = load_workbook(output)
        sheet = workbook["Проверка_кодов"]
        rows = list(sheet.iter_rows(values_only=True))
        headers = rows[0]
        assert "реплика_интервьюера_перед_ответом" in headers
        assert "определение_кода" in headers
        assert "решение" in headers
        assert "комментарий" in headers
        first = rows[1]
        context = first[headers.index("реплика_интервьюера_перед_ответом")]
        assert "Что может пойти не так" in str(context)
        definition = first[headers.index("определение_кода")]
        assert "Ошибка модели" in str(definition) and "Модель ошибается" in str(definition)
        assert first[headers.index("решение")] is None
        assert first[headers.index("комментарий")] is None
        output.unlink()


class TestReviewImport:
    """Импорт заполненного файла пишет решения в review и считает метрику."""

    def _fill(self, path: Path, decisions: list[str | None]) -> None:
        workbook = load_workbook(path)
        sheet = workbook["Проверка_кодов"]
        headers = [cell.value for cell in sheet[1]]
        decision_index = headers.index("решение") + 1
        comment_index = headers.index("комментарий") + 1
        for offset, decision in enumerate(decisions, start=2):
            sheet.cell(row=offset, column=decision_index, value=decision)
            sheet.cell(row=offset, column=comment_index, value="проверено")
        workbook.save(path)

    def test_import_writes_decisions_and_share(self, seeded: Seeded) -> None:
        conn = seeded.conn
        texts = fragment_texts(conn)
        provider = ScriptedProvider([answer_for_fragments(texts, code="R1"), answer_for_fragments(texts, code="R1")])
        result = code_interview(conn, "R01", _settings(), provider, code_version="test")
        assert result.run_id is not None
        output = Path.cwd() / "review_import.xlsx"
        export_coding_review(conn, "R01", result.run_id, output, sample_share=1.0)
        self._fill(output, ["принято", "отклонено"])

        imported = import_coding_review(conn, output)

        assert imported.approved == 1
        assert imported.edited == 0
        assert imported.rejected == 1
        assert imported.skipped == 0
        assert imported.approved_share == 0.5
        decisions = [row[0] for row in conn.execute("SELECT decision FROM review ORDER BY id").fetchall()]
        assert decisions == ["approved", "rejected"]
        statuses = {row[0] for row in conn.execute("SELECT status FROM annotation").fetchall()}
        assert statuses == {"accepted", "rejected"}
        output.unlink()

    def test_empty_decision_is_skipped(self, seeded: Seeded) -> None:
        conn = seeded.conn
        texts = fragment_texts(conn)
        provider = ScriptedProvider([answer_for_fragments(texts, code="R1"), answer_for_fragments(texts, code="R1")])
        result = code_interview(conn, "R01", _settings(), provider, code_version="test")
        assert result.run_id is not None
        output = Path.cwd() / "review_empty.xlsx"
        export_coding_review(conn, "R01", result.run_id, output, sample_share=1.0)

        imported = import_coding_review(conn, output)

        assert imported.skipped == 2
        assert imported.decided == 0
        assert imported.approved_share is None
        assert conn.execute("SELECT COUNT(*) FROM review").fetchone()[0] == 0
        output.unlink()

    def test_unknown_decision_raises(self, seeded: Seeded) -> None:
        conn = seeded.conn
        texts = fragment_texts(conn)
        provider = ScriptedProvider([answer_for_fragments(texts, code="R1"), answer_for_fragments(texts, code="R1")])
        result = code_interview(conn, "R01", _settings(), provider, code_version="test")
        assert result.run_id is not None
        output = Path.cwd() / "review_bad.xlsx"
        export_coding_review(conn, "R01", result.run_id, output, sample_share=1.0)
        self._fill(output, ["может быть", None])

        with pytest.raises(ValueError, match="Неизвестное решение"):
            import_coding_review(conn, output)
        output.unlink()

    def test_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            import_coding_review(sqlite3.connect(":memory:"), Path.cwd() / "no_such_review.xlsx")


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
