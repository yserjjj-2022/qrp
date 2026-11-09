"""Тесты кодирования M2: промпт на реплику, схема, проверки, запись.

Все тексты синтетические, модель подменяется заглушкой, сети нет.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from qa.pipeline.coding import (
    RELIABILITY_ATTRIBUTE,
    CodingError,
    code_interview,
    format_report,
    load_codes,
    plan_dry_run,
    require_reliability_attribute,
)
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


def _annotations(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute("SELECT * FROM annotation ORDER BY id").fetchall())
    finally:
        conn.row_factory = None


class TestMigration0007:
    """Схема кодирования после миграции 0007."""

    def test_columns(self, db: sqlite3.Connection) -> None:
        annotation = {row[1] for row in db.execute("PRAGMA table_info(annotation)").fetchall()}
        assert {
            "coding_id",
            "interpretation",
            "interpretation_only",
            "order_kind",
            "scenario",
            "reliability",
            "reliability_note",
            "oth_description",
        } <= annotation
        coding = {row[1] for row in db.execute("PRAGMA table_info(fragment_coding)").fetchall()}
        assert {"fragment_id", "run_id", "status", "note"} <= coding

    def test_reliability_range_enforced(self, seeded: Seeded) -> None:
        conn = seeded.conn
        row = conn.execute("SELECT id, code_id FROM annotation LIMIT 1").fetchone()
        assert row is None
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO annotation (fragment_id, code_id, quote, quote_start, quote_end, reliability)"
                " VALUES (?, (SELECT id FROM code LIMIT 1), 'x', 0, 1, 9)",
                (seeded.fragment_ids[0],),
            )


class TestPrompt:
    """Промпт собирается на реплику, а не на фрагмент."""

    def test_one_prompt_for_utterance(self, seeded: Seeded) -> None:
        conn = seeded.conn
        text = "Он может ошибиться и купить не то. Я потом сверяю по счёту."
        cut = text.index(" Я")
        seeded = seed(conn, fragments=[(2, 0, cut), (2, cut + 1, len(text))])
        _version_id, codes = load_codes(conn)
        texts = fragment_texts(conn)
        provider = ScriptedProvider([answer_for_fragments(texts)])

        result = code_interview(conn, "R01", _settings(), provider, code_version="test")

        assert provider.calls == 1
        assert result.stats.model_calls == 1
        assert len(result.stats.by_quality) >= 1
        # Оба фрагмента реплики в одном промпте.
        assert str(seeded.fragment_ids[0]) in provider.prompts[0]
        assert str(seeded.fragment_ids[1]) in provider.prompts[0]
        assert codes

    def test_prompt_has_question_and_prior_turns(self, seeded: Seeded) -> None:
        _version_id, codes = load_codes(seeded.conn)
        texts = fragment_texts(seeded.conn)
        provider = ScriptedProvider([answer_for_fragments(texts), answer_for_fragments(texts)])
        code_interview(seeded.conn, "R01", _settings(), provider, code_version="test")
        first = provider.prompts[0]
        assert "Что может пойти не так?" in first
        # Перед первым ответом интервьюер задал вопрос — он виден модели.
        assert "- Что может пойти не так, когда ИИ торгует?" in first
        assert codes


class TestQuoteValidation:
    """Цитата обязана быть срезом L2 внутри своего фрагмента."""

    def test_quote_outside_fragment_is_rejected(self, seeded: Seeded) -> None:
        conn = seeded.conn
        texts = fragment_texts(conn)
        first, second = seeded.fragment_ids
        # Цитата из второй реплики для первого фрагмента: в его срезе её нет.
        bad = {
            "fragments": [
                {
                    "fragment_id": first,
                    "codes": [
                        {
                            "code": "R1",
                            "quote": texts[second],
                            "rationale": "подходит",
                            "order": "spontaneous",
                            "scenario": None,
                            "reliability": 3,
                        }
                    ],
                    "interpretation": None,
                    "interpretation_only": False,
                }
            ]
        }
        good = answer_for_fragments({second: texts[second]})
        provider = ScriptedProvider([bad, good])
        result = code_interview(conn, "R01", _settings(), provider, code_version="test")

        assert result.stats.annotations == 1
        notes = [row[0] for row in conn.execute("SELECT note FROM fragment_coding").fetchall()]
        assert any(note and "цитата" in note for note in notes)
        assert result.stats.rejections.get("цитата не найдена внутри фрагмента", 0) >= 1

    def test_quote_positions_are_unit_offsets(self, seeded: Seeded) -> None:
        conn = seeded.conn
        texts = fragment_texts(conn)
        provider = ScriptedProvider([answer_for_fragments(texts), answer_for_fragments(texts)])
        code_interview(conn, "R01", _settings(), provider, code_version="test")

        rows = conn.execute(
            "SELECT a.quote, a.quote_start, a.quote_end, t.text FROM annotation a"
            " JOIN fragment f ON a.fragment_id = f.id JOIN text_unit t ON f.text_unit_id = t.id"
        ).fetchall()
        assert rows
        for quote, start, end, unit_text in rows:
            assert str(unit_text)[int(start) : int(end)] == str(quote)


class TestCodeValidation:
    """Код должен существовать в текущей версии кодбука."""

    def test_unknown_code_is_rejected(self, seeded: Seeded) -> None:
        conn = seeded.conn
        texts = fragment_texts(conn)
        bad = answer_for_fragments(texts, code="Z9")
        good = answer_for_fragments(texts, code="R1")
        provider = ScriptedProvider([bad, good])
        result = code_interview(conn, "R01", _settings(), provider, code_version="test")

        assert result.stats.rejections.get("код вне кодбука", 0) >= 1
        rows = conn.execute("SELECT c.name FROM annotation a JOIN code c ON a.code_id = c.id")
        assert all(row[0] == "R1" for row in rows)


class TestOthDescription:
    """OTH без описания не записывается."""

    def test_oth_without_description(self, seeded: Seeded) -> None:
        conn = seeded.conn
        texts = fragment_texts(conn)
        provider = ScriptedProvider([answer_for_fragments(texts, code="OTH"), answer_for_fragments(texts, code="R1")])
        result = code_interview(conn, "R01", _settings(), provider, code_version="test")

        assert result.stats.oth == 0
        assert result.stats.rejections.get("у кода OTH нет описания", 0) >= 1

    def test_oth_with_description(self, seeded: Seeded) -> None:
        conn = seeded.conn
        texts = fragment_texts(conn)
        oth = answer_for_fragments(texts, code="OTH", extra={"oth_description": "значимый новый смысл"})
        provider = ScriptedProvider([oth, answer_for_fragments(texts, code="R1")])
        result = code_interview(conn, "R01", _settings(), provider, code_version="test")

        assert result.stats.oth >= 1
        row = conn.execute("SELECT oth_description FROM annotation WHERE oth_description IS NOT NULL").fetchone()
        assert row is not None and row[0] == "значимый новый смысл"


class TestRequiredAttribute:
    """Без атрибута НАДЁЖНОСТЬ_КОДА кодирование не начинается."""

    def test_missing_attribute_raises(self, db: sqlite3.Connection) -> None:
        seed(db, attributes=["ГРУППА", "ПОНИМАНИЕ"])
        with pytest.raises(CodingError) as error:
            require_reliability_attribute(db)
        assert RELIABILITY_ATTRIBUTE in str(error.value)

    def test_present_attribute_passes(self, seeded: Seeded) -> None:
        require_reliability_attribute(seeded.conn)

    def test_code_interview_stops_without_attribute(self, db: sqlite3.Connection) -> None:
        seed(db, attributes=["ГРУППА"])
        with pytest.raises(CodingError, match=RELIABILITY_ATTRIBUTE):
            code_interview(db, "R01", _settings(), ScriptedProvider([]), code_version="test")


class TestDryRun:
    """Dry-run считает объём без сети."""

    def test_counts(self, seeded: Seeded) -> None:
        report = plan_dry_run(seeded.conn, "R01", _settings())
        assert report.fragments == 2
        assert report.utterances == 2
        assert report.model_calls == 2
        assert report.prompt_tokens > 0

    def test_tokens_cover_whole_prompt(self, seeded: Seeded) -> None:
        """Оценка включает шаблон и кодбук, а не только тексты фрагментов."""
        report = plan_dry_run(seeded.conn, "R01", _settings())
        fragment_chars = sum(len(text) for text in fragment_texts(seeded.conn).values())
        assert report.prompt_tokens > fragment_chars // 4

    def test_tokens_ignore_garbled(self, seeded: Seeded) -> None:
        """Фрагмент с маркером неразборчивости не создаёт вызов и не идёт в оценку."""
        before = plan_dry_run(seeded.conn, "R01", _settings())
        seeded.conn.execute(
            "UPDATE fragment SET text = text || ' [неразборчиво]' WHERE id = ?",
            (seeded.fragment_ids[0],),
        )
        after = plan_dry_run(seeded.conn, "R01", _settings())
        assert after.garbled == 1
        assert after.prompt_tokens < before.prompt_tokens

    def test_unknown_interview(self, db: sqlite3.Connection) -> None:
        with pytest.raises(CodingError):
            plan_dry_run(db, "R99", _settings())


class TestReport:
    """Отчёт печатает счётчики без текста реплик."""

    def test_report_lines(self, seeded: Seeded) -> None:
        texts = fragment_texts(seeded.conn)
        provider = ScriptedProvider([answer_for_fragments(texts), answer_for_fragments(texts)])
        result = code_interview(seeded.conn, "R01", _settings(), provider, code_version="test")
        lines = "\n".join(format_report(result))
        assert "coding R01" in lines
        assert "coded=" in lines
        for text in texts.values():
            assert text not in lines


class TestRunRecord:
    """Прогон записан в run с моделью, хешем промпта и версией кодбука."""

    def test_run_row(self, seeded: Seeded) -> None:
        texts = fragment_texts(seeded.conn)
        provider = ScriptedProvider([answer_for_fragments(texts), answer_for_fragments(texts)])
        result = code_interview(seeded.conn, "R01", _settings(), provider, code_version="abc123")
        row = seeded.conn.execute(
            "SELECT model, prompt_hash, codebook_version_id, code_version, params_json FROM run WHERE id = ?",
            (result.run_id,),
        ).fetchone()
        assert row is not None
        assert row[0] == "fake-model"
        assert len(str(row[1])) == 64
        assert row[2] is not None
        assert row[3] == "abc123"
        params = json.loads(str(row[4]))
        assert params["temperature"] == 0
