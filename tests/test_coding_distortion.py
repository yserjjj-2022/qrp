"""Тесты искажений ASR: качество по фрагменту, пропуск garbled, интерпретация."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from qa.pipeline.coding import code_interview
from qa.pipeline.fragment_quality import GARBLED_MARKER, fragment_asr_quality
from qa.pipeline.normalize_rules import LexiconEntry
from qa.settings import Settings
from tests.fixtures.coding_db import (
    ScriptedProvider,
    answer_for_fragments,
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


class TestFragmentQualityRule:
    """Качество считается по тексту фрагмента, а не по единице L2."""

    def test_garbled_by_marker(self) -> None:
        quality = fragment_asr_quality(
            f"тут {GARBLED_MARKER} слово",
            asr_fix_words=set(),
            new_term_words=set(),
            lexicon=[],
        )
        assert quality == "garbled"

    def test_ok_without_marks(self) -> None:
        quality = fragment_asr_quality(
            "я потом сверяю по счёту",
            asr_fix_words={"ошибиться"},
            new_term_words={"другое"},
            lexicon=[],
        )
        assert quality == "ok"

    def test_doubtful_by_asr_fix(self) -> None:
        quality = fragment_asr_quality(
            "я потом сверяю по счёту",
            asr_fix_words={"сверяю"},
            new_term_words=set(),
            lexicon=[],
        )
        assert quality == "doubtful"

    def test_doubtful_by_new_term(self) -> None:
        quality = fragment_asr_quality(
            "подключил транзакт",
            asr_fix_words=set(),
            new_term_words={"транзакт"},
            lexicon=[],
        )
        assert quality == "doubtful"

    def test_doubtful_by_suspect_word(self) -> None:
        quality = fragment_asr_quality(
            "какой-то квapк",
            asr_fix_words=set(),
            new_term_words=set(),
            lexicon=[LexiconEntry(canonical="Финам", variants=("Финанм",))],
        )
        assert quality == "doubtful"

    def test_case_does_not_matter(self) -> None:
        quality = fragment_asr_quality(
            "Потом Сверяю.",
            asr_fix_words={"сверяю"},
            new_term_words=set(),
            lexicon=[],
        )
        assert quality == "doubtful"


class TestGarbledSkipped:
    """Фрагмент с маркером неразборчивости не кодируется."""

    def test_garbled_fragment_has_no_annotations(self, db: sqlite3.Connection) -> None:
        text = "Он может ошибиться и купить не то. Я потом сверяю по счёту."
        cut = text.index(" Я")
        seeded = seed(db, fragments=[(2, 0, cut), (2, cut + 1, len(text))])
        first, second = seeded.fragment_ids
        # Второй фрагмент делаем неразборчивым: в его срезе есть маркер.
        unit_id = int(db.execute("SELECT text_unit_id FROM fragment WHERE id = ?", (second,)).fetchone()[0])
        db.execute("UPDATE text_unit SET text = ? WHERE id = ?", (text[: cut + 1] + GARBLED_MARKER, unit_id))
        db.execute("UPDATE fragment SET text = ? WHERE id = ?", (GARBLED_MARKER, second))
        db.commit()

        provider = ScriptedProvider([answer_for_fragments({first: text[:cut]})])
        result = code_interview(db, "R01", _settings(), provider, code_version="test")

        assert result.stats.skipped_garbled == 1
        statuses = dict(db.execute("SELECT fragment_id, status FROM fragment_coding").fetchall())
        assert statuses[second] == "skipped_garbled"
        annotations = [row[0] for row in db.execute("SELECT fragment_id FROM annotation").fetchall()]
        assert second not in annotations

    def test_garbled_counted_in_report(self, db: sqlite3.Connection) -> None:
        text = "Он может ошибиться и купить не то."
        seeded = seed(db, fragments=[(2, 0, len(text))])
        fragment_id = seeded.fragment_ids[0]
        db.execute("UPDATE fragment SET text = ? WHERE id = ?", (f"{text} {GARBLED_MARKER}", fragment_id))
        db.commit()

        provider = ScriptedProvider([])
        result = code_interview(db, "R01", _settings(), provider, code_version="test")
        assert result.stats.skipped_garbled == 1
        assert result.stats.model_calls == 0


class TestDoubtfulInterpretation:
    """Doubtful с непустой интерпретацией — надёжность не выше 2."""

    def test_cap_applied(self, db: sqlite3.Connection) -> None:
        text = "Он может ошибиться и купить не то."
        seeded = seed(db, fragments=[(2, 0, len(text))])
        fragment_id = seeded.fragment_ids[0]
        unit_id = int(db.execute("SELECT text_unit_id FROM fragment WHERE id = ?", (fragment_id,)).fetchone()[0])
        db.execute(
            "UPDATE text_unit SET changes_json = ? WHERE id = ?",
            (json.dumps([{"type": "asr_fix", "from": "ошибиться", "to": "ошибиться"}], ensure_ascii=False), unit_id),
        )
        db.commit()

        provider = ScriptedProvider(
            [answer_for_fragments({fragment_id: text}, reliability=3, interpretation="модель прочитала слово")]
        )
        result = code_interview(db, "R01", _settings(), provider, code_version="test")

        assert result.stats.by_quality.get("doubtful") == 1
        reliability = db.execute("SELECT reliability FROM annotation").fetchone()[0]
        assert reliability == 2
        assert result.stats.doubtful_with_interpretation == 1


class TestInterpretationOnly:
    """Код, держащийся только на интерпретации, — надёжность 1."""

    def test_reliability_one(self, db: sqlite3.Connection) -> None:
        text = "Он может ошибиться и купить не то."
        seeded = seed(db, fragments=[(2, 0, len(text))])
        fragment_id = seeded.fragment_ids[0]

        provider = ScriptedProvider(
            [
                answer_for_fragments(
                    {fragment_id: text},
                    reliability=3,
                    interpretation="догадка о смысле",
                    interpretation_only=True,
                )
            ]
        )
        result = code_interview(db, "R01", _settings(), provider, code_version="test")

        assert result.stats.interpretation_only == 1
        row = db.execute("SELECT reliability, interpretation_only FROM annotation").fetchone()
        assert row[0] == 1
        assert row[1] == 1
