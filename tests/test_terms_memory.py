"""Память терминов M7b: статусы лексикона, приоритет решений, подсказки."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from qa.db.workbook_repo import LexiconRepository
from qa.pipeline.normalize import load_lexicon
from qa.pipeline.normalize_rules import LexiconEntry
from qa.pipeline.terms_memory import (
    KEEP,
    UNREADABLE,
    TermHint,
    build_code_hints,
    merge_hints,
    pick_with_model,
)
from qa.settings import M7bSettings
from tests.fixtures.m8_db import M8Spec, UtteranceSpec, build_m8_db

_UTTERANCE = UtteranceSpec("RESPONDENT", "Текст", "Текст.", 10)


def _db(tmp_path: Path) -> sqlite3.Connection:
    return build_m8_db(tmp_path, M8Spec(utterances=[_UTTERANCE]))


class TestLexiconStatuses:
    def test_keep_is_known_without_replacement(self, tmp_path: Path) -> None:
        conn = _db(tmp_path)
        LexiconRepository(conn).set_form_decision("сенжин", "сенжин", "keep")
        entries = load_lexicon(conn)
        entry = next(item for item in entries if item.canonical == "сенжин")
        assert entry.variants == ()
        assert entry.status == "keep"

    def test_unreadable_replaces_form(self, tmp_path: Path) -> None:
        conn = _db(tmp_path)
        LexiconRepository(conn).set_form_decision("сенжин", "сенжин", "unreadable")
        entries = load_lexicon(conn)
        entry = next(item for item in entries if item.canonical == UNREADABLE)
        assert entry.variants == ("сенжин",)

    def test_auto_status_is_memory(self, tmp_path: Path) -> None:
        conn = _db(tmp_path)
        LexiconRepository(conn).set_form_decision("сенжин", "OsEngine", "auto")
        entry = next(item for item in load_lexicon(conn) if item.canonical == "OsEngine")
        assert entry.status == "auto"
        assert "сенжин" in entry.variants


class TestDecisionPriority:
    def test_operator_not_overwritten_by_auto(self, tmp_path: Path) -> None:
        conn = _db(tmp_path)
        repo = LexiconRepository(conn)
        assert repo.set_form_decision("сенжин", "OsEngine", "review") == "applied"
        assert repo.set_form_decision("сенжин", "Другой", "auto") == "operator_kept"
        owner = repo.form_owner("сенжин")
        assert owner == ("OsEngine", "review")

    def test_auto_upgraded_by_operator(self, tmp_path: Path) -> None:
        conn = _db(tmp_path)
        repo = LexiconRepository(conn)
        repo.set_form_decision("сенжин", "Другой", "auto")
        assert repo.set_form_decision("сенжин", "OsEngine", "review") == "applied"
        assert repo.form_owner("сенжин") == ("OsEngine", "review")
        assert repo.form_owner("сенжин") is not None

    def test_two_operator_decisions_conflict(self, tmp_path: Path) -> None:
        conn = _db(tmp_path)
        repo = LexiconRepository(conn)
        repo.set_form_decision("сенжин", "OsEngine", "review")
        assert repo.set_form_decision("сенжин", "Другой", "review") == "conflict"
        assert repo.form_owner("сенжин") == ("OsEngine", "review")

    def test_same_decision_is_unchanged(self, tmp_path: Path) -> None:
        conn = _db(tmp_path)
        repo = LexiconRepository(conn)
        repo.set_form_decision("сенжин", "OsEngine", "review")
        assert repo.set_form_decision("сенжин", "OsEngine", "review") == "unchanged"


class TestCodeHints:
    def test_phonetic_hint_for_reference_spelling(self) -> None:
        settings = M7bSettings()
        entries = [LexiconEntry(canonical="OsEngine")]
        hints = build_code_hints("сенжин", entries, settings)
        assert hints and hints[0].to == "OsEngine"
        assert hints[0].source == "phonetic"

    def test_random_letters_give_no_hints(self) -> None:
        settings = M7bSettings()
        assert build_code_hints("щкхц", [LexiconEntry(canonical="OsEngine")], settings) == []


class _Provider:
    """Заглушка модели: возвращает заданные ответы по очереди."""

    def __init__(self, answers: list[dict[str, Any]]) -> None:
        self._answers = list(answers)

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        return self._answers.pop(0)


class TestModelPick:
    def test_confidence_is_share_of_samples(self) -> None:
        hints = [TermHint("OsEngine", 0.9, "phonetic"), TermHint("Финам", 0.6, "phonetic")]
        providers = [
            _Provider([{"verdict": "candidate", "candidate": 0, "reason": ""}]),
            _Provider([{"verdict": "candidate", "candidate": 0, "reason": ""}]),
            _Provider([{"verdict": "keep", "candidate": 0, "reason": ""}]),
        ]
        picked = pick_with_model("сенжин", "контекст", hints, [], providers)
        by_target = {hint.to: hint.confidence for hint in picked}
        assert by_target["OsEngine"] == 2 / 3
        assert by_target[KEEP] == 1 / 3

    def test_index_outside_list_is_dropped(self) -> None:
        """Модель не может вернуть слово вне списка: индекс проверяется."""
        hints = [TermHint("OsEngine", 0.9, "phonetic")]
        providers = [_Provider([{"verdict": "candidate", "candidate": 99, "reason": ""}])]
        assert pick_with_model("сенжин", "контекст", hints, [], providers) == []

    def test_no_providers_gives_no_model_hints(self) -> None:
        hints = [TermHint("OsEngine", 0.9, "phonetic")]
        assert pick_with_model("сенжин", "", hints, [], []) == []


class TestMerge:
    def test_model_hint_wins_and_variants_unique(self) -> None:
        code = [TermHint("OsEngine", 0.8, "phonetic"), TermHint("Финам", 0.6, "phonetic")]
        model = [TermHint("OsEngine", 1.0, "model"), TermHint(KEEP, 0.3, "model")]
        merged = merge_hints(code, model, 5)
        assert [hint.to for hint in merged] == ["OsEngine", "Финам", KEEP]
        assert merged[0].confidence == 1.0
