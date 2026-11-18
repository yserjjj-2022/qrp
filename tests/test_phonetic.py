"""Фонетические подсказки M7b: общее пространство и пороги."""

from __future__ import annotations

from qa.pipeline.phonetic import (
    phonetic_candidates,
    phonetic_form,
    phonetic_key,
    phonetic_score,
    split_candidates,
    transliterate,
)


class TestTransliteration:
    def test_latin_becomes_cyrillic(self) -> None:
        assert transliterate("OsEngine") == "осенгине"

    def test_yo_folded(self) -> None:
        assert transliterate("ёж") == "еж"

    def test_form_keeps_vowels(self) -> None:
        assert phonetic_form("Сенжин") == "сенгин"


class TestKey:
    def test_reference_words_share_skeleton(self) -> None:
        """Три искажения и латинский канон дают один скелет."""
        assert phonetic_key("Сенжин") == phonetic_key("OsEngine") == "снгн"

    def test_short_word_uses_full_form(self) -> None:
        assert phonetic_key("эм") == "эм"

    def test_score_reference_words(self) -> None:
        assert phonetic_score("Сенжин", "OsEngine") == 1.0
        assert phonetic_score("Всенж", "OsEngine") >= 0.70
        assert phonetic_score("Усеннджин", "OsEngine") >= 0.70


class TestCandidates:
    def test_all_three_spellings_lead_to_osengine(self) -> None:
        targets = {"осенгине": "OsEngine", "финам": "Финам"}
        for form in ("сенжин", "всенж", "усеннджин"):
            found = phonetic_candidates(
                form,
                targets,
                threshold=0.55,
                short_threshold=0.70,
                short_length=5,
                limit=5,
            )
            assert found and found[0][0] == "OsEngine"

    def test_random_letters_have_no_candidates(self) -> None:
        found = phonetic_candidates(
            "щкхц",
            {"осенгине": "OsEngine"},
            threshold=0.55,
            short_threshold=0.70,
            short_length=5,
            limit=5,
        )
        assert found == []

    def test_short_word_threshold_is_stricter(self) -> None:
        targets = {"финам": "Финам"}
        assert phonetic_candidates("фин", targets, threshold=0.55, short_threshold=0.99, short_length=5, limit=5) == []


class TestSplit:
    def test_known_parts_split(self) -> None:
        known = {"сен", "жин"}
        assert split_candidates("сенжин", lambda part: part in known) == ["сен жин"]

    def test_unknown_parts_not_split(self) -> None:
        assert split_candidates("сенжин", lambda part: False) == []
