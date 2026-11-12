"""Тесты M7: отбор релевантных терминов и хеш снимка лексикона.

Все тексты синтетические. Функции чистые, сеть не используется.
"""

from __future__ import annotations

from qa.pipeline.normalize_rules import (
    LexiconEntry,
    lexicon_match_score,
    lexicon_snapshot,
    lexicon_snapshot_hash,
    relevant_lexicon,
    suspect_frequencies,
    suspect_words,
)


class TestRelevantLexicon:
    """Термины, которые могут понадобиться конкретной реплике."""

    def test_exact_canonical_hit(self) -> None:
        entries = [LexiconEntry(canonical="MCP"), LexiconEntry(canonical="ребалансировка")]
        result = relevant_lexicon("Подключил MCP и всё работает", entries)
        assert [item.canonical for item in result] == ["MCP"]

    def test_exact_variant_hit(self) -> None:
        entries = [LexiconEntry(canonical="MCP", variants=("эсипи",)), LexiconEntry(canonical="ребалансировка")]
        result = relevant_lexicon("Подключил эсипи в приложении", entries)
        assert [item.canonical for item in result] == ["MCP"]

    def test_fuzzy_hit_by_suspect_word(self) -> None:
        """Подозрительное слово похоже на термин — термин идёт в промпт."""
        entries = [LexiconEntry(canonical="ребалансировка")]
        result = relevant_lexicon("сделал ребалансировку портфеля", entries)
        assert [item.canonical for item in result] == ["ребалансировка"]

    def test_no_relevant_terms(self) -> None:
        entries = [LexiconEntry(canonical="MCP", variants=("эсипи",))]
        assert relevant_lexicon("Сегодня хорошая погода", entries) == []

    def test_multiword_term_hit(self) -> None:
        entries = [LexiconEntry(canonical="Transaq Connector", variants=("транзак коннектор",))]
        result = relevant_lexicon("Работает через транзак коннектор", entries)
        assert [item.canonical for item in result] == ["Transaq Connector"]

    def test_empty_lexicon(self) -> None:
        assert relevant_lexicon("любой текст реплики", []) == []

    def test_whole_word_only_for_exact_match(self) -> None:
        """Точное совпадение требует границ слова: «репо» не найдено в «репортаж»."""
        entry = LexiconEntry(canonical="репо")
        assert lexicon_match_score("репортаж о рынке", entry) == 1

    def test_threshold_controls_fuzzy_selection(self) -> None:
        """Строгий порог отсекает дальний нечёткий кандидат."""
        entries = [LexiconEntry(canonical="ребалансировка")]
        assert relevant_lexicon("сделал ребалансировку", entries, threshold=0.99) == []

    def test_match_score_levels(self) -> None:
        entry = LexiconEntry(canonical="ребалансировка")
        assert lexicon_match_score("про ребалансировка", entry) == 2
        assert lexicon_match_score("ребалансировка", entry) == 2
        # Словоформа — не точное вхождение: её покрывает нечёткое сравнение.
        assert lexicon_match_score("про ребалансировку", entry) == 1
        assert lexicon_match_score("совсем другое предложение", entry) == 0


class TestLexiconSnapshot:
    """Хеш версии глоссария для run.params_json."""

    def test_order_independent(self) -> None:
        first = [LexiconEntry(canonical="MCP", variants=("эсипи",)), LexiconEntry(canonical="ребалансировка")]
        second = list(reversed(first))
        assert lexicon_snapshot_hash(first) == lexicon_snapshot_hash(second)

    def test_variant_order_independent(self) -> None:
        first = [LexiconEntry(canonical="MCP", variants=("эсипи", "мсипи"))]
        second = [LexiconEntry(canonical="MCP", variants=("мсипи", "эсипи"))]
        assert lexicon_snapshot_hash(first) == lexicon_snapshot_hash(second)

    def test_added_variant_changes_hash(self) -> None:
        before = [LexiconEntry(canonical="MCP", variants=("эсипи",))]
        after = [LexiconEntry(canonical="MCP", variants=("эсипи", "мсипи"))]
        assert lexicon_snapshot_hash(before) != lexicon_snapshot_hash(after)

    def test_added_term_changes_hash(self) -> None:
        before = [LexiconEntry(canonical="MCP")]
        after = [LexiconEntry(canonical="MCP"), LexiconEntry(canonical="ребалансировка")]
        assert lexicon_snapshot_hash(before) != lexicon_snapshot_hash(after)

    def test_stable_between_calls(self) -> None:
        entries = [LexiconEntry(canonical="MCP", variants=("эсипи",))]
        assert lexicon_snapshot_hash(entries) == lexicon_snapshot_hash(entries)

    def test_snapshot_format(self) -> None:
        entries = [
            LexiconEntry(canonical="MCP", variants=("эсипи", "мсипи")),
            LexiconEntry(canonical="ребалансировка"),
        ]
        assert lexicon_snapshot(entries) == "MCP|мсипи,эсипи\nребалансировка|"


class TestSuspectFrequencies:
    """Частота подозрительных слов: нужна для порядка в очереди глоссария."""

    def test_counts_all_occurrences(self) -> None:
        assert suspect_frequencies("нечечно там нечечно опять", []) == {"нечечно": 2}

    def test_matches_suspect_words_set(self) -> None:
        text = "автослеле и нечечно, снова автослеле"
        assert set(suspect_frequencies(text, [])) == set(suspect_words(text, []))

    def test_known_words_not_counted(self) -> None:
        assert suspect_frequencies("это обычное интервью про счёт", []) == {}

    def test_empty_text(self) -> None:
        assert suspect_frequencies("", []) == {}
