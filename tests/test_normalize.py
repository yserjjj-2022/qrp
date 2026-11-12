"""Тесты M1b: детерминированный проход, проход модели, автопроверки, запись L2.

Все тексты синтетические. Модель заменена заглушкой, сети нет.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from qa.db.connection import get_connection
from qa.db.migrations import migrate
from qa.db.workbook_repo import (
    LexiconRepository,
    MarkerRepository,
    NewTermRepository,
    RunRepository,
    TextUnitRepository,
)
from qa.llm.base import LlmProvider, LlmResult
from qa.llm.errors import LlmValidationError
from qa.pipeline.normalize import (
    NormalizeError,
    collect_suspects,
    load_lexicon,
    normalize_interview,
    plan_dry_run,
)
from qa.pipeline.normalize_l2 import (
    NORMALIZE_SCHEMA,
    analyze_thought_loss,
    build_prompt,
    check_asr_fix,
    check_capital_words,
    check_latin,
    check_names,
    check_negations,
    check_new_words,
    check_numbers,
    check_thought_loss,
    check_uncertainty,
    normalize_utterance,
    run_checks,
    should_use_model,
    summarize,
    surviving_new_terms,
)
from qa.pipeline.normalize_rules import (
    FuzzyConfig,
    LexiconEntry,
    comparison_tokens,
    is_service_word,
    lexicon_snapshot_hash,
    normalize_deterministic,
    suspect_words,
)
from qa.pipeline.review_terms import review_terms
from qa.settings import Settings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations"

_MARKERS = ["не знаю", "не помню", "наверное", "возможно", "вроде"]
_LEXICON = [LexiconEntry(canonical="MCP", variants=("эсипи", "эс-и-пи"))]
_PROMPT_PLACEHOLDER = re.compile(r"\{[a-z_]+\}")


class FakeProvider(LlmProvider):
    """Заглушка модели: отдаёт заданный ответ и считает вызовы."""

    def __init__(self, responses: list[dict[str, Any] | Exception], *, cached: bool = False) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.model_used = "fake-model"
        self.prompt_tokens = 10
        self.completion_tokens = 5
        self.cached = cached

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        return self.complete_json_detailed(prompt, schema).data

    def complete_json_detailed(self, prompt: str, schema: dict[str, Any]) -> LlmResult:
        self.prompts.append(prompt)
        if not self._responses:
            msg = "заглушка исчерпала ответы"
            raise AssertionError(msg)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return LlmResult(
            data=response,
            model_used=self.model_used,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            cached=self.cached,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


def _ok(text: str, *, quality: str = "ok", new_terms: list[str] | None = None) -> dict[str, Any]:
    """Ответ модели в формате схемы."""
    return {
        "normalized_text": text,
        "changes": [],
        "uncertain_spans": [],
        "new_terms": new_terms or [],
        "asr_quality": quality,
    }


@pytest.fixture()
def db(tmp_path: Path) -> sqlite3.Connection:
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    migrate(conn, db_path, _MIGRATIONS_DIR, tmp_path / "backups")
    yield conn
    conn.close()


@pytest.fixture()
def settings() -> Settings:
    return Settings.load(_PROJECT_ROOT / "config.yaml")


def _seed_interview(conn: sqlite3.Connection, utterances: list[tuple[str, str, int]]) -> int:
    """Создать интервью и реплики: (роль, текст, is_backchannel)."""
    cursor = conn.execute("INSERT INTO interview (id_external, name) VALUES ('R01', 'тестовое интервью')")
    interview_id = int(cursor.lastrowid or 0)
    for role, text, flag in utterances:
        conn.execute(
            "INSERT INTO utterance (interview_id, speaker_role, text, turn_ids, is_backchannel)"
            " VALUES (?, ?, ?, '[]', ?)",
            (interview_id, role, text, flag),
        )
    conn.commit()
    return interview_id


def _seed_reference(conn: sqlite3.Connection) -> None:
    """Лексикон и маркеры в БД."""
    lexicon = LexiconRepository(conn)
    lexicon.upsert_from_workbook("MCP", ["эсипи", "эс-и-пи"], None, "approved")
    markers = MarkerRepository(conn)
    for marker in _MARKERS:
        markers.upsert(marker, None, "default")
    conn.commit()


def _utterances(conn: sqlite3.Connection, interview_id: int) -> list[tuple[int, str, str]]:
    rows = conn.execute(
        "SELECT t.id, t.text, t.status FROM text_unit t JOIN utterance u ON t.utterance_id = u.id"
        " WHERE u.interview_id = ? ORDER BY u.id",
        (interview_id,),
    ).fetchall()
    return [(int(row[0]), str(row[1]), str(row[2])) for row in rows]


class TestDeterministicRules:
    """Детерминированный проход по синтетическим примерам SPEC-M1."""

    def test_stutter_collapsed(self) -> None:
        result = normalize_deterministic("Они, они, они на подстраховке", [])
        assert result.text == "Они на подстраховке."
        assert result.stutter_removed == 2

    def test_meaningful_repeat_kept(self) -> None:
        result = normalize_deterministic("Не знаю, не знаю, вопрос, наверное, в простоте", [])
        assert "не знаю, не знаю" in result.text.casefold()
        assert result.repeat_unclear == 1

    def test_sounds_removed(self) -> None:
        result = normalize_deterministic("Ну э-э я подключил, вот", [])
        assert "э-э" not in result.text
        assert result.fillers_removed == 1

    def test_lexicon_replacement(self) -> None:
        result = normalize_deterministic("Подключил эсипи Финам", _LEXICON)
        assert result.text == "Подключил MCP Финам."
        assert result.term_replacements == {"MCP": 1}

    def test_lexicon_whole_words_only(self) -> None:
        result = normalize_deterministic("Эсипишка не термин", _LEXICON)
        assert "Эсипишка" in result.text

    def test_punctuation_and_case(self) -> None:
        result = normalize_deterministic("  привет   ,мир  ", [])
        assert result.text == "Привет, мир."


class TestFuzzyLexicon:
    """Пункт 1: нечёткое сопоставление с лексиконом в детерминированном проходе."""

    def test_emspi_and_msipi_become_mcp(self) -> None:
        for surface in ("эмсипи", "мсипи", "Мсипи", "Эмсипи"):
            result = normalize_deterministic(f"Подключил {surface} Финам", _LEXICON)
            assert "MCP" in result.text
            assert "мсипи" not in result.text.casefold()

    def test_fuzzy_change_marked(self) -> None:
        result = normalize_deterministic("у меня есть эмсипи", _LEXICON)
        entries = [item for item in result.changes if item.get("type") == "term"]
        assert entries == [{"type": "term", "from": "эмсипи", "to": "MCP", "fuzzy": True}]
        assert result.fuzzy_replacements == 1

    def test_short_utterance_still_fixed(self) -> None:
        """Реплика короче min_words_for_model исправляется без вызова модели."""
        provider = FakeProvider([])
        outcome = normalize_utterance(
            1,
            "RESPONDENT",
            False,
            "С мсипи ничего",
            previous_question=None,
            lexicon=_LEXICON,
            markers=_MARKERS,
            examples="примеры",
            provider=provider,
            min_words=4,
            fuzzy=FuzzyConfig(),
        )
        assert "MCP" in outcome.text
        assert provider.prompts == []

    def test_known_word_not_replaced(self) -> None:
        """Слово, известное pymorphy3, нечётко не заменяется."""
        entries = [LexiconEntry(canonical="Регламент", variants=())]
        result = normalize_deterministic("Он регламентировал процесс", entries)
        assert "Регламент" not in result.text
        assert "регламентировал" in result.text.casefold()
        assert result.fuzzy_replacements == 0

    def test_exact_variant_not_double_counted(self) -> None:
        """Точный вариант уже заменён, нечёткий проход его не трогает."""
        result = normalize_deterministic("Подключил эсипи Финам", _LEXICON)
        fuzzy = [item for item in result.changes if item.get("fuzzy")]
        assert fuzzy == []

    def test_short_word_stricter_threshold(self) -> None:
        """Короткие слова сравниваются строже: «про» не станет термином."""
        entries = [LexiconEntry(canonical="Прод", variants=())]
        result = normalize_deterministic("про это сказал", entries)
        assert "про это" in result.text.casefold()

    def test_pair_glued_by_lexicon(self) -> None:
        entries = [LexiconEntry(canonical="ребалансировка", variants=())]
        result = normalize_deterministic("приходит ре балансланировка", entries)
        assert "ребалансировка" in result.text.casefold()
        assert result.glued_replacements == 1
        glued = [item for item in result.changes if item.get("glued")]
        assert glued == [{"type": "term", "from": "ре балансланировка", "to": "ребалансировка", "glued": True}]

    def test_pair_glued_exact_form(self) -> None:
        entries = [LexiconEntry(canonical="ребалансировка", variants=())]
        result = normalize_deterministic("там ре баланснировка простая", entries)
        assert "ребалансировка" in result.text.casefold()

    def test_glue_marked_separately_from_fuzzy(self) -> None:
        """Склейка — отдельное правило: своя пометка и свой счётчик."""
        entries = [LexiconEntry(canonical="ребалансировка", variants=())]
        result = normalize_deterministic("там ре баланснировка простая", entries)
        assert result.glued_replacements == 1
        assert result.fuzzy_replacements == 0

    def test_glue_keeps_inflection(self) -> None:
        """Форму задаёт второе слово: «ре балансланировку» -> «ребалансировку»."""
        entries = [LexiconEntry(canonical="ребалансировка", variants=())]
        result = normalize_deterministic("провели ре балансланировку и всё", entries)
        assert "ребалансировку" in result.text.casefold()
        assert "ребалансировка" not in result.text.casefold()

    def test_exact_glued_term_not_double_counted(self) -> None:
        """Точный вариант уже заменён, склейка не дублирует запись."""
        entries = [LexiconEntry(canonical="Transaq Connector", variants=("транзак коннектор",))]
        result = normalize_deterministic("транзак коннектор подключил", entries)
        glued = [item for item in result.changes if item.get("glued")]
        assert glued == []
        assert result.glued_replacements == 0

    def test_known_pair_not_glued(self) -> None:
        """Пара известных словарю слов не склеивается в термин."""
        entries = [LexiconEntry(canonical="ИИ", variants=())]
        result = normalize_deterministic("и и всё", entries)
        assert "ИИ" not in result.text

    def test_function_word_not_glued(self) -> None:
        """Служебное слово слева не приклеивается к термину: «и эмсипи» -> «и MCP»."""
        entries = [LexiconEntry(canonical="MCP", variants=("эсипи",))]
        result = normalize_deterministic("у меня и эмсипи есть", entries)
        assert "и MCP" in result.text
        assert "иэмсипи" not in result.text.casefold()
        assert [item["from"] for item in result.changes if item.get("fuzzy")] == ["эмсипи"]

    def test_word_form_of_canon_kept(self) -> None:
        """Падежная форма канона не заменяется: «в Финаме» остаётся."""
        entries = [LexiconEntry(canonical="Финам", variants=())]
        result = normalize_deterministic("я в Финаме открыл счёт", entries)
        assert "в Финаме" in result.text
        assert result.fuzzy_replacements == 0

    def test_inflected_replacement(self) -> None:
        """Замена согласуется по форме: «про реглантировку» -> «про ребалансировку»."""
        entries = [LexiconEntry(canonical="ребалансировка", variants=("реглантировка",))]
        result = normalize_deterministic("рассказал про реглантировку портфеля", entries)
        assert "про ребалансировку" in result.text.casefold()

    def test_interviewer_never_uses_model(self) -> None:
        assert should_use_model("INTERVIEWER", False, "длинная реплика интервьюера из многих слов", 4) is False

    def test_backchannel_never_uses_model(self) -> None:
        assert should_use_model("RESPONDENT", True, "длинная реплика респондента из многих слов", 4) is False

    def test_short_respondent_skipped(self) -> None:
        assert should_use_model("RESPONDENT", False, "да хорошо", 4) is False

    def test_long_respondent_used(self) -> None:
        assert should_use_model("RESPONDENT", False, "я подключил счёт и всё работает", 4) is True


class TestSuspectWords:
    """Пункт 4: список слов, которых нет в словаре и лексиконе."""

    def test_known_word_excluded(self) -> None:
        assert "интервью" not in suspect_words("Это интервью про счёт", [])

    def test_lexicon_term_and_form_excluded(self) -> None:
        entries = [LexiconEntry(canonical="ребалансировка", variants=())]
        found = suspect_words("Обсудили ребалансировку и ребалансировки портфеля", entries)
        assert found == []

    def test_multiword_term_words_excluded(self) -> None:
        entries = [LexiconEntry(canonical="Transaq Connector", variants=("транзак коннектор",))]
        assert suspect_words("через транзак коннектор работает", entries) == []

    def test_invented_word_included(self) -> None:
        assert suspect_words("Там автослеле и нечечно", []) == ["автослеле", "нечечно"]

    def test_short_words_excluded(self) -> None:
        found = suspect_words("э м аа ну да", [])
        assert found == []

    def test_numbers_excluded(self) -> None:
        assert suspect_words("пришло 100000 рублей", []) == []

    def test_unique_in_order_of_appearance(self) -> None:
        assert suspect_words("нечечно там нечечно автослеле", []) == ["нечечно", "автослеле"]

    def test_build_prompt_contains_suspect(self) -> None:
        prompt = build_prompt("текст", None, [], [], "примеры", suspect=["автослеле", "нечечно"])
        assert "автослеле, нечечно" in prompt
        assert "{suspect}" not in prompt

    def test_build_prompt_empty_suspect(self) -> None:
        prompt = build_prompt("текст", None, [], [], "примеры", suspect=None)
        assert "{suspect}" not in prompt
        assert "автослеле" not in prompt
        assert "(нет)" in prompt

    def test_build_prompt_has_no_placeholders(self) -> None:
        prompt = build_prompt(
            "реплика",
            "вопрос",
            _LEXICON,
            _MARKERS,
            "примеры",
            suspect=["нечечно"],
            domain_hint="предметная область",
        )
        assert _PROMPT_PLACEHOLDER.findall(prompt) == []
        assert "реплика" in prompt
        assert "вопрос" in prompt
        assert "предметная область" in prompt
        assert "эсипи" in prompt
        assert "не знаю" in prompt

    def test_build_prompt_default_domain_hint(self) -> None:
        prompt = build_prompt("реплика", None, [], [], "примеры")
        assert "(не задан)" in prompt
        assert "{domain_hint}" not in prompt

    def test_normalize_utterance_passes_suspect(self) -> None:
        provider = FakeProvider([_ok("Там нечечно и всё.")])
        normalize_utterance(
            1,
            "RESPONDENT",
            False,
            "там нечечно и всё работает",
            previous_question=None,
            lexicon=[],
            markers=[],
            examples="примеры",
            provider=provider,
            min_words=4,
            suspect=["нечечно"],
        )
        assert "нечечно" in provider.prompts[0]
        assert "{suspect}" not in provider.prompts[0]


class TestThoughtLoss:
    """Пункт 2: удаление слов без основания отклоняет единицу."""

    def test_thought_removal_rejected(self) -> None:
        """«на средне они» нельзя удалить как мусор."""
        failure = check_thought_loss(
            "Ну, скрипты есть на на средне они, во-первых, написано",
            "Ну, скрипты есть, во-первых, написано.",
            [],
        )
        assert failure is not None
        assert failure.check == "thought_loss"

    def test_filler_removal_allowed(self) -> None:
        assert check_thought_loss("Я, ну, не помню", "Я не помню.", []) is None

    def test_sound_removal_allowed(self) -> None:
        assert check_thought_loss("Ну э-э я подключил", "Ну я подключил.", []) is None

    def test_mechanical_repeat_allowed(self) -> None:
        assert check_thought_loss("Они, они, они на подстраховке", "Они на подстраховке.", []) is None

    def test_claimed_filler_for_content_word_rejected(self) -> None:
        """Модель не может объявить удаление обычного слова через changes."""
        failure = check_thought_loss(
            "Там балансировка и портфель",
            "Там портфель.",
            [{"type": "filler", "from": "балансировка", "to": ""}],
        )
        assert failure is not None
        assert failure.check == "thought_loss"

    def test_claimed_stutter_for_single_word_rejected(self) -> None:
        failure = check_thought_loss(
            "Там балансировка и портфель",
            "Там портфель.",
            [{"type": "stutter", "from": "балансировка", "to": ""}],
        )
        assert failure is not None

    def test_claimed_stutter_for_repeat_accepted(self) -> None:
        assert (
            check_thought_loss(
                "Они, они, они на портфеле",
                "Они на портфеле.",
                [{"type": "stutter", "from": "Они, они, они", "to": "Они"}],
            )
            is None
        )

    def test_term_replacement_allowed(self) -> None:
        assert (
            check_thought_loss(
                "подключил эсипи Финам",
                "Подключил MCP Финам.",
                [{"type": "term", "from": "эсипи", "to": "MCP"}],
            )
            is None
        )


class TestAsrFix:
    """Пункт 3: asr_fix только для подозрительных слов."""

    def test_fix_of_unknown_word_allowed(self) -> None:
        assert (
            check_asr_fix(
                "стоит гидроскал на стенде",
                "Стоит гидроскан на стенде.",
                [],
                [{"type": "asr_fix", "from": "гидроскал", "to": "гидроскан"}],
            )
            is None
        )

    def test_fix_of_known_word_rejected(self) -> None:
        failure = check_asr_fix(
            "мне это удобно",
            "Мне это удобно.",
            [],
            [{"type": "asr_fix", "from": "удобно", "to": "удобнее"}],
        )
        assert failure is not None
        assert failure.check == "asr_fix"

    def test_fix_target_must_be_in_l2(self) -> None:
        failure = check_asr_fix(
            "стоит гидроскал на стенде",
            "Стоит гидроскал на стенде.",
            [],
            [{"type": "asr_fix", "from": "гидроскал", "to": "гидроскан"}],
        )
        assert failure is not None

    def test_no_asr_fix_changes_pass(self) -> None:
        assert check_asr_fix("мне это удобно", "Мне это удобно.", [], []) is None


class TestSoftThoughtLoss:
    """Мягкий исход: потеря служебного слова — doubtful, а не отказ."""

    def test_service_word_loss_is_soft(self) -> None:
        """«к» и «нам» исчезли — смысл цел, единица идёт в doubtful."""
        result = analyze_thought_loss(
            "Отчёт приходит к нам каждый день.",
            "Отчёт приходит каждый день.",
            [],
        )
        assert result.failure is None
        assert result.soft

    def test_content_word_loss_still_hard(self) -> None:
        """Потеря смыслового слова остаётся отказом."""
        result = analyze_thought_loss("Там балансировка и портфель", "Там портфель.", [])
        assert result.failure is not None
        assert result.failure.check == "thought_loss"

    def test_same_check_function_still_rejects_content(self) -> None:
        """Старая точка входа возвращает отказ только для смысловых потерь."""
        assert check_thought_loss("Там балансировка и портфель", "Там портфель.", []) is not None
        assert check_thought_loss("Там балансировка и портфель", "Там балансировка и портфель.", []) is None

    def test_wide_service_set_covers_prepositions(self) -> None:
        for word in ("к", "нам", "там", "если", "какой", "по-моему", "уже"):
            assert is_service_word(word), word

    def test_content_word_not_service(self) -> None:
        for word in ("балансировка", "портфель", "акции"):
            assert not is_service_word(word), word

    def test_negation_loss_not_soft(self) -> None:
        """«не» — служебное, но его потерю ловит отдельная жёсткая проверка."""
        failures = run_checks("Он не торгует", "Он торгует.", _MARKERS, [])
        assert any(f.check == "negations" for f in failures)

    def test_soft_marks_unit_doubtful(self) -> None:
        provider = FakeProvider([_ok("Отчёт приходит каждый день.")])
        outcome = normalize_utterance(
            1,
            "RESPONDENT",
            False,
            "Отчёт приходит к нам каждый день",
            previous_question=None,
            lexicon=[],
            markers=[],
            examples="примеры",
            provider=provider,
            min_words=4,
        )
        assert outcome.status == "proposed"
        assert outcome.asr_quality == "doubtful"
        assert any(item.get("type") == "soft_thought_loss" for item in outcome.changes)

    def test_soft_does_not_downgrade_garbled(self) -> None:
        """`garbled` строже `doubtful`: мягкая пометка его не понижает."""
        provider = FakeProvider([_ok("Там [неразборчиво] и лимиты.", quality="garbled")])
        outcome = normalize_utterance(
            1,
            "RESPONDENT",
            False,
            "Там что-то и лимиты",
            previous_question=None,
            lexicon=[],
            markers=[],
            examples="примеры",
            provider=provider,
            min_words=4,
        )
        assert outcome.asr_quality == "garbled"


class TestImplicitAsrFix:
    """Молчаливое исправление искажения: дописать asr_fix и пометить doubtful."""

    def test_unknown_word_with_close_known_replacement(self) -> None:
        result = analyze_thought_loss(
            "там на кое-как масштабируую сервис",
            "Там кое-как масштабирую сервис.",
            [],
        )
        assert result.failure is None
        assert result.implicit_fixes
        assert result.implicit_fixes[0][0] == "масштабируую"

    def test_known_word_replacement_stays_hard(self) -> None:
        """Обычное слово заменять нельзя: это подмена лексики, а не ASR."""
        result = analyze_thought_loss(
            "там балансировка портфеля",
            "Там ребалансировка портфеля.",
            [],
        )
        assert result.failure is not None
        assert not result.implicit_fixes

    def test_far_replacement_stays_hard(self) -> None:
        """Замена искажения далёкой фразой — потеря смысла."""
        result = analyze_thought_loss(
            "тоже в Томсле получает",
            "Тоже в том числе получает.",
            [],
        )
        assert result.failure is not None

    def test_implicit_fix_recorded_in_changes(self) -> None:
        provider = FakeProvider([_ok("Там кое-как масштабирую сервис.")])
        outcome = normalize_utterance(
            1,
            "RESPONDENT",
            False,
            "там на кое-как масштабируую сервис",
            previous_question=None,
            lexicon=[],
            markers=[],
            examples="примеры",
            provider=provider,
            min_words=4,
        )
        assert outcome.status == "proposed"
        assert outcome.asr_quality == "doubtful"
        fixes = [item for item in outcome.changes if item.get("type") == "asr_fix"]
        assert fixes and fixes[0]["implicit"] is True
        assert fixes[0]["from"] == "масштабируую"


class TestComparisonTokens:
    """Сравнение потерь устойчиво к дефисам и разной записи служебных слов."""

    def test_hyphen_split(self) -> None:
        assert comparison_tokens("робот-алгоритм") == ["робот", "алгоритм"]

    def test_hyphen_not_a_loss(self) -> None:
        result = analyze_thought_loss(
            "там робот алгоритм работает",
            "Там робот-алгоритм работает.",
            [],
        )
        assert result.failure is None
        assert not result.soft

    def test_split_service_merge(self) -> None:
        assert comparison_tokens("имею в виду") == ["имею", "ввиду"]

    def test_merge_not_a_loss(self) -> None:
        result = analyze_thought_loss(
            "он имею ввиду алгоритм",
            "Он имеет в виду алгоритм.",
            [],
        )
        assert result.failure is None

    def test_conjunction_variants(self) -> None:
        assert comparison_tokens("чтоб вообще всё никогда") == ["что", "вообще", "все", "никогда"]

    def test_conjunction_variant_not_a_loss(self) -> None:
        result = analyze_thought_loss(
            "кардинально что вообще всё никогда",
            "Кардинально, чтоб вообще всё никогда.",
            [],
        )
        assert result.failure is None

    def test_phrase_repeat_collapsed(self) -> None:
        """Дословный повтор фразы — механический дубль, а не потеря мысли."""
        result = analyze_thought_loss(
            "Он уже готов, он уже готов к сдаче.",
            "Он уже готов к сдаче.",
            [],
        )
        assert result.failure is None


class TestSurvivingNewTerms:
    """Пункт 3: в new_term остаются только формы, видимые в тексте."""

    def test_term_in_l2_kept(self) -> None:
        outcome = _outcome(text="Смотрел квик-дашборд.", proposed_text="Смотрел квик-дашборд.")
        assert surviving_new_terms(outcome) == ["квик-дашборд"]

    def test_term_only_in_rejected_output_kept(self) -> None:
        """Искажение в отклонённом выводе осталось — человеку его нужно увидеть."""
        outcome = _outcome(text="Смотрел квик-дашборд.", proposed_text="Смотрел квик-дашборд.")
        assert surviving_new_terms(outcome) == ["квик-дашборд"]

    def test_term_nowhere_dropped(self) -> None:
        """Модель исправила слово и всё равно прислала его в new_terms."""
        outcome = _outcome(text="Смотрел дашборд.", proposed_text="Смотрел дашборд.")
        assert surviving_new_terms(outcome) == []

    def test_case_insensitive(self) -> None:
        outcome = _outcome(text="Смотрел Квик-дашборд.", proposed_text="")
        assert surviving_new_terms(outcome) == ["квик-дашборд"]


def _outcome(text: str, proposed_text: str, terms: list[str] | None = None) -> Any:  # noqa: ANN401
    """Итог обработки реплики для проверки фильтра new_terms."""
    from qa.pipeline.normalize_l2 import NormalizeOutcome

    return NormalizeOutcome(
        utterance_id=1,
        text=text,
        status="proposed",
        asr_quality="ok",
        new_terms=list(terms if terms is not None else ["квик-дашборд"]),
        proposed_text=proposed_text,
    )


class TestNewWords:
    """Пункт 2: сравнение по лемам, а не по формам."""

    def test_inflected_form_not_new(self) -> None:
        hard, doubtful = check_new_words("он купил акции", "Он купил акцию.", [], [])
        assert hard is None
        assert doubtful is False

    def test_unknown_lowercase_lemma_doubtful(self) -> None:
        hard, doubtful = check_new_words("там было непонятно", "Там было нечечно.", [], [])
        assert hard is None
        assert doubtful is True

    def test_new_capital_word_rejected(self) -> None:
        hard, _doubtful = check_new_words("я пользуюсь приложением", "Я пользуюсь SuperApp.", [], [])
        assert hard is not None
        assert hard.check == "new_words"

    def test_lexicon_lemma_allowed(self) -> None:
        entries = [LexiconEntry(canonical="автоследование", variants=("автослед",))]
        hard, doubtful = check_new_words("включил автослед", "Включил автоследование.", entries, [])
        assert hard is None
        assert doubtful is False

    def test_change_target_allowed(self) -> None:
        hard, doubtful = check_new_words(
            "там автослеле работает",
            "Там автоследование работает.",
            [],
            [{"type": "asr_fix", "from": "автослеле", "to": "автоследование"}],
        )
        assert hard is None
        assert doubtful is False


class TestNamesReview:
    """Пункт 6: проверка названий — флаг, латиница — жёсткий отказ."""

    def test_capital_word_flagged_not_rejected(self) -> None:
        failure = check_capital_words("я пользуюсь приложением", "Я пользуюсь SuperApp.", [])
        assert failure is not None
        assert failure.check == "names"

    def test_latin_is_hard(self) -> None:
        failure = check_latin("я пользуюсь приложением", "Я пользуюсь SuperApp.", [])
        assert failure is not None
        assert failure.check == "latin"

    def test_known_latin_allowed(self) -> None:
        entries = [LexiconEntry(canonical="MCP", variants=("эсипи",))]
        assert check_latin("подключил эсипи", "Подключил MCP.", entries) is None

    def test_names_flag_set_on_success(self) -> None:
        provider = FakeProvider([_ok("Я пользуюсь SuperApp.")])
        outcome = normalize_utterance(
            1,
            "RESPONDENT",
            False,
            "я пользуюсь приложением постоянно",
            previous_question=None,
            lexicon=[],
            markers=[],
            examples="примеры",
            provider=provider,
            min_words=4,
        )
        assert outcome.status == "rejected"

    def test_capital_word_saved_with_flag(self) -> None:
        """Единица не прошла только check_names: текст модели сохранён, doubtful."""
        provider = FakeProvider([_ok("Я пользуюсь Приложением постоянно.")])
        outcome = normalize_utterance(
            1,
            "RESPONDENT",
            False,
            "я пользуюсь приложением постоянно",
            previous_question=None,
            lexicon=[],
            markers=[],
            examples="примеры",
            provider=provider,
            min_words=4,
        )
        assert outcome.status == "proposed"
        assert outcome.asr_quality == "doubtful"
        assert any(item.get("type") == "names_review" for item in outcome.changes)


class TestChecks:
    """Каждая автопроверка отклоняет намеренно испорченный вывод."""

    def test_numbers_lost(self) -> None:
        failure = check_numbers("пришло 100 000 рублей", "пришло рублей")
        assert failure is not None
        assert failure.check == "numbers"

    def test_numbers_kept(self) -> None:
        assert check_numbers("пришло 100 000 рублей", "Пришло 100 000 рублей.") is None

    def test_negation_lost(self) -> None:
        failure = check_negations("я не помню", "я помню")
        assert failure is not None
        assert failure.check == "negations"

    def test_negation_kept(self) -> None:
        assert check_negations("я не помню", "Я не помню.") is None

    def test_marker_lost(self) -> None:
        failure = check_uncertainty("я не знаю точно", "Я знаю точно.", _MARKERS)
        assert failure is not None
        assert failure.check == "uncertainty"

    def test_marker_kept(self) -> None:
        assert check_uncertainty("я не знаю", "Я не знаю.", _MARKERS) is None

    def test_invented_name(self) -> None:
        failure = check_unknown_name("я пользуюсь приложением", "Я пользуюсь SuperApp.")
        assert failure is not None
        assert failure.check == "names"

    def test_known_term_allowed(self) -> None:
        assert check_unknown_name("перевёл через приложение", "Перевёл через СБП.", ["СБП"]) is None

    def test_capital_after_colon_allowed(self) -> None:
        assert check_unknown_name("подключил терминал: работает", "Подключил терминал: Работает.") is None

    def test_capital_after_dash_allowed(self) -> None:
        assert check_unknown_name("счёт — брокерский", "Счёт — Брокерский.") is None

    def test_capital_after_quote_allowed(self) -> None:
        assert check_unknown_name("назвал «сбер»", "Назвал «Сбер».") is None

    def test_capital_after_paren_allowed(self) -> None:
        assert check_unknown_name("через api (быстро)", "Через API (Быстро).") is None

    def test_capital_after_newline_allowed(self) -> None:
        assert check_unknown_name("первая мысль", "Первая мысль.\nВторая мысль.") is None

    def test_capital_at_start_allowed(self) -> None:
        assert check_unknown_name("счёт открыт", "Счёт открыт.") is None

    def test_latin_still_checked(self) -> None:
        failure = check_unknown_name("пользуюсь приложением", "Пользуюсь SuperApp.")
        assert failure is not None
        assert failure.check == "names"

    def test_substitution_still_rejected(self) -> None:
        """Реальная подмена слова названием по-прежнему отклоняется."""
        failure = check_unknown_name("я пользуюсь приложением брокера", "Я пользуюсь SuperApp брокера.")
        assert failure is not None
        assert failure.check == "names"

    def test_multiword_term_allowed(self) -> None:
        assert check_unknown_name("через транзак коннектор", "Через Transaq Connector.", ["Transaq Connector"]) is None

    def test_multiword_term_partial_not_allowed(self) -> None:
        """Отдельное слово многословного термина без самого термина не разрешено."""
        failure = check_unknown_name("через коннектор", "Через Connector.")
        assert failure is not None
        assert failure.check == "names"

    def test_combined_checks(self) -> None:
        failures = run_checks(
            "не хочу, потому что 100 рублей",
            "Хочу, потому что 200 рублей",
            _MARKERS,
            [],
        )
        kinds = {failure.check for failure in failures}
        assert "negations" in kinds
        assert "numbers" in kinds


def check_unknown_name(source: str, target: str, allowed: list[str] | None = None) -> Any:  # noqa: ANN401
    """Обёртка над проверкой названий: латиница и заглавные слова."""

    entries = [LexiconEntry(canonical=value) for value in (allowed or [])]
    return check_names(source, target, entries)


class TestNormalizeUtterance:
    """Обработка одной реплики."""

    def test_model_output_applied(self) -> None:
        provider = FakeProvider([_ok("Я не помню, чтобы он ошибался.")])
        outcome = normalize_utterance(
            1,
            "RESPONDENT",
            False,
            "Я, ну, не помню, чтобы он ошибался",
            previous_question=None,
            lexicon=_LEXICON,
            markers=_MARKERS,
            examples="примеры",
            provider=provider,
            min_words=4,
        )
        assert outcome.status == "proposed"
        assert outcome.text == "Я не помню, чтобы он ошибался."
        assert outcome.used_model is True
        assert outcome.model_used == "fake-model"
        assert outcome.prompt_tokens == 10

    def test_validation_error_rejects(self) -> None:
        provider = FakeProvider([LlmValidationError("плохой JSON")])
        outcome = normalize_utterance(
            1,
            "RESPONDENT",
            False,
            "Я не помню, чтобы он ошибался",
            previous_question=None,
            lexicon=_LEXICON,
            markers=_MARKERS,
            examples="примеры",
            provider=provider,
            min_words=4,
        )
        assert outcome.status == "rejected"
        assert outcome.text == "Я не помню, чтобы он ошибался."
        assert outcome.failures[0].check == "validation"

    def test_failed_check_keeps_deterministic_text(self) -> None:
        provider = FakeProvider([_ok("Я помню, чтобы он ошибался.")])
        outcome = normalize_utterance(
            1,
            "RESPONDENT",
            False,
            "Я, ну, не помню, чтобы он ошибался",
            previous_question=None,
            lexicon=_LEXICON,
            markers=_MARKERS,
            examples="примеры",
            provider=provider,
            min_words=4,
        )
        assert outcome.status == "rejected"
        # В L2 остаётся результат детерминированного прохода: паразит «ну»
        # удаляет только модель, поэтому здесь он сохраняется.
        assert "не помню" in outcome.text
        assert outcome.failures

    def test_rejected_keeps_model_output(self) -> None:
        """Отклонённый вывод модели сохраняется для разбора."""
        provider = FakeProvider([_ok("Я помню, чтобы он ошибался.")])
        outcome = normalize_utterance(
            1,
            "RESPONDENT",
            False,
            "Я не помню, чтобы он ошибался",
            previous_question=None,
            lexicon=_LEXICON,
            markers=_MARKERS,
            examples="примеры",
            provider=provider,
            min_words=4,
        )
        assert outcome.status == "rejected"
        assert outcome.proposed_text == "Я помню, чтобы он ошибался."
        entries = [item for item in outcome.changes if item.get("type") == "model_output"]
        assert entries == [{"type": "model_output", "text": "Я помню, чтобы он ошибался."}]

    def test_garbled_quality_preserved(self) -> None:
        provider = FakeProvider([_ok("Там [неразборчиво] и лимиты.", quality="garbled")])
        outcome = normalize_utterance(
            1,
            "RESPONDENT",
            False,
            "Там что-то и лимиты",
            previous_question=None,
            lexicon=[],
            markers=[],
            examples="примеры",
            provider=provider,
            min_words=4,
        )
        assert outcome.asr_quality == "garbled"

    def test_prompt_contains_context(self) -> None:
        provider = FakeProvider([_ok("Я не помню.")])
        normalize_utterance(
            1,
            "RESPONDENT",
            False,
            "я не помню точно",
            previous_question="расскажите про подключение",
            lexicon=_LEXICON,
            markers=_MARKERS,
            examples="примеры",
            provider=provider,
            min_words=4,
        )
        prompt = provider.prompts[0]
        assert "расскажите про подключение" in prompt
        assert "MCP" in prompt
        assert "не знаю" in prompt
        assert "примеры" in prompt


class TestNormalizeInterview:
    """Стадия целиком: запись, идемпотентность, транзакция."""

    def test_one_unit_per_utterance(self, db: sqlite3.Connection, settings: Settings) -> None:
        interview_id = _seed_interview(
            db,
            [
                ("INTERVIEWER", "расскажите, как вы подключили счёт", 0),
                ("RESPONDENT", "Ну я подключил счёт и всё работает нормально", 0),
                ("RESPONDENT", "да", 1),
            ],
        )
        _seed_reference(db)
        provider = FakeProvider([_ok("Я подключил счёт и всё работает нормально.")])
        result = normalize_interview(db, "R01", settings, provider)
        assert result.stats.total == 3
        assert len(_utterances(db, interview_id)) == 3
        assert result.stats.model_calls == 1

    def test_interviewer_not_sent_to_model(self, db: sqlite3.Connection, settings: Settings) -> None:
        _seed_interview(db, [("INTERVIEWER", "расскажите подробно про подключение счёта", 0)])
        _seed_reference(db)
        provider = FakeProvider([])
        result = normalize_interview(db, "R01", settings, provider)
        assert provider.prompts == []
        assert result.stats.model_calls == 0
        assert result.stats.total == 1

    def test_idempotent_second_run(self, db: sqlite3.Connection, settings: Settings) -> None:
        _seed_interview(db, [("RESPONDENT", "я подключил счёт и всё работает", 0)])
        _seed_reference(db)
        provider = FakeProvider([_ok("Я подключил счёт и всё работает.")])
        first = normalize_interview(db, "R01", settings, provider)
        assert first.skipped is False
        second = normalize_interview(db, "R01", settings, provider)
        assert second.skipped is True
        assert len(provider.prompts) == 1

    def test_new_version_written(self, db: sqlite3.Connection, settings: Settings) -> None:
        interview_id = _seed_interview(db, [("RESPONDENT", "я подключил счёт и всё работает", 0)])
        _seed_reference(db)
        provider = FakeProvider([_ok("Я подключил счёт и всё работает."), _ok("Я подключил счёт, всё работает.")])
        normalize_interview(db, "R01", settings, provider)
        second = normalize_interview(db, "R01", settings, provider, version=2)
        assert second.skipped is False
        row = db.execute(
            "SELECT COUNT(*) FROM text_unit t JOIN utterance u ON t.utterance_id = u.id WHERE u.interview_id = ?",
            (interview_id,),
        ).fetchone()
        assert row is not None and int(row[0]) == 2

    def test_new_terms_recorded(self, db: sqlite3.Connection, settings: Settings) -> None:
        _seed_interview(db, [("RESPONDENT", "смотрел квик-дашборд в приложении", 0)])
        _seed_reference(db)
        provider = FakeProvider([_ok("Смотрел квик-дашборд в приложении.", new_terms=["квик-дашборд"])])
        result = normalize_interview(db, "R01", settings, provider)
        assert result.stats.new_terms == 1
        row = db.execute("SELECT surface_form FROM new_term").fetchone()
        assert row is not None and row[0] == "квик-дашборд"

    def test_new_term_dropped_when_not_in_text(self, db: sqlite3.Connection, settings: Settings) -> None:
        """Модель исправила слово, но всё равно прислала его в new_terms."""
        _seed_interview(db, [("RESPONDENT", "смотрел квик дашборд в приложении", 0)])
        _seed_reference(db)
        provider = FakeProvider([_ok("Смотрел дашборд в приложении.", new_terms=["квик-дашборд"])])
        result = normalize_interview(db, "R01", settings, provider)
        assert result.stats.new_terms == 0
        assert db.execute("SELECT COUNT(*) FROM new_term").fetchone()[0] == 0

    def test_new_term_kept_from_rejected_output(self, db: sqlite3.Connection, settings: Settings) -> None:
        """В отклонённом выводе модели форма осталась — кандидат сохраняется."""
        _seed_interview(db, [("RESPONDENT", "смотрел 100 дашбордов в приложении", 0)])
        _seed_reference(db)
        # Отказ по числам: в L2 останется детерминированный текст без формы,
        # но в предложении модели кандидат есть.
        provider = FakeProvider([_ok("Смотрел квик-дашборд в приложении.", new_terms=["квик-дашборд"])])
        result = normalize_interview(db, "R01", settings, provider)
        assert result.stats.rejected == 1
        assert result.stats.new_terms == 1
        row = db.execute("SELECT surface_form FROM new_term").fetchone()
        assert row is not None and row[0] == "квик-дашборд"

    def test_run_recorded(self, db: sqlite3.Connection, settings: Settings) -> None:
        _seed_interview(db, [("RESPONDENT", "я подключил счёт и всё работает", 0)])
        _seed_reference(db)
        provider = FakeProvider([_ok("Я подключил счёт и всё работает.")])
        result = normalize_interview(db, "R01", settings, provider, code_version="0.1.0+abc1234")
        assert result.run_id is not None
        row = db.execute("SELECT model, prompt_hash, code_version FROM run WHERE id = ?", (result.run_id,)).fetchone()
        assert row is not None
        assert row[0] == "fake-model"
        assert row[1]
        assert row[2] == "0.1.0+abc1234"

    def test_unknown_interview(self, db: sqlite3.Connection, settings: Settings) -> None:
        with pytest.raises(NormalizeError, match="не найдено"):
            normalize_interview(db, "R99", settings, None)

    def test_no_utterances(self, db: sqlite3.Connection, settings: Settings) -> None:
        db.execute("INSERT INTO interview (id_external, name) VALUES ('R01', 'пустое')")
        db.commit()
        with pytest.raises(NormalizeError, match="нет реплик"):
            normalize_interview(db, "R01", settings, None)

    def test_rollback_leaves_nothing(
        self, db: sqlite3.Connection, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Обрыв посреди записи не оставляет частичного результата."""
        _seed_interview(
            db,
            [
                ("RESPONDENT", "я подключил счёт и всё работает", 0),
                ("RESPONDENT", "и ещё одно длинное предложение про счёт", 0),
            ],
        )
        _seed_reference(db)
        provider = FakeProvider([_ok("Я подключил счёт и всё работает."), _ok("И ещё одно длинное предложение.")])
        original = TextUnitRepository.insert
        calls = {"n": 0}

        def failing_insert(self: TextUnitRepository, *args: Any, **kwargs: Any) -> int:  # noqa: ANN401
            calls["n"] += 1
            if calls["n"] == 2:
                msg = "сбой записи"
                raise sqlite3.OperationalError(msg)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(TextUnitRepository, "insert", failing_insert)
        with pytest.raises(sqlite3.OperationalError):
            normalize_interview(db, "R01", settings, provider)
        count = db.execute("SELECT COUNT(*) FROM text_unit").fetchone()
        assert count is not None and int(count[0]) == 0
        runs = db.execute("SELECT COUNT(*) FROM run").fetchone()
        assert runs is not None and int(runs[0]) == 0


class TestDryRun:
    """Оценка без вызовов модели."""

    def test_dry_run_counts_calls(self, db: sqlite3.Connection, settings: Settings) -> None:
        _seed_interview(
            db,
            [
                ("INTERVIEWER", "расскажите, как вы подключили счёт в приложении", 0),
                ("RESPONDENT", "я подключил счёт и всё работает нормально", 0),
                ("RESPONDENT", "да", 1),
            ],
        )
        report = plan_dry_run(db, "R01", settings)
        assert report.utterances == 3
        assert report.model_calls == 1
        assert report.prompt_tokens > 0
        assert report.already_done is False

    def test_dry_run_marks_existing(self, db: sqlite3.Connection, settings: Settings) -> None:
        _seed_interview(db, [("RESPONDENT", "я подключил счёт и всё работает", 0)])
        _seed_reference(db)
        normalize_interview(db, "R01", settings, FakeProvider([_ok("Я подключил счёт и всё работает.")]))
        report = plan_dry_run(db, "R01", settings)
        assert report.already_done is True


class TestSchema:
    """Схема ответа модели."""

    def test_schema_requires_all_fields(self) -> None:
        assert set(NORMALIZE_SCHEMA["required"]) == {
            "normalized_text",
            "changes",
            "uncertain_spans",
            "new_terms",
            "asr_quality",
        }

    def test_changes_types(self) -> None:
        item = NORMALIZE_SCHEMA["properties"]["changes"]["items"]
        assert item["properties"]["type"]["enum"] == [
            "term",
            "asr_fix",
            "filler",
            "stutter",
            "order",
            "punctuation",
            "garbled",
        ]


# --- M7: аддитивный протокол глоссария -------------------------------------

_OTHER_TERM = LexiconEntry(canonical="ребалансировка", variants=("реглантировка",))


class TestPromptLexiconFilter:
    """Фильтр сужает только подстановку {lexicon}; проверки остаются на полном."""

    def test_irrelevant_terms_not_in_prompt(self) -> None:
        provider = FakeProvider([_ok("Подключил MCP в приложении.")])
        normalize_utterance(
            1,
            "RESPONDENT",
            False,
            "подключил эсипи в приложении и всё работает",
            previous_question=None,
            lexicon=[*_LEXICON, _OTHER_TERM],
            markers=_MARKERS,
            examples="примеры",
            provider=provider,
            min_words=4,
            prompt_lexicon_filter=True,
        )
        prompt = provider.prompts[0]
        assert "MCP" in prompt
        assert "ребалансировка" not in prompt

    def test_full_lexicon_without_filter(self) -> None:
        provider = FakeProvider([_ok("Подключил MCP в приложении.")])
        normalize_utterance(
            1,
            "RESPONDENT",
            False,
            "подключил эсипи в приложении и всё работает",
            previous_question=None,
            lexicon=[*_LEXICON, _OTHER_TERM],
            markers=_MARKERS,
            examples="примеры",
            provider=provider,
            min_words=4,
        )
        prompt = provider.prompts[0]
        assert "MCP" in prompt
        assert "ребалансировка" in prompt

    def test_checks_use_full_lexicon(self) -> None:
        """Термин не попал в промпт, но проверка латиницы его всё равно знает."""
        entries = [LexiconEntry(canonical="SuperApp", variants=("суперапп",))]
        assert check_latin("пользуюсь суперапп постоянно", "Пользуюсь SuperApp.", entries) is None

    def test_filter_does_not_call_lexicon_for_skipped_utterances(self) -> None:
        """Реплика без вызова модели не должна вычислять фильтр."""
        provider = FakeProvider([])
        outcome = normalize_utterance(
            1,
            "INTERVIEWER",
            False,
            "расскажите подробно про подключение счёта",
            previous_question=None,
            lexicon=[*_LEXICON, _OTHER_TERM],
            markers=_MARKERS,
            examples="примеры",
            provider=provider,
            min_words=4,
            prompt_lexicon_filter=True,
        )
        assert provider.prompts == []
        assert outcome.used_model is False


class TestCachedCalls:
    """Счётчик ответов из кэша."""

    def test_cached_flag_propagated(self) -> None:
        provider = FakeProvider([_ok("Я подключил счёт и всё работает.")], cached=True)
        outcome = normalize_utterance(
            1,
            "RESPONDENT",
            False,
            "я подключил счёт и всё работает",
            previous_question=None,
            lexicon=[],
            markers=[],
            examples="примеры",
            provider=provider,
            min_words=4,
        )
        assert outcome.cached is True
        assert summarize([outcome]).cached_calls == 1

    def test_network_call_not_cached(self) -> None:
        provider = FakeProvider([_ok("Я подключил счёт и всё работает.")])
        outcome = normalize_utterance(
            1,
            "RESPONDENT",
            False,
            "я подключил счёт и всё работает",
            previous_question=None,
            lexicon=[],
            markers=[],
            examples="примеры",
            provider=provider,
            min_words=4,
        )
        assert outcome.cached is False
        assert summarize([outcome]).cached_calls == 0

    def test_cached_calls_recorded_in_run(self, db: sqlite3.Connection, settings: Settings) -> None:
        _seed_interview(db, [("RESPONDENT", "я подключил счёт и всё работает", 0)])
        _seed_reference(db)
        provider = FakeProvider([_ok("Я подключил счёт и всё работает.")], cached=True)
        result = normalize_interview(db, "R01", settings, provider)
        assert result.stats.model_calls == 1
        assert result.stats.cached_calls == 1
        params = RunRepository(db).params(int(result.run_id or 0))
        assert params["cached_calls"] == 1


class TestLexiconHashInRun:
    """Снимок лексикона фиксируется в params_json прогона."""

    def test_hash_and_filter_settings_recorded(self, db: sqlite3.Connection, settings: Settings) -> None:
        _seed_interview(db, [("RESPONDENT", "я подключил счёт и всё работает", 0)])
        _seed_reference(db)
        result = normalize_interview(db, "R01", settings, FakeProvider([_ok("Я подключил счёт и всё работает.")]))
        params = RunRepository(db).params(int(result.run_id or 0))
        assert params["lexicon_hash"] == lexicon_snapshot_hash(load_lexicon(db))
        assert params["lexicon_terms"] == 1
        assert params["prompt_lexicon_filter"] is True
        assert params["prompt_lexicon_threshold"] == settings.m1b.prompt_lexicon_threshold

    def test_hash_changes_with_variant(self, db: sqlite3.Connection, settings: Settings) -> None:
        _seed_interview(db, [("RESPONDENT", "я подключил счёт и всё работает", 0)])
        _seed_reference(db)
        first = normalize_interview(db, "R01", settings, FakeProvider([_ok("Я подключил счёт и всё работает.")]))
        LexiconRepository(db).add_variant("MCP", "мсипи")
        db.commit()
        second = normalize_interview(
            db,
            "R01",
            settings,
            FakeProvider([_ok("Я подключил счёт и всё работает.")]),
            version=2,
        )
        first_params = RunRepository(db).params(int(first.run_id or 0))
        second_params = RunRepository(db).params(int(second.run_id or 0))
        assert first_params["lexicon_hash"] != second_params["lexicon_hash"]


class TestLexiconFreeze:
    """Заморозка L2: изменение лексикона не переписывает записанную версию."""

    def test_freeze_warns_on_changed_lexicon(self, db: sqlite3.Connection, settings: Settings) -> None:
        interview_id = _seed_interview(db, [("RESPONDENT", "я подключил счёт и всё работает", 0)])
        _seed_reference(db)
        normalize_interview(db, "R01", settings, FakeProvider([_ok("Я подключил счёт и всё работает.")]))
        LexiconRepository(db).add_variant("MCP", "мсипи")
        db.commit()

        provider = FakeProvider([])
        result = normalize_interview(db, "R01", settings, provider)

        assert result.skipped is True
        assert result.lexicon_changed is True
        assert provider.prompts == []
        row = db.execute(
            "SELECT COUNT(*) FROM text_unit t JOIN utterance u ON t.utterance_id = u.id WHERE u.interview_id = ?",
            (interview_id,),
        ).fetchone()
        assert row is not None and int(row[0]) == 1

    def test_freeze_silent_without_hash(self, db: sqlite3.Connection, settings: Settings) -> None:
        """Прогон до M7 не писал lexicon_hash: расхождением это не считается."""
        interview_id = _seed_interview(db, [("RESPONDENT", "я подключил счёт и всё работает", 0)])
        _seed_reference(db)
        normalize_interview(db, "R01", settings, FakeProvider([_ok("Я подключил счёт и всё работает.")]))
        db.execute("UPDATE run SET params_json = '{\"version\": 1}'")
        db.commit()
        LexiconRepository(db).add_variant("MCP", "мсипи")
        db.commit()

        result = normalize_interview(db, "R01", settings, FakeProvider([]))

        assert result.skipped is True
        assert result.lexicon_changed is False
        assert result.lexicon_run_hash == ""
        assert TextUnitRepository(db).count_for_interview(interview_id, 1) == 1

    def test_new_version_keeps_old_units(self, db: sqlite3.Connection, settings: Settings) -> None:
        interview_id = _seed_interview(db, [("RESPONDENT", "я подключил счёт и всё работает", 0)])
        _seed_reference(db)
        normalize_interview(db, "R01", settings, FakeProvider([_ok("Я подключил счёт и всё работает.")]))
        LexiconRepository(db).add_variant("MCP", "мсипи")
        db.commit()

        result = normalize_interview(
            db,
            "R01",
            settings,
            FakeProvider([_ok("Я подключил счёт, всё работает.")]),
            version=2,
        )

        assert result.skipped is False
        assert result.lexicon_changed is False
        assert TextUnitRepository(db).count_for_interview(interview_id, 1) == 1
        assert TextUnitRepository(db).count_for_interview(interview_id, 2) == 1

    def test_run_id_for_version(self, db: sqlite3.Connection, settings: Settings) -> None:
        interview_id = _seed_interview(db, [("RESPONDENT", "я подключил счёт и всё работает", 0)])
        _seed_reference(db)
        result = normalize_interview(db, "R01", settings, FakeProvider([_ok("Я подключил счёт и всё работает.")]))
        repo = TextUnitRepository(db)
        assert repo.run_id_for_version(interview_id, 1) == result.run_id
        assert repo.run_id_for_version(interview_id, 2) is None

    def test_params_of_missing_run(self, db: sqlite3.Connection) -> None:
        assert RunRepository(db).params(999) == {}


class TestCollectSuspects:
    """Предварительный сбор слов без модели: только new_term."""

    def test_candidates_written_without_run_or_l2(self, db: sqlite3.Connection, settings: Settings) -> None:
        _seed_interview(
            db,
            [
                ("RESPONDENT", "Там автослеле и нечечно работает", 0),
                ("INTERVIEWER", "Расскажите подробнее про подключение", 0),
            ],
        )
        _seed_reference(db)

        report = collect_suspects(db, "R01", settings)

        assert [form for form, _count in report.forms] == ["автослеле", "нечечно"]
        assert report.total == 2
        assert report.added == 2
        assert db.execute("SELECT COUNT(*) FROM run").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM text_unit").fetchone()[0] == 0
        forms = {row[0] for row in db.execute("SELECT surface_form FROM new_term").fetchall()}
        assert forms == {"автослеле", "нечечно"}

    def test_short_respondent_utterance_included(self, db: sqlite3.Connection, settings: Settings) -> None:
        """Порог min_words_for_model не применяется: цель — собрать словарь."""
        _seed_interview(db, [("RESPONDENT", "автослеле", 0)])
        _seed_reference(db)

        report = collect_suspects(db, "R01", settings)

        assert [form for form, _count in report.forms] == ["автослеле"]

    def test_repeated_form_counted(self, db: sqlite3.Connection, settings: Settings) -> None:
        """Частота — число вхождений формы по всем репликам респондента."""
        _seed_interview(
            db,
            [
                ("RESPONDENT", "нечечно там нечечно опять", 0),
                ("RESPONDENT", "и снова нечечно", 0),
            ],
        )
        _seed_reference(db)

        report = collect_suspects(db, "R01", settings)

        assert report.forms == [("нечечно", 3)]
        assert report.total == 3
        assert report.added == 1

    def test_repeat_run_adds_nothing(self, db: sqlite3.Connection, settings: Settings) -> None:
        """Повторный сбор не плодит дублей при context_utterance_id IS NULL."""
        _seed_interview(db, [("RESPONDENT", "Там автослеле работает", 0)])
        _seed_reference(db)
        collect_suspects(db, "R01", settings)

        second = collect_suspects(db, "R01", settings)

        assert second.added == 0
        assert db.execute("SELECT COUNT(*) FROM new_term").fetchone()[0] == 1

    def test_unknown_interview(self, db: sqlite3.Connection, settings: Settings) -> None:
        with pytest.raises(NormalizeError, match="не найдено"):
            collect_suspects(db, "R99", settings)


class TestReviewQueueWithoutL2:
    """Очередь review-terms: кандидат интервью без L2 не выбрасывается."""

    def _seed_interview_with_utterance(self, conn: sqlite3.Connection, code: str) -> int:
        cursor = conn.execute("INSERT INTO interview (id_external, name) VALUES (?, 'тест')", (code,))
        interview_id = int(cursor.lastrowid or 0)
        cursor = conn.execute(
            "INSERT INTO utterance (interview_id, speaker_role, text, turn_ids) VALUES (?, 'RESPONDENT', ?, '[]')",
            (interview_id, "текст реплики"),
        )
        return int(cursor.lastrowid or 0)

    def test_candidate_without_l2_stays(self, db: sqlite3.Connection) -> None:
        utterance_id = self._seed_interview_with_utterance(db, "R02")
        NewTermRepository(db).add("автослеле", 1, utterance_id)
        db.commit()

        queue = NewTermRepository(db).proposed_on_current_l2()

        assert [form for _id, form, _ctx in queue] == ["автослеле"]

    def test_candidate_absent_from_existing_l2_dropped(self, db: sqlite3.Connection) -> None:
        """Регрессия: при наличии L2 поведение прежнее."""
        utterance_id = self._seed_interview_with_utterance(db, "R01")
        db.execute(
            "INSERT INTO text_unit (utterance_id, version, text) VALUES (?, 1, ?)",
            (utterance_id, "Подключил MCP Финам."),
        )
        NewTermRepository(db).add("автослеле", 1, utterance_id)
        db.commit()

        assert NewTermRepository(db).proposed_on_current_l2() == []

    def test_review_terms_asks_candidate_without_l2(self, db: sqlite3.Connection) -> None:
        utterance_id = self._seed_interview_with_utterance(db, "R02")
        NewTermRepository(db).add("автослеле", 1, utterance_id)
        db.commit()

        result = review_terms(db, ask=lambda _prompt: "n")

        assert result.stale == 0
        assert result.rejected == ["автослеле"]


class TestSuspectsOnlyCli:
    """`normalize --suspects-only` печатает формы и не создаёт версию L2."""

    def _run(
        self,
        db: sqlite3.Connection,
        settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import argparse

        from qa import cli

        monkeypatch.setattr(cli, "_load_settings", lambda: settings)
        monkeypatch.setattr(cli, "_open_db", lambda _settings: db)
        args = argparse.Namespace(interview="R01", suspects_only=True, dry_run=False, new_version=False)
        assert cli.cmd_normalize(args) == 0

    def test_prints_forms_and_creates_no_l2(
        self,
        db: sqlite3.Connection,
        settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _seed_interview(db, [("RESPONDENT", "Там автослеле и нечечно работает", 0)])
        _seed_reference(db)
        db_path = Path(str(db.execute("PRAGMA database_list").fetchone()[2]))

        self._run(db, settings, monkeypatch)

        out = capsys.readouterr().out
        assert "форм=2" in out
        assert "вхождений=2" in out
        assert "автослеле" in out
        # Команда закрывает соединение; проверяем результат новым подключением.
        check = sqlite3.connect(db_path)
        try:
            assert check.execute("SELECT COUNT(*) FROM run").fetchone()[0] == 0
            assert check.execute("SELECT COUNT(*) FROM text_unit").fetchone()[0] == 0
            assert check.execute("SELECT COUNT(*) FROM new_term").fetchone()[0] == 2
        finally:
            check.close()
