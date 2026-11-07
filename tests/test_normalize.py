"""Тесты M1b: детерминированный проход, проход модели, автопроверки, запись L2.

Все тексты синтетические. Модель заменена заглушкой, сети нет.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from qa.db.connection import get_connection
from qa.db.migrations import migrate
from qa.db.workbook_repo import LexiconRepository, MarkerRepository, TextUnitRepository
from qa.llm.base import LlmProvider, LlmResult
from qa.llm.errors import LlmValidationError
from qa.pipeline.normalize import NormalizeError, normalize_interview, plan_dry_run
from qa.pipeline.normalize_l2 import (
    NORMALIZE_SCHEMA,
    check_names,
    check_negations,
    check_numbers,
    check_uncertainty,
    normalize_utterance,
    run_checks,
    should_use_model,
)
from qa.pipeline.normalize_rules import LexiconEntry, normalize_deterministic
from qa.settings import Settings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations"

_MARKERS = ["не знаю", "не помню", "наверное", "возможно", "вроде"]
_LEXICON = [LexiconEntry(canonical="MCP", variants=("эсипи", "эс-и-пи"))]


class FakeProvider(LlmProvider):
    """Заглушка модели: отдаёт заданный ответ и считает вызовы."""

    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.model_used = "fake-model"
        self.prompt_tokens = 10
        self.completion_tokens = 5

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


class TestModelSelection:
    """Кто попадает в модель."""

    def test_interviewer_never_uses_model(self) -> None:
        assert should_use_model("INTERVIEWER", False, "длинная реплика интервьюера из многих слов", 4) is False

    def test_backchannel_never_uses_model(self) -> None:
        assert should_use_model("RESPONDENT", True, "длинная реплика респондента из многих слов", 4) is False

    def test_short_respondent_skipped(self) -> None:
        assert should_use_model("RESPONDENT", False, "да хорошо", 4) is False

    def test_long_respondent_used(self) -> None:
        assert should_use_model("RESPONDENT", False, "я подключил счёт и всё работает", 4) is True


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
            "filler",
            "stutter",
            "order",
            "punctuation",
            "garbled",
        ]
