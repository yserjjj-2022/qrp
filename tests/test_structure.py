"""Тесты M1c: привязка реплик к вопросам гайда и фрагменты L2.

Все тексты синтетические. Модель заменяется заглушкой, сети нет.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from qa.db.connection import get_connection
from qa.db.migrations import migrate
from qa.db.workbook_repo import StructureRepository, UtteranceWithL2
from qa.llm.base import LlmProvider, LlmResult
from qa.llm.errors import LlmError
from qa.pipeline.structure import (
    StructureError,
    _block_categories,
    build_blocks,
    build_segment_prompt,
    fallback_spans,
    format_report,
    lexical_similarity,
    structure_interview,
)
from qa.settings import Settings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations"

_GUIDE = [
    ("1", "Как давно вы инвестируете?"),
    ("2", "Пользуетесь ли вы ИИ-помощниками?"),
    ("3", "Что может пойти не так, когда ИИ имеет доступ к счёту?"),
]

# (роль, текст, is_backchannel)
_TRANSCRIPT = [
    ("INTERVIEWER", "Здравствуйте. Как давно вы инвестируете?", 0),
    ("RESPONDENT", "Года три уже. Начинал с облигаций, потом добавил акции.", 0),
    ("INTERVIEWER", "Угу.", 1),
    ("INTERVIEWER", "Пользуетесь ли вы ИИ-помощниками в инвестировании?", 0),
    ("RESPONDENT", "Пробовал один раз. Не понравилось, больше не открываю.", 0),
    ("INTERVIEWER", "Спасибо, на этом всё.", 0),
]

_BINDINGS = {
    "blocks": [
        {"block": 1, "question": "1", "is_probe": False},
        {"block": 2, "question": "2", "is_probe": False},
        {"block": 3, "question": None, "is_probe": False},
    ]
}


class FakeProvider(LlmProvider):
    """Заглушка модели: отдаёт заданные ответы по очереди, затем ответ по умолчанию.

    Исчерпанные ответы дают `LlmError`: код трактует это как «модель не ответила»
    и переходит к детерминированному делению, а не как отказ проверки. Тест
    привязки не обязан описывать ответ на каждую реплику.
    """

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        return self.complete_json_detailed(prompt, schema).data

    def complete_json_detailed(self, prompt: str, schema: dict[str, Any]) -> LlmResult:  # noqa: ARG002
        self.calls += 1
        if not self._responses:
            msg = "заглушка исчерпала ответы"
            raise LlmError(msg)
        return LlmResult(data=self._responses.pop(0), model_used="fake-model")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Нулевые векторы: привязка уходит на лексическое сходство."""
        return [[0.0] * 4 for _ in texts]


def _settings() -> Settings:
    """Настройки проекта из config.yaml (без фикстуры: вызывается напрямую)."""
    return Settings.load(_PROJECT_ROOT / "config.yaml")


@pytest.fixture()
def db(tmp_path: Path) -> sqlite3.Connection:
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    migrate(conn, db_path, _MIGRATIONS_DIR, tmp_path / "backups")
    yield conn
    conn.close()


def _seed_interview(conn: sqlite3.Connection, code: str = "R01") -> int:
    """Интервью и реплики; turn_ids заполняются реальными id записей."""
    cursor = conn.execute("INSERT INTO interview (id_external, name) VALUES (?, 'тестовое интервью')", (code,))
    interview_id = int(cursor.lastrowid or 0)
    for index, (role, text, flag) in enumerate(_TRANSCRIPT, start=1):
        cursor = conn.execute(
            "INSERT INTO turn (interview_id, line_no, start_sec, speaker_role, text) VALUES (?, ?, ?, ?, ?)",
            (interview_id, index, index * 10, role, text),
        )
        turn_id = int(cursor.lastrowid or 0)
        conn.execute(
            "INSERT INTO utterance (interview_id, speaker_role, text, turn_ids, is_backchannel) VALUES (?, ?, ?, ?, ?)",
            (interview_id, role, text, f"[{turn_id}]", flag),
        )
    conn.commit()
    return interview_id


def _seed_guide(conn: sqlite3.Connection) -> dict[str, int]:
    """Версия кодбука и вопросы гайда; карта «идентификатор -> id»."""
    cursor = conn.execute("INSERT INTO codebook_version (file_hash) VALUES ('h1')")
    version_id = int(cursor.lastrowid or 0)
    ids: dict[str, int] = {}
    for identifier, wording in _GUIDE:
        cursor = conn.execute(
            'INSERT INTO question (codebook_version_id, identifier, "group", wording) VALUES (?, ?, ?, ?)',
            (version_id, identifier, "Все", wording),
        )
        ids[identifier] = int(cursor.lastrowid or 0)
    conn.commit()
    return ids


def _seed_l2(conn: sqlite3.Connection, interview_id: int, version: int = 1, text: str | None = None) -> None:
    """Записать L2 версии для всех реплик интервью."""
    rows = conn.execute("SELECT id, text FROM utterance WHERE interview_id = ? ORDER BY id", (interview_id,)).fetchall()
    for utterance_id, source in rows:
        conn.execute(
            "INSERT INTO text_unit (utterance_id, version, text) VALUES (?, ?, ?)",
            (int(utterance_id), version, text if text is not None else str(source)),
        )
    conn.commit()


@pytest.fixture()
def seeded(db: sqlite3.Connection) -> tuple[sqlite3.Connection, int, dict[str, int]]:
    """Интервью с репликами, гайдом и L2 версии 1."""
    interview_id = _seed_interview(db)
    ids = _seed_guide(db)
    _seed_l2(db, interview_id)
    return db, interview_id, ids


def _l2_text(conn: sqlite3.Connection, utterance_id: int, version: int = 1) -> str:
    row = conn.execute(
        "SELECT text FROM text_unit WHERE utterance_id = ? AND version = ?", (utterance_id, version)
    ).fetchone()
    return "" if row is None else str(row[0])


def _fragments_match_l2(conn: sqlite3.Connection, interview_id: int) -> bool:
    """Каждый фрагмент равен срезу своего text_unit по start:end."""
    rows = conn.execute(
        'SELECT f.text, f."start", f."end", t.text FROM fragment f'
        " JOIN utterance u ON f.utterance_id = u.id"
        " JOIN text_unit t ON f.text_unit_id = t.id"
        " WHERE u.interview_id = ?",
        (interview_id,),
    ).fetchall()
    if not rows:
        return False
    return all(str(text) == str(l2)[int(start) : int(end)] for text, start, end, l2 in rows)


def rows_question(conn: sqlite3.Connection, utterance_id: int) -> int | None:
    """question_id реплики (None — вне гайда)."""
    row = conn.execute("SELECT question_id FROM utterance WHERE id = ?", (utterance_id,)).fetchone()
    return None if row is None or row[0] is None else int(row[0])


class TestMigration0006:
    """Схема fragment и utterance после миграции 0006."""

    def test_fragment_columns(self, db: sqlite3.Connection) -> None:
        columns = {row[1] for row in db.execute("PRAGMA table_info(fragment)").fetchall()}
        assert {
            "utterance_id",
            "text_unit_id",
            "start",
            "end",
            "text",
            "question_id",
            "prior_interviewer_turn_id",
            "asr_quality",
        } <= columns
        assert "turn_id" not in columns
        assert "start_pos" not in columns

    def test_utterance_columns(self, db: sqlite3.Connection) -> None:
        columns = {row[1] for row in db.execute("PRAGMA table_info(utterance)").fetchall()}
        assert {"question_id", "is_probe", "prior_interviewer_turn_id"} <= columns

    def test_fragment_rejects_reversed_span(self, seeded: tuple[sqlite3.Connection, int, dict[str, int]]) -> None:
        conn, interview_id, _ids = seeded
        utterance_id, text_unit_id = conn.execute(
            "SELECT u.id, t.id FROM utterance u"
            " JOIN text_unit t ON t.utterance_id = u.id"
            " WHERE u.interview_id = ? ORDER BY u.id LIMIT 1",
            (interview_id,),
        ).fetchone()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                'INSERT INTO fragment (utterance_id, text_unit_id, "start", "end", text) VALUES (?, ?, 5, 2, ?)',
                (int(utterance_id), int(text_unit_id), "текст"),
            )


class TestSpanValidation:
    """Границы фрагментов и запасное деление кодом."""

    def test_fallback_covers_text(self) -> None:
        text = "Первая мысль. Вторая мысль. Третья."
        spans = fallback_spans(text, target_chars=10)
        assert spans[0].start == 0
        assert spans[-1].end == len(text)
        assert "".join(span.text_of(text) for span in spans) == text

    def test_fallback_single_sentence(self) -> None:
        text = "Одна мысль без точки"
        spans = fallback_spans(text)
        assert len(spans) == 1
        assert spans[0].text_of(text) == text

    def test_fallback_respects_limit(self) -> None:
        text = " ".join(["слово"] * 60)
        spans = fallback_spans(text, target_chars=50)
        assert spans[0].start == 0
        assert spans[-1].end == len(text)
        assert all(span.end - span.start <= 50 for span in spans)
        assert "".join(span.text_of(text) for span in spans) == text

    def test_lexical_similarity_orders_questions(self) -> None:
        high = lexical_similarity("Как давно вы инвестируете?", _GUIDE[0][1])
        low = lexical_similarity("Как давно вы инвестируете?", _GUIDE[2][1])
        assert high > low


class TestSegmentValidation:
    """Границы модели принимаются только при полном покрытии текста."""

    def test_rejects_unknown_chunk(self, seeded: tuple[sqlite3.Connection, int, dict[str, int]]) -> None:
        """Куска нет в тексте — ответ отклонён целиком, работает деление кодом."""
        conn, interview_id, _ids = seeded
        provider = FakeProvider([_BINDINGS, {"fragments": ["Куска нет в тексте.", "И этого тоже."]}])
        result = structure_interview(conn, "R01", _settings(), provider, code_version="test")
        assert result.stats.rejected_segments >= 1
        assert _fragments_match_l2(conn, interview_id)

    def test_rejects_reordered_chunks(self, seeded: tuple[sqlite3.Connection, int, dict[str, int]]) -> None:
        """Куски в обратном порядке не находятся последовательным поиском."""
        conn, interview_id, _ids = seeded
        respondent_id = conn.execute(
            "SELECT id FROM utterance WHERE interview_id = ? AND speaker_role = 'RESPONDENT' ORDER BY id LIMIT 1",
            (interview_id,),
        ).fetchone()[0]
        text = _l2_text(conn, int(respondent_id))
        head, tail = text[:10], text[10:]
        provider = FakeProvider([_BINDINGS, {"fragments": [tail, head]}])
        result = structure_interview(conn, "R01", _settings(), provider, code_version="test")
        assert result.stats.rejected_segments >= 1
        assert _fragments_match_l2(conn, interview_id)

    def test_accepts_text_chunks(self, seeded: tuple[sqlite3.Connection, int, dict[str, int]]) -> None:
        """Дословные куски превращаются в границы без участия модели в смещениях."""
        conn, interview_id, _ids = seeded
        respondent_id = conn.execute(
            "SELECT id FROM utterance WHERE interview_id = ? AND speaker_role = 'RESPONDENT' ORDER BY id LIMIT 1",
            (interview_id,),
        ).fetchone()[0]
        text = _l2_text(conn, int(respondent_id))
        cut = text.index(". ") + 2 if ". " in text else len(text)
        provider = FakeProvider([_BINDINGS, {"fragments": [text[:cut].strip(), text[cut:].strip()]}])
        result = structure_interview(conn, "R01", _settings(), provider, code_version="test")
        assert result.stats.rejected_segments == 0
        assert _fragments_match_l2(conn, interview_id)

    def test_fragments_do_not_intersect(self, seeded: tuple[sqlite3.Connection, int, dict[str, int]]) -> None:
        conn, interview_id, _ids = seeded
        structure_interview(conn, "R01", _settings(), None, code_version="test")
        rows = conn.execute(
            'SELECT f.utterance_id, f."start", f."end" FROM fragment f'
            " JOIN utterance u ON f.utterance_id = u.id WHERE u.interview_id = ?"
            " ORDER BY f.utterance_id, f.start",
            (interview_id,),
        ).fetchall()
        assert rows
        previous: tuple[int, int] | None = None
        for utterance_id, start, end in rows:
            if previous is not None and previous[0] == int(utterance_id):
                assert int(start) >= previous[1], "фрагменты одной реплики пересекаются"
            previous = (int(utterance_id), int(end))


class TestQuestionBinding:
    """Привязка блоков «вопрос -> ответ» к вопросам гайда."""

    def test_out_of_guide_gets_null(self, seeded: tuple[sqlite3.Connection, int, dict[str, int]]) -> None:
        conn, _interview_id, _ids = seeded
        structure_interview(conn, "R01", _settings(), FakeProvider([_BINDINGS]), code_version="test")
        rows = dict(conn.execute("SELECT id, question_id FROM utterance WHERE speaker_role = 'INTERVIEWER'").fetchall())
        assert rows[6] is None

    def test_two_questions_share_one_block(self, seeded: tuple[sqlite3.Connection, int, dict[str, int]]) -> None:
        """«Угу» и следующий вопрос — один контекст: обе реплики получают вопрос блока."""
        conn, _interview_id, ids = seeded
        structure_interview(conn, "R01", _settings(), FakeProvider([_BINDINGS]), code_version="test")
        rows = dict(conn.execute("SELECT id, question_id FROM utterance WHERE speaker_role = 'INTERVIEWER'").fetchall())
        assert rows[3] == ids["2"]
        assert rows[4] == ids["2"]

    def test_backchannel_does_not_create_block(self, seeded: tuple[sqlite3.Connection, int, dict[str, int]]) -> None:
        conn, _interview_id, _ids = seeded
        result = structure_interview(conn, "R01", _settings(), FakeProvider([_BINDINGS]), code_version="test")
        # Реплики: (вопрос), (ответ), (Угу + вопрос), (ответ), (хвост) -> три блока.
        assert result.stats.blocks == 3

    def test_respondent_inherits_question(self, seeded: tuple[sqlite3.Connection, int, dict[str, int]]) -> None:
        conn, interview_id, ids = seeded
        structure_interview(conn, "R01", _settings(), FakeProvider([_BINDINGS]), code_version="test")
        rows = conn.execute(
            "SELECT id, question_id, prior_interviewer_turn_id FROM utterance"
            " WHERE interview_id = ? AND speaker_role = 'RESPONDENT' ORDER BY id",
            (interview_id,),
        ).fetchall()
        assert [row[1] for row in rows] == [ids["1"], ids["2"]]
        # Второй ответ идёт после двух реплик интервьюера: привязка к последней.
        assert rows[1][2] == 4

    def test_probe_flag(self, seeded: tuple[sqlite3.Connection, int, dict[str, int]]) -> None:
        conn, _interview_id, ids = seeded
        blocks = {
            "blocks": [
                {"block": 1, "question": "1", "is_probe": False},
                {"block": 2, "question": "1", "is_probe": True},
                {"block": 3, "question": None, "is_probe": False},
            ]
        }
        structure_interview(conn, "R01", _settings(), FakeProvider([blocks]), code_version="test")
        flags = dict(conn.execute("SELECT id, is_probe FROM utterance").fetchall())
        assert flags[3] == 1
        assert rows_question(conn, 3) == ids["1"]

    def test_backchannel_probe_not_counted(self, seeded: tuple[sqlite3.Connection, int, dict[str, int]]) -> None:
        """Блок из «Угу» и содержательной реплики — одно уточнение, а не два."""
        conn, _interview_id, _ids = seeded
        blocks = {
            "blocks": [
                {"block": 1, "question": "1", "is_probe": False},
                {"block": 2, "question": "2", "is_probe": True},
                {"block": 3, "question": None, "is_probe": False},
            ]
        }
        result = structure_interview(conn, "R01", _settings(), FakeProvider([blocks]), code_version="test")
        # В блоке 2 две реплики интервьюера: «Угу» (is_backchannel=1) и вопрос.
        assert result.stats.probes == 1

    def test_backchannel_probe_not_counted_in_report(
        self, seeded: tuple[sqlite3.Connection, int, dict[str, int]]
    ) -> None:
        conn, _interview_id, _ids = seeded
        blocks = {
            "blocks": [
                {"block": 1, "question": "1", "is_probe": False},
                {"block": 2, "question": "2", "is_probe": True},
                {"block": 3, "question": None, "is_probe": False},
            ]
        }
        result = structure_interview(conn, "R01", _settings(), FakeProvider([blocks]), code_version="test")
        report = "\n".join(format_report(result))
        assert "уточнений=1" in report

    def test_report_splits_guide_blocks(self, seeded: tuple[sqlite3.Connection, int, dict[str, int]]) -> None:
        """Отчёт делит блоки на непересекающиеся категории, а не смешивает их."""
        conn, _interview_id, _ids = seeded
        blocks = {
            "blocks": [
                {"block": 1, "question": "1", "is_probe": False},
                {"block": 2, "question": "2", "is_probe": True},
                {"block": 3, "question": None, "is_probe": False},
            ]
        }
        result = structure_interview(conn, "R01", _settings(), FakeProvider([blocks]), code_version="test")
        report = "\n".join(format_report(result))
        assert "основных=1" in report
        assert "уточнений=1" in report
        assert "вне гайда=1" in report
        assert "без реплики интервьюера=0" in report
        assert result.stats.main_blocks + result.stats.probes + result.stats.out_of_guide == result.stats.blocks

    def test_block_without_interviewer_is_separate_category(self) -> None:
        """Ответ без реплики интервьюера не становится основным вопросом гайда."""
        utterances = [
            UtteranceWithL2(
                id=1,
                speaker_role="RESPONDENT",
                text="Отдельная мысль без вопроса.",
                is_backchannel=False,
                text_unit_id=1,
                l2_text="Отдельная мысль без вопроса.",
                asr_quality="ok",
            ),
            UtteranceWithL2(
                id=2,
                speaker_role="INTERVIEWER",
                text="Как давно вы инвестируете?",
                is_backchannel=False,
                text_unit_id=2,
                l2_text="Как давно вы инвестируете?",
                asr_quality="ok",
            ),
            UtteranceWithL2(
                id=3,
                speaker_role="RESPONDENT",
                text="Года три.",
                is_backchannel=False,
                text_unit_id=3,
                l2_text="Года три.",
                asr_quality="ok",
            ),
        ]
        blocks = build_blocks(utterances)
        # Модель привязала оба блока к вопросам гайда, но у первого нет интервьюера.
        categories = _block_categories(blocks, {1: (1, False), 2: (1, False)}, utterances)
        assert categories.main == 1
        assert categories.without_interviewer == 1
        assert categories.probes == 0
        assert categories.out_of_guide == 0
        assert categories.main + categories.without_interviewer == len(blocks)

    def test_unknown_question_identifier_ignored(self, seeded: tuple[sqlite3.Connection, int, dict[str, int]]) -> None:
        conn, _interview_id, _ids = seeded
        bindings = {
            "blocks": [
                {"block": 1, "question": "99Т", "is_probe": False},
                {"block": 2, "question": None, "is_probe": False},
                {"block": 3, "question": None, "is_probe": False},
            ]
        }
        result = structure_interview(conn, "R01", _settings(), FakeProvider([bindings]), code_version="test")
        assert result.stats.unknown_questions == 1
        assert rows_question(conn, 1) is None

    def test_no_model_uses_similarity(self, seeded: tuple[sqlite3.Connection, int, dict[str, int]]) -> None:
        conn, _interview_id, ids = seeded
        result = structure_interview(conn, "R01", _settings(), None, code_version="test")
        assert result.stats.similarity_source == "lexical"
        assert rows_question(conn, 1) == ids["1"]


class TestUnansweredQuestions:
    """Вопросы гайда без ответа: только применимые к группе респондента."""

    def _seed_grouped_guide(self, conn: sqlite3.Connection) -> None:
        """Гайд из трёх вопросов: «Все», «ТОРГ», «Все» (вместо _seed_guide)."""
        cursor = conn.execute("INSERT INTO codebook_version (file_hash) VALUES ('h2')")
        version_id = int(cursor.lastrowid or 0)
        for identifier, group in (("1", "Все"), ("2", "ТОРГ"), ("3", "Все")):
            conn.execute(
                'INSERT INTO question (codebook_version_id, identifier, "group", wording) VALUES (?, ?, ?, ?)',
                (version_id, identifier, group, f"Вопрос {identifier}?"),
            )
        conn.commit()

    def test_question_of_other_group_is_not_a_gap(self, db: sqlite3.Connection) -> None:
        interview_id = _seed_interview(db)
        db.execute('UPDATE interview SET "group" = ? WHERE id = ?', ("Не торгующие", interview_id))
        self._seed_grouped_guide(db)
        _seed_l2(db, interview_id)

        blocks = {
            "blocks": [
                {"block": 1, "question": "1", "is_probe": False},
                {"block": 2, "question": None, "is_probe": False},
                {"block": 3, "question": None, "is_probe": False},
            ]
        }
        result = structure_interview(db, "R01", _settings(), FakeProvider([blocks]), code_version="test")

        # Вопрос «2» группы ТОРГ к «Не торгующим» не относится — это не пропуск.
        assert result.stats.unanswered_questions == ["3"]

    def test_question_without_respondent_counts_as_unanswered(
        self, seeded: tuple[sqlite3.Connection, int, dict[str, int]]
    ) -> None:
        """Блок-хвост интервьюера без ответа не закрывает вопрос."""
        conn, _interview_id, _ids = seeded
        blocks = {
            "blocks": [
                {"block": 1, "question": "1", "is_probe": False},
                {"block": 2, "question": "2", "is_probe": False},
                {"block": 3, "question": "3", "is_probe": False},
            ]
        }
        result = structure_interview(conn, "R01", _settings(), FakeProvider([blocks]), code_version="test")
        assert result.stats.unanswered_questions == ["3"]

    def test_report_lists_unanswered(self, seeded: tuple[sqlite3.Connection, int, dict[str, int]]) -> None:
        conn, _interview_id, _ids = seeded
        result = structure_interview(conn, "R01", _settings(), FakeProvider([_BINDINGS]), code_version="test")
        report = "\n".join(format_report(result))
        assert "вопросы гайда без ответов респондента: 3" in report


class TestLatestVersionOnly:
    """В обработку идёт только выбранная версия L2, а не все версии подряд."""

    def test_uses_latest_version_by_default(self, seeded: tuple[sqlite3.Connection, int, dict[str, int]]) -> None:
        conn, interview_id, _ids = seeded
        _seed_l2(conn, interview_id, version=2, text="Вторая версия текста. Ещё одна мысль.")

        result = structure_interview(conn, "R01", _settings(), None, code_version="test")
        assert result.l2_version == 2
        assert _fragments_match_l2(conn, interview_id)
        assert StructureRepository(conn).fragment_l2_versions(interview_id) == {2}

    def test_fragments_from_old_version_are_rebuilt(
        self, seeded: tuple[sqlite3.Connection, int, dict[str, int]]
    ) -> None:
        conn, interview_id, _ids = seeded
        structure_interview(conn, "R01", _settings(), None, code_version="test")
        assert StructureRepository(conn).fragment_l2_versions(interview_id) == {1}

        _seed_l2(conn, interview_id, version=2, text="Новая версия. Вторая мысль.")

        # Без --rebuild: фрагменты версии 1 не должны остаться рядом с версией 2.
        result = structure_interview(conn, "R01", _settings(), None, code_version="test")
        assert result.skipped is False
        assert StructureRepository(conn).fragment_l2_versions(interview_id) == {2}
        assert _fragments_match_l2(conn, interview_id)

    def test_explicit_version_respected(self, seeded: tuple[sqlite3.Connection, int, dict[str, int]]) -> None:
        conn, interview_id, _ids = seeded
        _seed_l2(conn, interview_id, version=2, text="Вторая версия. Ещё мысль.")
        result = structure_interview(conn, "R01", _settings(), None, l2_version=1, code_version="test")
        assert result.l2_version == 1
        assert StructureRepository(conn).fragment_l2_versions(interview_id) == {1}

    def test_missing_version_raises(self, seeded: tuple[sqlite3.Connection, int, dict[str, int]]) -> None:
        conn, _interview_id, _ids = seeded
        with pytest.raises(StructureError, match="версии"):
            structure_interview(conn, "R01", _settings(), None, l2_version=9)

    def test_versions_are_per_interview(self, db: sqlite3.Connection) -> None:
        """Номер версии L2 локальный: у разных интервью последние версии не совпадают."""
        _seed_guide(db)
        first = _seed_interview(db, "R01")
        second = _seed_interview(db, "R02")
        _seed_l2(db, first, version=1)
        _seed_l2(db, first, version=2, text="Вторая версия. Ещё мысль.")
        _seed_l2(db, second, version=1)
        _seed_l2(db, second, version=2, text="Вторая версия. Ещё мысль.")
        _seed_l2(db, second, version=3, text="Третья версия. Мысль.")

        r01 = structure_interview(db, "R01", _settings(), None, code_version="test")
        r02 = structure_interview(db, "R02", _settings(), None, code_version="test")
        assert r01.l2_version == 2
        assert r02.l2_version == 3
        assert StructureRepository(db).fragment_l2_versions(first) == {2}
        assert StructureRepository(db).fragment_l2_versions(second) == {3}
        assert _fragments_match_l2(db, first)
        assert _fragments_match_l2(db, second)


class TestStructureRun:
    """Запись прогона и повторный запуск."""

    def test_writes_run(self, seeded: tuple[sqlite3.Connection, int, dict[str, int]]) -> None:
        conn, _interview_id, _ids = seeded
        result = structure_interview(conn, "R01", _settings(), None, code_version="test")
        assert result.run_id is not None
        model, prompt = conn.execute("SELECT model, prompt_hash FROM run WHERE id = ?", (result.run_id,)).fetchone()
        assert model == "deterministic"
        assert prompt

    def test_second_run_skipped(self, seeded: tuple[sqlite3.Connection, int, dict[str, int]]) -> None:
        conn, _interview_id, _ids = seeded
        structure_interview(conn, "R01", _settings(), None, code_version="test")
        assert structure_interview(conn, "R01", _settings(), None, code_version="test").skipped is True

    def test_rebuild_replaces_fragments(self, seeded: tuple[sqlite3.Connection, int, dict[str, int]]) -> None:
        conn, interview_id, _ids = seeded
        structure_interview(conn, "R01", _settings(), None, code_version="test")
        first = len(StructureRepository(conn).fragments_of_interview(interview_id))
        structure_interview(conn, "R01", _settings(), None, code_version="test", rebuild=True)
        assert len(StructureRepository(conn).fragments_of_interview(interview_id)) == first

    def test_missing_interview_raises(self, db: sqlite3.Connection) -> None:
        with pytest.raises(StructureError, match="не найдено"):
            structure_interview(db, "R99", _settings(), None)

    def test_missing_l2_raises(self, db: sqlite3.Connection) -> None:
        _seed_interview(db)
        _seed_guide(db)
        with pytest.raises(StructureError, match="L2"):
            structure_interview(db, "R01", _settings(), None)


class TestPrompts:
    """Промпты заполняются без незакрытых подстановок."""

    def test_segment_prompt_substitutions(self) -> None:
        prompt = build_segment_prompt(7, "Текст реплики.", "Вопрос гайда?", ["термин: а -> б"])
        assert "{text}" not in prompt
        assert "{hints}" not in prompt
        assert "Текст реплики." in prompt
        assert "Вопрос гайда?" in prompt


@pytest.mark.real_transcript
class TestRealTranscript:
    """Распределение реплик по вопросам на реальном интервью."""

    def test_distribution_printed(self, tmp_path: Path) -> None:
        db_path = os.environ.get("QA_REAL_DB_PATH")
        if not os.environ.get("QA_REAL_TRANSCRIPT_PATH") or not db_path:
            pytest.skip("Нужны QA_REAL_TRANSCRIPT_PATH и QA_REAL_DB_PATH")
        source = Path(db_path)
        if not source.exists():
            pytest.skip(f"Файл БД не найден: {source}")

        target = tmp_path / "real.db"
        target.write_bytes(source.read_bytes())
        conn = get_connection(target)
        try:
            migrate(conn, target, _MIGRATIONS_DIR, tmp_path / "backups")
            result = structure_interview(conn, "R01", _settings(), None, code_version="test", rebuild=True)
            assert result.stats.fragments > 0
            print("распределение реплик по вопросам:", result.stats.per_question)
        finally:
            conn.close()
