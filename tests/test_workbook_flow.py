"""Сквозные проверки книги: группы, отсутствие вопросов ТОРГ, строгие колонки, пустые темы."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from qa.db.connection import get_connection
from qa.db.migrations import migrate
from qa.pipeline.init_interview import build_plan, plan_to_meta
from qa.pipeline.load_codebook import load_codebook
from qa.pipeline.load_project import load_project
from qa.pipeline.load_transcript import load_transcript
from qa.pipeline.workbook import WorkbookError
from qa.settings import Settings
from tests.fixtures.workbook_factory import CODES, QUESTIONS, build_workbook

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations"

_TRANSCRIPT = "\n".join(
    [
        "Интервью R02, 2026-11-03",
        "00:00:01\tИванов И.И.\tЗдравствуйте, расскажите про подключение",
        "00:00:10\tПетров П.П.\tНу я подключил счёт и всё работает хорошо уже месяц",
        "00:00:20\tИванов И.И.\tУгу.",
        "00:00:25\tПетров П.П.\tДа, всё нормально",
    ]
)


@pytest.fixture()
def settings() -> Settings:
    return Settings.load(_PROJECT_ROOT / "config.yaml")


@pytest.fixture()
def db(tmp_path: Path) -> sqlite3.Connection:
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    migrate(conn, db_path, _MIGRATIONS_DIR, tmp_path / "backups")
    yield conn
    conn.close()


def _book_without_torg(tmp_path: Path) -> Path:
    """Книга без вопросов группы «ТОРГ»: коды не ссылаются на отсутствующие вопросы."""
    questions = [row for row in QUESTIONS if row[2] != "ТОРГ"]
    codes = [[*row[:8], "—" if row[0] in {"E0", "E1"} else row[8]] for row in CODES]
    return build_workbook(
        tmp_path / "book.xlsx",
        questions=questions,
        codes=codes,
        extra_sheets={
            "Интервью": [["R01", "R01", "не торгующие", "2026-11-02", "R01.txt", None, None]],
        },
    )


class TestFlowWithoutTorgQuestions:
    """Вопросы ТОРГ не обязательны: книга и транскрипт загружаются, группы приведены."""

    def test_full_flow(self, db: sqlite3.Connection, tmp_path: Path, settings: Settings) -> None:
        path = _book_without_torg(tmp_path)
        codebook = load_codebook(db, path, settings.workbook, groups=settings.groups)
        assert codebook.counts["questions"] == len([row for row in QUESTIONS if row[2] != "ТОРГ"])

        project = load_project(
            db,
            path,
            settings.workbook,
            groups=settings.groups,
            lexicon_status_map=settings.lexicon_status_map,
        )
        assert project.counts["interviews"] == 1
        stored_group = db.execute("SELECT \"group\" FROM interview WHERE id_external = 'R01'").fetchone()[0]
        assert stored_group == "Не торгующие"

        # Интервью уже описано листом «Интервью»: повторная загрузка не создаёт дублей.
        transcript_r01 = tmp_path / "R01.txt"
        transcript_r01.write_text(_TRANSCRIPT.replace("R02", "R01"), encoding="utf-8")
        plan_r01 = build_plan(db, transcript_r01, "не торгующие", settings.groups)
        assert plan_r01.group == "Не торгующие"
        result_r01 = load_transcript(db, transcript_r01, plan_to_meta(plan_r01), {}, settings.pipeline)
        assert result_r01.skipped is True
        assert db.execute("SELECT COUNT(*) FROM interview WHERE id_external = 'R01'").fetchone()[0] == 1

        # Новое интервью без init-interview: план берёт группу из --group, транскрипт пишется.
        transcript = tmp_path / "R02.txt"
        transcript.write_text(_TRANSCRIPT, encoding="utf-8")
        plan = build_plan(db, transcript, "не торгующие", settings.groups)
        assert plan.group == "Не торгующие"
        result = load_transcript(db, transcript, plan_to_meta(plan), {}, settings.pipeline)
        assert result.skipped is False
        assert result.n_turns == 4
        group = db.execute("SELECT \"group\" FROM interview WHERE id_external = 'R02'").fetchone()[0]
        assert group == "Не торгующие"

    def test_question_groups_normalized(self, db: sqlite3.Connection, tmp_path: Path, settings: Settings) -> None:
        path = _book_without_torg(tmp_path)
        load_codebook(db, path, settings.workbook, groups=settings.groups)
        groups = {row[0] for row in db.execute('SELECT "group" FROM question')}
        assert groups <= {"Все", "ТОРГ", "Не торгующие"}
        assert "ТОРГ" not in groups


class TestStrictThemeColumn:
    """Опечатка в кодах листа «Темы» останавливает загрузку с листом и строкой."""

    def test_typo_in_code_raises(self, db: sqlite3.Connection, tmp_path: Path, settings: Settings) -> None:
        path = build_workbook(
            tmp_path / "book.xlsx",
            themes=[["Риски", "Риски", "R 1", "черновик", "—"]],
        )
        with pytest.raises(WorkbookError) as error:
            load_codebook(db, path, settings.workbook, groups=settings.groups)
        message = str(error.value)
        assert "Темы" in message
        assert "строка 2" in message
        assert "R 1" in message


class TestEmptyThemes:
    """Пустая тема и узел без тем — заметки, а не ошибка."""

    def test_node_without_theme_is_note(self, db: sqlite3.Connection, tmp_path: Path, settings: Settings) -> None:
        path = build_workbook(
            tmp_path / "book.xlsx",
            themes=[["Риски", None, None, "черновик", "—"]],
        )
        result = load_codebook(db, path, settings.workbook, groups=settings.groups)
        nodes = {node for node, _themes in result.tree}
        assert "Риски" in nodes
        assert any("без темы" in note for note in result.notes)

    def test_theme_without_codes_is_note(self, db: sqlite3.Connection, tmp_path: Path, settings: Settings) -> None:
        path = build_workbook(
            tmp_path / "book.xlsx",
            themes=[
                ["Риски", "Риски", "R1; R2", "черновик", "—"],
                ["Риски", "Пустая тема", "—", "черновик", "—"],
            ],
        )
        result = load_codebook(db, path, settings.workbook, groups=settings.groups)
        tree = dict((node, dict(themes)) for node, themes in result.tree)
        assert "Пустая тема" in tree["Риски"]
        assert tree["Риски"]["Пустая тема"] == []
        assert any("без кодов" in note for note in result.notes)

    def test_row_without_node_is_note(self, db: sqlite3.Connection, tmp_path: Path, settings: Settings) -> None:
        path = build_workbook(
            tmp_path / "book.xlsx",
            themes=[["Риски", "Риски", "R1", "черновик", "—"], [None, "Без узла", "R2", "черновик", "—"]],
        )
        result = load_codebook(db, path, settings.workbook, groups=settings.groups)
        assert any("без узла" in note for note in result.notes)
        nodes = {node for node, _themes in result.tree}
        assert nodes == {"Риски"}
