"""Тесты применения ролей из листа «Роли_говорящих» к записям и репликам."""

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
from qa.pipeline.speaker_roles import SpeakerRolesError, apply_speaker_roles
from qa.settings import PipelineSettings, Settings
from tests.fixtures.workbook_factory import build_workbook

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations"

_TRANSCRIPT = "\n".join(
    [
        "Интервью R01, 2026-11-02",
        "00:00:01\tA\tЗдравствуйте, расскажите про подключение",
        "00:00:10\tБ\tНу я подключил счёт и всё работает хорошо уже месяц",
        "00:00:20\tA\tУгу.",
        "00:00:25\tБ\tДа, всё нормально",
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


@pytest.fixture()
def transcript(tmp_path: Path) -> Path:
    path = tmp_path / "R01.txt"
    path.write_text(_TRANSCRIPT, encoding="utf-8")
    return path


def _load(db: sqlite3.Connection, transcript: Path, settings: Settings) -> None:
    plan = build_plan(db, transcript, "не торгующие", settings.groups)
    load_transcript(db, transcript, plan_to_meta(plan), {}, settings.pipeline, groups=settings.groups)


def _roles(db: sqlite3.Connection) -> dict[str, str]:
    return dict(db.execute("SELECT speaker_label, role FROM interview_speaker").fetchall())


class TestTurnLabel:
    """Метка говорящего пишется в записи транскрипта."""

    def test_label_written(self, db: sqlite3.Connection, transcript: Path, settings: Settings) -> None:
        _load(db, transcript, settings)
        labels = {row[0] for row in db.execute("SELECT DISTINCT speaker_label FROM turn")}
        assert labels == {"A", "Б"}
        assert db.execute("SELECT COUNT(*) FROM turn WHERE speaker_label IS NULL").fetchone()[0] == 0


class TestApplySpeakerRoles:
    """Правка роли доезжает до turn и utterance."""

    def test_role_change_reaches_turns_and_utterances(
        self, db: sqlite3.Connection, transcript: Path, settings: Settings
    ) -> None:
        _load(db, transcript, settings)
        assert _roles(db)["A"] == "INTERVIEWER"

        result = apply_speaker_roles(
            db,
            "R01",
            {"A": "OBSERVER", "Б": "RESPONDENT"},
            settings.pipeline,
            transcript_path=transcript,
            groups=settings.groups,
        )
        db.commit()

        assert _roles(db)["A"] == "OBSERVER"
        assert result.turns_updated == 4
        turn_roles = dict(db.execute("SELECT speaker_label, speaker_role FROM turn").fetchall())
        assert turn_roles == {"A": "OBSERVER", "Б": "RESPONDENT"}
        utterance_roles = {row[0] for row in db.execute("SELECT DISTINCT speaker_role FROM utterance")}
        assert utterance_roles == {"OBSERVER", "RESPONDENT"}

    def test_idempotent(self, db: sqlite3.Connection, transcript: Path, settings: Settings) -> None:
        _load(db, transcript, settings)
        roles = {"A": "OBSERVER", "Б": "RESPONDENT"}
        apply_speaker_roles(db, "R01", roles, settings.pipeline, transcript_path=transcript)
        first = db.execute("SELECT COUNT(*), SUM(is_backchannel) FROM utterance").fetchone()
        apply_speaker_roles(db, "R01", roles, settings.pipeline, transcript_path=transcript)
        db.commit()
        second = db.execute("SELECT COUNT(*), SUM(is_backchannel) FROM utterance").fetchone()
        assert first == second

    def test_unknown_role_rejected(self, db: sqlite3.Connection, transcript: Path, settings: Settings) -> None:
        _load(db, transcript, settings)
        with pytest.raises(ValueError, match="Недопустимые роли"):
            apply_speaker_roles(db, "R01", {"A": "LEAD"}, settings.pipeline)

    def test_missing_interview_rejected(self, db: sqlite3.Connection, settings: Settings) -> None:
        with pytest.raises(SpeakerRolesError, match="нет в БД"):
            apply_speaker_roles(db, "R99", {"A": "RESPONDENT"}, settings.pipeline)

    def test_new_term_links_block_rebuild(self, db: sqlite3.Connection, transcript: Path, settings: Settings) -> None:
        _load(db, transcript, settings)
        utterance_id = db.execute("SELECT id FROM utterance ORDER BY id LIMIT 1").fetchone()[0]
        interview_db_id = db.execute("SELECT id FROM interview WHERE id_external = 'R01'").fetchone()[0]
        db.execute(
            "INSERT INTO new_term (interview_id, surface_form, context_utterance_id) VALUES (?, ?, ?)",
            (interview_db_id, "термин", utterance_id),
        )
        db.commit()

        with pytest.raises(SpeakerRolesError, match="review-terms"):
            apply_speaker_roles(db, "R01", {"A": "OBSERVER"}, settings.pipeline, transcript_path=transcript)


class TestBackfill:
    """Записи, загруженные до появления колонки, получают метку из транскрипта."""

    def test_labels_filled_from_transcript(self, db: sqlite3.Connection, transcript: Path, settings: Settings) -> None:
        _load(db, transcript, settings)
        db.execute("UPDATE turn SET speaker_label = NULL")
        db.commit()

        result = apply_speaker_roles(db, "R01", {"A": "OBSERVER"}, settings.pipeline, transcript_path=transcript)
        db.commit()
        assert result.labels_filled == 4
        assert db.execute("SELECT COUNT(*) FROM turn WHERE speaker_label IS NULL").fetchone()[0] == 0

    def test_missing_transcript_is_note(self, db: sqlite3.Connection, transcript: Path, settings: Settings) -> None:
        _load(db, transcript, settings)
        db.execute("UPDATE turn SET speaker_label = NULL")
        db.commit()

        result = apply_speaker_roles(db, "R01", {}, settings.pipeline, transcript_path=None)
        assert result.labels_filled == 0
        assert any("транскрипт не найден" in note for note in result.notes)


class TestLoadProjectAppliesRoles:
    """load-project применяет лист «Роли_говорящих» к загруженному интервью."""

    def test_roles_applied_after_edit(
        self, db: sqlite3.Connection, tmp_path: Path, transcript: Path, settings: Settings
    ) -> None:
        path = build_workbook(
            tmp_path / "book.xlsx",
            extra_sheets={
                "Интервью": [["R01", "R01", "не торгующие", "2026-11-02", "R01.txt", None, None]],
                "Роли_говорящих": [["R01", "A", "OBSERVER", 5], ["R01", "Б", "RESPONDENT", 12]],
            },
        )
        load_codebook(db, path, settings.workbook, groups=settings.groups)
        load_project(
            db,
            path,
            settings.workbook,
            groups=settings.groups,
            lexicon_status_map=settings.lexicon_status_map,
            pipeline=settings.pipeline,
            transcripts_dir=transcript.parent,
        )
        _load(db, transcript, settings)

        result = load_project(
            db,
            path,
            settings.workbook,
            groups=settings.groups,
            lexicon_status_map=settings.lexicon_status_map,
            pipeline=settings.pipeline,
            transcripts_dir=transcript.parent,
        )
        db.commit()
        assert result.counts["speaker_roles_applied"] == 1
        assert _roles(db)["A"] == "OBSERVER"
        turn_roles = {row[0] for row in db.execute("SELECT DISTINCT speaker_role FROM turn")}
        assert turn_roles == {"OBSERVER", "RESPONDENT"}

    def test_edit_does_not_require_codebook_reload(
        self, db: sqlite3.Connection, tmp_path: Path, settings: Settings
    ) -> None:
        path = build_workbook(tmp_path / "book.xlsx")
        load_codebook(db, path, settings.workbook, groups=settings.groups)
        load_project(db, path, settings.workbook, groups=settings.groups)

        # Правка листа меняет хеш файла, но не кодбук: load-project должен пройти.
        edited = build_workbook(
            tmp_path / "book.xlsx",
            extra_sheets={"Роли_говорящих": [["R01", "A", "RESPONDENT", 3]]},
        )
        result = load_project(
            db,
            edited,
            settings.workbook,
            groups=settings.groups,
            pipeline=PipelineSettings(),
        )
        assert result.counts["interviews"] == 1
