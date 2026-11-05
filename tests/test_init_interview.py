"""Тесты init-interview: код из имени файла, дата из заголовка, роли, NAMES_MAP_PATH."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import yaml

from qa.db.connection import get_connection
from qa.db.migrations import migrate
from qa.pipeline.init_interview import (
    apply_plan,
    build_plan,
    create_names_map_skeleton,
    normalize_group,
    plan_to_meta,
    render_plan,
)
from qa.pipeline.load_transcript import TranscriptMetaError, load_names_map
from qa.settings import Settings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_GROUPS = Settings.load(_PROJECT_ROOT / "config.yaml").groups
_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations"

_TRANSCRIPT = "\n".join(
    [
        "Интервью R01, 2026-11-02",
        "00:00:01\tИванов И.И.\tЗдравствуйте, расскажите про подключение",
        "00:00:10\tПетров П.П.\tНу я подключил счёт и всё работает хорошо уже месяц",
        "00:00:20\tИванов И.И.\tУгу.",
        "00:00:25\tПетров П.П.\tДа, всё нормально",
    ]
)


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


class TestBuildPlan:
    """План интервью из имени файла, заголовка и числа слов."""

    def test_code_from_filename_roles_by_words(self, db: sqlite3.Connection, transcript: Path) -> None:
        plan = build_plan(db, transcript, "не торгующие")
        assert plan.interview_id == "R01"
        assert plan.respondent_code == "R01"
        assert plan.group == "Не торгующие"
        assert plan.interview_date == "2026-11-02"
        # Больше слов у респондента, второй по словам — интервьюер
        assert plan.roles["Петров П.П."] == "RESPONDENT"
        assert plan.roles["Иванов И.И."] == "INTERVIEWER"
        assert plan.needs_confirmation is True

    def test_dotted_header_date(self, db: sqlite3.Connection, tmp_path: Path) -> None:
        path = tmp_path / "R02.txt"
        path.write_text("Интервью, 02.11.2026\n00:00:01\tА\tтекст", encoding="utf-8")
        plan = build_plan(db, path, "ТОРГ")
        assert plan.interview_date == "2026-11-02"

    def test_missing_group_raises(self, db: sqlite3.Connection, transcript: Path) -> None:
        with pytest.raises(TranscriptMetaError, match="--group"):
            build_plan(db, transcript, None)

    def test_bad_filename_raises(self, db: sqlite3.Connection, tmp_path: Path) -> None:
        path = tmp_path / "interview.txt"
        path.write_text(_TRANSCRIPT, encoding="utf-8")
        with pytest.raises(TranscriptMetaError, match="R01"):
            build_plan(db, path, "ТОРГ")

    def test_group_and_date_from_db_without_roles(self, db: sqlite3.Connection, transcript: Path) -> None:
        """Лист «Интервью» заполнен: группа и дата берутся из БД."""
        db.execute(
            'INSERT INTO interview (id_external, name, respondent_code, "group", interview_date)'
            " VALUES ('R01', NULL, 'R01', 'не торгующие', '2026-10-15')"
        )
        db.commit()
        plan = build_plan(db, transcript, None)
        assert plan.group == "Не торгующие"
        assert plan.interview_date == "2026-11-02"
        assert plan.existing is True
        # Роли ещё не подтверждены — спросить оператора
        assert plan.needs_confirmation is True

    def test_roles_from_db_not_asked_again(self, db: sqlite3.Connection, transcript: Path) -> None:
        plan = build_plan(db, transcript, "не торгующие")
        apply_plan(db, plan)
        second = build_plan(db, transcript, None)
        assert second.from_workbook is True
        assert second.needs_confirmation is False
        assert second.roles == plan.roles
        assert any("не спрашиваются" in note for note in second.notes)


class TestApplyPlan:
    """Запись плана в БД."""

    def test_idempotent(self, db: sqlite3.Connection, transcript: Path) -> None:
        plan = build_plan(db, transcript, "не торгующие")
        first = apply_plan(db, plan)
        second = apply_plan(db, plan)
        assert first == second
        n_interviews = db.execute("SELECT COUNT(*) FROM interview").fetchone()[0]
        n_speakers = db.execute("SELECT COUNT(*) FROM interview_speaker").fetchone()[0]
        assert n_interviews == 1
        assert n_speakers == 2

    def test_speakers_written(self, db: sqlite3.Connection, transcript: Path) -> None:
        plan = build_plan(db, transcript, "не торгующие")
        apply_plan(db, plan)
        rows = dict(
            db.execute(
                "SELECT s.speaker_label, s.role FROM interview_speaker s"
                " JOIN interview i ON s.interview_id = i.id WHERE i.id_external = 'R01'"
            ).fetchall()
        )
        assert rows == {"Петров П.П.": "RESPONDENT", "Иванов И.И.": "INTERVIEWER"}

    def test_plan_to_meta(self, db: sqlite3.Connection, transcript: Path) -> None:
        plan = build_plan(db, transcript, "не торгующие")
        meta = plan_to_meta(plan, {"Петров": "R02"})
        assert meta.interview_id == "R01"
        assert meta.date == "2026-11-02"
        assert meta.names_to_codes == {"Петров": "R02"}

    def test_render_plan_contains_roles(self, db: sqlite3.Connection, transcript: Path) -> None:
        plan = build_plan(db, transcript, "не торгующие")
        text = render_plan(plan)
        assert "R01" in text
        assert "RESPONDENT" in text
        assert "2026-11-02" in text


class TestGroup:
    """Нормализация группы по config.yaml."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("ТОРГ", "ТОРГ"),
            ("торг", "ТОРГ"),
            ("Торгующие", "ТОРГ"),
            ("Не торгующие", "Не торгующие"),
            ("не торгующие", "Не торгующие"),
            ("не_торгующие", "Не торгующие"),
            ("  Не Торгующие  ", "Не торгующие"),
            ("Все", "Все"),
            ("все", "Все"),
        ],
    )
    def test_normalize(self, value: str, expected: str) -> None:
        assert normalize_group(value, _GROUPS) == expected

    def test_case_insensitive_same_value(self) -> None:
        assert normalize_group("не торгующие", _GROUPS) == normalize_group("Не торгующие", _GROUPS)

    def test_unknown_kept_as_is(self) -> None:
        """Неизвестное значение сохраняется как есть, без исключения."""
        assert normalize_group("  смешанная  ", _GROUPS) == "смешанная"

    def test_config_missing_alias_raises(self) -> None:
        """Один алиас у двух канонических значений — ошибка конфига."""
        from qa.settings import GroupsSettings

        with pytest.raises(ValueError, match="Алиас группы"):
            GroupsSettings(canonical=["ТОРГ", "Все"], aliases={"ТОРГ": ["все"]}).lookup()


class TestNamesMap:
    """Заготовка таблицы «имя -> код» вне проекта."""

    def test_skeleton_created(
        self, db: sqlite3.Connection, transcript: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "external" / "names.yaml"
        monkeypatch.setenv("NAMES_MAP_PATH", str(target))
        plan = build_plan(db, transcript, "не торгующие")
        path = create_names_map_skeleton(plan)
        assert path == target
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert set(data) == {"Петров П.П.", "Иванов И.И."}
        assert all(value == "" for value in data.values())
        assert load_names_map(str(target)) == {key: "" for key in data}

    def test_existing_file_not_overwritten(
        self, db: sqlite3.Connection, transcript: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "names.yaml"
        target.write_text("Петров П.П.: R07\n", encoding="utf-8")
        monkeypatch.setenv("NAMES_MAP_PATH", str(target))
        plan = build_plan(db, transcript, "не торгующие")
        create_names_map_skeleton(plan)
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert data["Петров П.П."] == "R07"
        assert data["Иванов И.И."] == ""

    def test_no_env_no_skeleton(
        self, db: sqlite3.Connection, transcript: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NAMES_MAP_PATH", raising=False)
        plan = build_plan(db, transcript, "не торгующие")
        assert create_names_map_skeleton(plan) is None


class TestHeaderSanitizing:
    """Заголовок интервью сохраняется без имён."""

    def test_header_is_pseudonymized(self, db: sqlite3.Connection, tmp_path: Path) -> None:
        from qa.pipeline.load_transcript import build_replacement_map, load_transcript
        from qa.settings import PipelineSettings

        header = "Интервью с Петров П.П., 01.10.2026"
        path = tmp_path / "R01.txt"
        path.write_text(header + "\n00:00:01	Петров П.П.	Я подключил счёт", encoding="utf-8")
        plan = build_plan(db, path, "не торгующие")
        meta = plan_to_meta(plan)
        assert meta.roles == {"Петров П.П.": "RESPONDENT"}
        mapping = build_replacement_map(meta)
        assert mapping["Петров П.П."] == "R01"

        load_transcript(db, path, meta, mapping, PipelineSettings(), groups=_GROUPS)

        name, interview_date = db.execute(
            "SELECT name, interview_date FROM interview WHERE id_external = 'R01'"
        ).fetchone()
        assert "Петров" not in name
        assert "R01" in name
        assert interview_date == "2026-10-01"


class TestInitThenLoad:
    """Цепочка init-interview -> load-transcript на одной БД."""

    def test_init_then_load_fills_turns(self, db: sqlite3.Connection, transcript: Path) -> None:
        from qa.pipeline.load_transcript import build_replacement_map, load_transcript
        from qa.settings import PipelineSettings

        plan = build_plan(db, transcript, "не торгующие", _GROUPS)
        apply_plan(db, plan, _GROUPS)
        assert db.execute("SELECT COUNT(*) FROM interview").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM turn").fetchone()[0] == 0

        meta = plan_to_meta(plan)
        result = load_transcript(db, transcript, meta, build_replacement_map(meta), PipelineSettings(), groups=_GROUPS)
        assert result.skipped is False
        assert result.n_turns == 4
        assert db.execute("SELECT COUNT(*) FROM interview").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM turn").fetchone()[0] == 4
        group, interview_date = db.execute('SELECT "group", interview_date FROM interview').fetchone()
        assert group == "Не торгующие"
        assert interview_date == "2026-11-02"

    def test_repeat_load_skips_without_duplicates(self, db: sqlite3.Connection, transcript: Path) -> None:
        from qa.pipeline.load_transcript import build_replacement_map, load_transcript
        from qa.settings import PipelineSettings

        plan = build_plan(db, transcript, "не торгующие", _GROUPS)
        apply_plan(db, plan, _GROUPS)
        meta = plan_to_meta(plan)
        mapping = build_replacement_map(meta)
        first = load_transcript(db, transcript, meta, mapping, PipelineSettings(), groups=_GROUPS)
        second = load_transcript(db, transcript, meta, mapping, PipelineSettings(), groups=_GROUPS)

        assert first.skipped is False
        assert second.skipped is True
        assert db.execute("SELECT COUNT(*) FROM interview").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM turn").fetchone()[0] == first.n_turns
        assert db.execute("SELECT COUNT(*) FROM utterance").fetchone()[0] == first.n_utterances

    def test_metadata_row_completed_not_replaced(self, db: sqlite3.Connection, transcript: Path) -> None:
        """Строка из листа «Интервью» без записей дополняется заголовком и датой транскрипта."""
        from qa.pipeline.load_transcript import build_replacement_map, load_transcript
        from qa.settings import PipelineSettings

        db.execute(
            'INSERT INTO interview (id_external, name, respondent_code, "group", interview_date)'
            " VALUES ('R01', NULL, 'R01', 'не торгующие', '2026-10-15')"
        )
        db.commit()
        plan = build_plan(db, transcript, None, _GROUPS)
        meta = plan_to_meta(plan)
        result = load_transcript(db, transcript, meta, build_replacement_map(meta), PipelineSettings(), groups=_GROUPS)

        assert result.skipped is False
        rows = db.execute('SELECT id, name, "group", interview_date FROM interview').fetchall()
        assert len(rows) == 1
        _id, name, group, interview_date = rows[0]
        assert name is not None
        assert group == "Не торгующие"
        assert interview_date == "2026-11-02"
