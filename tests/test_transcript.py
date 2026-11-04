"""Тесты M1a: разбор транскрипта, псевдонимизация, pii_flag, сборка реплик."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest
import yaml

from qa.db.connection import get_connection
from qa.db.migrations import migrate
from qa.pipeline.load_transcript import (
    TranscriptMetaError,
    TranscriptParseError,
    build_replacement_map,
    build_utterances,
    detect_pii,
    is_backchannel,
    load_meta,
    load_transcript,
    parse_transcript,
    pseudonymize,
)
from qa.settings import PipelineSettings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations"

_HEADER = "Интервью с респондентом, 2026-10-01"

_TRANSCRIPT = "\n".join(
    [
        _HEADER,
        "00:00:05\tИванов И.И.\tЗдравствуйте, расскажите про подключение.",
        "00:00:20\tПетров П.П.\tНу, я подключил счёт, мой телефон 89991234567",
        "00:00:25\tИванов И.И.\tУгу.",
        "00:00:26\tПетров П.П.\tи, наверное, удобно, как мне Петров объяснил",
        "00:00:40\tИванов И.И.\tА по почте куда писать?",
        "00:00:50\tПетров П.П.\tКлиент писал на ivan@example.com, я не помню",
        "00:01:00\tСидорова А.А.\tСпасибо всем.",
        "00:01:05\tПетров П.П.\tДа.",
    ]
)

_META = {
    "interview_id": "IV01",
    "respondent_code": "R01",
    "group": "ТОРГ",
    "roles": {
        "Иванов И.И.": "INTERVIEWER",
        "Петров П.П.": "RESPONDENT",
        "Сидорова А.А.": "OBSERVER",
    },
    "names_to_codes": {"Петров": "R02"},
    "date": "2026-10-01",
    "transcript_source": "machine ASR",
}


@pytest.fixture()
def db(tmp_path: Path) -> sqlite3.Connection:
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    migrate(conn, db_path, _MIGRATIONS_DIR, tmp_path / "backups")
    yield conn
    conn.close()


@pytest.fixture()
def transcript_path(tmp_path: Path) -> Path:
    path = tmp_path / "transcript.txt"
    path.write_text(_TRANSCRIPT, encoding="utf-8")
    return path


@pytest.fixture()
def meta_path(tmp_path: Path) -> Path:
    path = tmp_path / "interview.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(_META, f, allow_unicode=True)
    return path


@pytest.fixture()
def meta(meta_path: Path):  # noqa: ANN201
    return load_meta(meta_path)


class TestParse:
    """Разбор формата транскрипта."""

    def test_records_count_and_order(self, transcript_path: Path) -> None:
        header, records = parse_transcript(transcript_path)
        assert header == _HEADER
        assert len(records) == 8
        assert [r.start_sec for r in records] == sorted(r.start_sec for r in records)
        assert records[0].speaker_label == "Иванов И.И."
        assert records[-1].text == "Да."

    def test_bom_ok(self, tmp_path: Path) -> None:
        path = tmp_path / "bom.txt"
        path.write_text("\ufeff" + _TRANSCRIPT, encoding="utf-8")
        header, records = parse_transcript(path)
        assert header == _HEADER
        assert len(records) == 8

    def test_malformed_line_raises_with_number(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.txt"
        path.write_text(_HEADER + "\n00:00:01\tМетка\tтекст\nэто не по формату\n", encoding="utf-8")
        with pytest.raises(TranscriptParseError, match="Строка 3"):
            parse_transcript(path)

    def test_decreasing_time_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_time.txt"
        path.write_text(_HEADER + "\n00:05:00\tМетка\tтекст\n00:04:00\tМетка\tтекст\n", encoding="utf-8")
        with pytest.raises(TranscriptParseError, match="убывает"):
            parse_transcript(path)

    def test_meta_missing_field(self, tmp_path: Path) -> None:
        path = tmp_path / "meta.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump({"interview_id": "X"}, f)
        with pytest.raises(TranscriptMetaError, match="не хватает полей"):
            load_meta(path)

    def test_meta_bad_role(self, tmp_path: Path) -> None:
        path = tmp_path / "meta.yaml"
        data = {**_META, "roles": {"Кто-то": "GUEST"}}
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)
        with pytest.raises(TranscriptMetaError, match="Недопустимые роли"):
            load_meta(path)


class TestPseudonymize:
    """Псевдонимизация и PII."""

    def test_names_replaced(self, meta) -> None:  # noqa: ANN001
        mapping = build_replacement_map(meta)
        out = pseudonymize("Петров сказал, что Петров прав", mapping)
        assert "Петров" not in out
        assert "R02" in out

    def test_labels_replaced(self, meta) -> None:  # noqa: ANN001
        mapping = build_replacement_map(meta)
        out = pseudonymize("Как говорил Петров П.П., это риск", mapping)
        assert "Петров П.П." not in out
        assert "R01" in out

    def test_pii_phone_email(self) -> None:
        assert detect_pii("телефон 89991234567")
        assert detect_pii("почта a@b.ru")
        assert detect_pii("счёт 40817810123456789012")
        assert not detect_pii("обычный текст без данных")


class TestBackchannel:
    """Вставки-подтверждения."""

    def test_yes(self) -> None:
        words = {"угу", "ага", "да", "так"}
        assert is_backchannel("Угу.", words, 3)
        assert is_backchannel("Да, да.", words, 3)

    def test_no(self) -> None:
        words = {"угу", "ага", "да", "так"}
        assert not is_backchannel("А расскажите подробнее", words, 3)
        assert not is_backchannel("Угу, и что дальше случилось", words, 3)


class TestAssembly:
    """Сборка реплик."""

    def test_merge_across_backchannel(self) -> None:
        turns = [
            (1, "RESPONDENT", "часть одна"),
            (2, "INTERVIEWER", "Угу."),
            (3, "RESPONDENT", "часть две"),
        ]
        drafts = build_utterances(turns, {"угу"}, 3)
        assert len(drafts) == 2
        main = drafts[0]
        assert main.speaker_role == "RESPONDENT"
        assert main.turn_ids == [1, 3]
        # Критерий 5: объединение текстов записей (без вставок) = текст реплики
        assert main.text == "часть одна часть две"
        assert drafts[1].is_backchannel is True
        assert drafts[1].text == "Угу."

    def test_real_speaker_change_breaks(self) -> None:
        turns = [
            (1, "RESPONDENT", "ответ"),
            (2, "INTERVIEWER", "следующий вопрос длинный"),
            (3, "RESPONDENT", "второй ответ"),
        ]
        drafts = build_utterances(turns, {"угу"}, 3)
        assert len(drafts) == 3
        assert all(not d.is_backchannel for d in drafts)


class TestLoadTranscript:
    """Сквозная загрузка интервью в БД."""

    def test_full_load(self, db: sqlite3.Connection, transcript_path: Path, meta) -> None:  # noqa: ANN001
        mapping = build_replacement_map(meta)
        result = load_transcript(db, transcript_path, meta, mapping, PipelineSettings())
        assert result.n_turns == 8
        assert result.skipped is False
        # Три роли присутствуют
        roles = {row[0] for row in db.execute("SELECT DISTINCT speaker_role FROM turn")}
        assert roles == {"INTERVIEWER", "RESPONDENT", "OBSERVER"}
        # pii_flag проставлен (телефон + почта)
        pii_rows = db.execute("SELECT COUNT(*) FROM turn WHERE pii_flag = 1").fetchone()[0]
        assert pii_rows == 2
        # Меток говорящих и имён из names_to_codes в тексте нет
        rows = db.execute("SELECT text FROM turn").fetchall()
        all_text = " ".join(r[0] for r in rows)
        for label in _META["roles"]:
            assert label not in all_text
        assert "Петров" not in all_text

    def test_no_duplicates_on_reload(self, db: sqlite3.Connection, transcript_path: Path, meta) -> None:  # noqa: ANN001
        mapping = build_replacement_map(meta)
        load_transcript(db, transcript_path, meta, mapping, PipelineSettings())
        n_uttr_before = db.execute("SELECT COUNT(*) FROM utterance").fetchone()[0]
        result2 = load_transcript(db, transcript_path, meta, mapping, PipelineSettings())
        assert result2.skipped is True
        n_int = db.execute("SELECT COUNT(*) FROM interview").fetchone()[0]
        n_turns = db.execute("SELECT COUNT(*) FROM turn").fetchone()[0]
        n_uttr = db.execute("SELECT COUNT(*) FROM utterance").fetchone()[0]
        assert n_int == 1
        assert n_turns == 8
        assert n_uttr == n_uttr_before

    def test_unknown_label_raises(self, db: sqlite3.Connection, tmp_path: Path, meta_path: Path) -> None:
        path = tmp_path / "t2.txt"
        path.write_text(_HEADER + "\n00:00:01\tНеизвестный X\tтекст", encoding="utf-8")
        meta = load_meta(meta_path)
        with pytest.raises(TranscriptMetaError, match="не описаны"):
            load_transcript(db, path, meta, {}, PipelineSettings())

    def test_utterance_text_matches_turns(self, db: sqlite3.Connection, transcript_path: Path, meta) -> None:  # noqa: ANN001
        """Критерий 5: текст не теряется при сборке."""
        mapping = build_replacement_map(meta)
        load_transcript(db, transcript_path, meta, mapping, PipelineSettings())
        import json

        rows = db.execute("SELECT id, text, turn_ids FROM utterance WHERE is_backchannel = 0").fetchall()
        turn_texts = dict(db.execute("SELECT id, text FROM turn").fetchall())
        for _uid, text, turn_ids_json in rows:
            ids = json.loads(turn_ids_json)
            joined = " ".join(turn_texts[tid] for tid in ids)
            assert joined == text

    def test_interview_columns(self, db: sqlite3.Connection, transcript_path: Path, meta) -> None:  # noqa: ANN001
        mapping = build_replacement_map(meta)
        load_transcript(db, transcript_path, meta, mapping, PipelineSettings())
        row = db.execute(
            "SELECT respondent_code, \"group\", transcript_source FROM interview WHERE id_external = 'IV01'"
        ).fetchone()
        assert row == ("R01", "ТОРГ", "machine ASR")


@pytest.mark.real_transcript
class TestRealTranscript:
    """Тест на реальном транскрипте. Пропускается без env-переменных."""

    def test_218_records(self, db: sqlite3.Connection) -> None:
        tr_path = os.environ.get("QA_REAL_TRANSCRIPT_PATH")
        meta_p = os.environ.get("QA_REAL_INTERVIEW_META")
        if not tr_path or not meta_p:
            pytest.skip("Не заданы QA_REAL_TRANSCRIPT_PATH / QA_REAL_INTERVIEW_META")
        transcript = Path(tr_path)
        meta = load_meta(Path(meta_p))
        if not transcript.exists():
            pytest.skip(f"Файл не найден: {transcript}")

        header, records = parse_transcript(transcript)
        assert len(records) == 218, f"Ожидалось 218 записей, получено {len(records)}"
        labels = {r.speaker_label for r in records}
        assert len(labels) == 3, f"Ожидалось 3 метки, получено {len(labels)}"
        assert [r.start_sec for r in records] == sorted(r.start_sec for r in records)

        names_map: dict[str, str] = {}
        names_map_path = os.environ.get("NAMES_MAP_PATH")
        if names_map_path:
            from qa.pipeline.load_transcript import load_names_map

            names_map = load_names_map(names_map_path)
        mapping = build_replacement_map(meta, names_map)
        result = load_transcript(db, transcript, meta, mapping, PipelineSettings())
        assert result.n_turns == 218

        all_text = " ".join(r[0] for r in db.execute("SELECT text FROM turn").fetchall())
        for label in labels:
            assert label not in all_text, f"Метка {label!r} осталась в тексте"
        for name in mapping:
            assert name not in all_text, f"Имя {name!r} осталось в тексте"
