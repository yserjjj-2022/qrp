"""Тесты команды `run` (M8): выбор шага, ворота, флаги, идемпотентность.

БД синтетическая, модель не вызывается: тесты проверяют решение о шаге и
отсутствие записи там, где шаг не выполняется. Рабочая база не участвует.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from qa.cli import main
from qa.pipeline.orchestrate import PipelineRunner, warn_names_map
from qa.settings import Settings
from tests.fixtures.m8_db import (
    AnnotationSpec,
    M8Spec,
    TermSpec,
    UtteranceSpec,
    build_m8_db,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

UTTERANCES = [
    UtteranceSpec("INTERVIEWER", "Что ИИ мог бы делать сам?", "Что ИИ мог бы делать сам?", 5, "5"),
    UtteranceSpec(
        "RESPONDENT",
        "Сверять данные я бы доверил, но решение оставляю за собой",
        "Сверять данные я бы доверил, но решение оставляю за собой.",
        40,
        "5",
    ),
    UtteranceSpec(
        "RESPONDENT",
        "Советую новичкам сначала смотреть на комиссию и только потом на доходность",
        "Советую новичкам сначала смотреть на комиссию и только потом на доходность.",
        130,
        "17",
    ),
]


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Настройки с путями во временном каталоге."""
    loaded = Settings.load(_PROJECT_ROOT / "config.yaml")
    loaded.paths.exports = str(tmp_path / "exports")
    loaded.paths.reports = str(tmp_path / "reports")
    loaded.paths.transcripts = str(tmp_path / "transcripts")
    return loaded


def _spec(*, terms: bool = False) -> M8Spec:
    return M8Spec(
        utterances=UTTERANCES,
        annotations=[AnnotationSpec(1, "C1", "Сверять данные"), AnnotationSpec(2, "C1", "Советую новичкам")],
        terms=[TermSpec("комиссию", 2)] if terms else [],
    )


def _run_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM run").fetchone()[0])


def _unit_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM text_unit").fetchone()[0])


def test_plan_reports_state_without_writing(tmp_path: Path, settings: Settings) -> None:
    """План читает состояние и ничего не пишет."""
    conn = build_m8_db(tmp_path, _spec(terms=True))
    before = (_run_count(conn), _unit_count(conn))
    runner = PipelineRunner(conn, "R01", settings)
    plan = runner.plan()
    assert [step.number for step in plan] == list(range(1, 11))
    assert (_run_count(conn), _unit_count(conn)) == before
    conn.close()


def test_gate_one_stops_pipeline(tmp_path: Path, settings: Settings) -> None:
    """Неразобранные кандидаты терминов останавливают конвейер на шаге 4."""
    conn = build_m8_db(tmp_path, _spec(terms=True))
    runner = PipelineRunner(conn, "R01", settings)
    result = runner.execute()
    assert result.stopped_at_gate == 4
    assert not result.all_done
    # L2 не тронут: шаг 5 не выполнялся.
    assert _unit_count(conn) == len(UTTERANCES)
    conn.close()


def test_skip_terms_passes_gate(tmp_path: Path, settings: Settings) -> None:
    """`--skip-terms` проходит ворота 1 и доходит до отчёта."""
    conn = build_m8_db(tmp_path, _spec(terms=True))
    runner = PipelineRunner(conn, "R01", settings, skip_terms=True)
    result = runner.execute()
    assert result.stopped_at_gate is None
    assert result.all_done
    assert (tmp_path / "reports" / "report_R01.html").is_file()
    conn.close()


def test_second_run_changes_nothing(tmp_path: Path, settings: Settings) -> None:
    """Повторный запуск не создаёт новых прогонов и единиц L2."""
    conn = build_m8_db(tmp_path, _spec())
    runner = PipelineRunner(conn, "R01", settings)
    runner.execute()
    before = (_run_count(conn), _unit_count(conn))
    again = PipelineRunner(conn, "R01", settings).execute()
    assert (_run_count(conn), _unit_count(conn)) == before
    assert again.all_done
    assert all(step.status != "done" for step in again.steps)
    conn.close()


def test_quality_gate_disabled_by_default(tmp_path: Path, settings: Settings) -> None:
    """Ворота 2 выключены по умолчанию: файла нет, конвейер идёт дальше."""
    conn = build_m8_db(tmp_path, _spec())
    result = PipelineRunner(conn, "R01", settings).execute()
    assert result.stopped_at_gate is None
    assert not (tmp_path / "exports" / "quality_sample_R01.xlsx").exists()
    step8 = next(step for step in result.steps if step.number == 8)
    assert step8.status == "skipped"
    assert "выключены" in step8.detail
    conn.close()


def test_quality_gate_exports_when_enabled(tmp_path: Path, settings: Settings) -> None:
    """Включённые ворота 2 выгружают выборку и не блокируют конвейер."""
    settings.m8.quality_sample_enabled = True
    conn = build_m8_db(tmp_path, _spec())
    result = PipelineRunner(conn, "R01", settings).execute()
    assert result.stopped_at_gate is None
    assert (tmp_path / "exports" / "quality_sample_R01.xlsx").is_file()
    step8 = next(step for step in result.steps if step.number == 8)
    assert step8.status == "done"
    conn.close()


def test_missing_interview_blocks_at_step_three(tmp_path: Path, settings: Settings) -> None:
    """Без интервью в БД и без транскрипта шаг 3 останавливает конвейер."""
    conn = build_m8_db(tmp_path, M8Spec())
    conn.execute("DELETE FROM annotation")
    conn.execute("DELETE FROM fragment_coding")
    conn.execute("DELETE FROM fragment")
    conn.execute("DELETE FROM text_unit")
    conn.execute("DELETE FROM utterance")
    conn.execute("DELETE FROM turn")
    conn.execute("DELETE FROM interview")
    conn.commit()
    result = PipelineRunner(conn, "R01", settings).execute()
    assert result.steps[2].status == "blocked"
    assert not result.all_done
    conn.close()


def test_cli_dry_run_writes_nothing(
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`run --dry-run` печатает план и не меняет БД."""
    # `.env` не читаем: иначе значения оператора попадут в окружение других тестов.
    monkeypatch.setattr("qa.cli.load_env_file", lambda: None)
    db_path = tmp_path / "m8.db"
    conn = build_m8_db(tmp_path, _spec(terms=True))
    conn.close()
    config = tmp_path / "config.yaml"
    config.write_text(
        f"paths:\n  data: {tmp_path / 'data'}\n  db: {db_path}\n"
        f"  backups: {tmp_path / 'backups'}\n  cache: {tmp_path / 'cache'}\n"
        f"  exports: {tmp_path / 'exports'}\n  reports: {tmp_path / 'reports'}\n",
        encoding="utf-8",
    )
    check = sqlite3.connect(db_path)
    before = _run_count(check)
    check.close()
    with patch.object(sys, "argv", ["qa-pipeline", "run", "R01", "--db", str(db_path), "--dry-run"]):
        assert main() == 0
    check = sqlite3.connect(db_path)
    assert _run_count(check) == before
    check.close()
    assert config.is_file()


def test_cli_uses_db_override_not_working_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`run --db` работает с копией и не создаёт рабочую базу."""
    work_db = tmp_path / "work.db"
    copy_db = tmp_path / "copy.db"
    (tmp_path / "config.yaml").write_text(
        f"paths:\n  data: {tmp_path / 'data'}\n  db: {work_db}\n"
        f"  backups: {tmp_path / 'backups'}\n  cache: {tmp_path / 'cache'}\n"
        f"  exports: {tmp_path / 'exports'}\n  reports: {tmp_path / 'reports'}\n",
        encoding="utf-8",
    )
    conn = build_m8_db(tmp_path, _spec())
    conn.close()
    (tmp_path / "m8.db").replace(copy_db)
    monkeypatch.setattr("qa.cli.load_env_file", lambda: None)
    monkeypatch.chdir(tmp_path)
    with patch.object(sys, "argv", ["qa-pipeline", "run", "R01", "--db", str(copy_db), "--dry-run"]):
        assert main() == 0
    assert copy_db.exists()
    assert not work_db.exists()


def test_names_map_warning_mentions_operator_action(monkeypatch: pytest.MonkeyPatch) -> None:
    """Пустой NAMES_MAP_PATH даёт предупреждение с действием оператора."""
    monkeypatch.delenv("NAMES_MAP_PATH", raising=False)
    warning = warn_names_map()
    assert warning is not None
    assert "NAMES_MAP_PATH" in warning
    monkeypatch.setenv("NAMES_MAP_PATH", "data/names.csv")
    assert warn_names_map() is None
