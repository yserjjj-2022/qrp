"""Тесты ограничений отчёта, зависящих от состояния конвейера (M8).

Ограничение — не украшение: оно сообщает читателю, чего в числах и цитатах
нет. Проверяется, что оно появляется только при соответствующем состоянии и
исчезает, когда состояние исправлено.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from qa.pipeline.quality_sample import export_quality_sample, write_marker
from qa.report.limitations import pipeline_limitations
from qa.report.model import Limitation
from qa.settings import Settings
from tests.fixtures.m8_db import M8Spec, TermSpec, UtteranceSpec, build_m8_db

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _spec(*, terms: bool = False) -> M8Spec:
    return M8Spec(
        utterances=[
            UtteranceSpec("RESPONDENT", "Сверять данные я готов.", "Сверять данные я готов.", 65, "5"),
        ],
        terms=[TermSpec("сверять", 0)] if terms else [],
    )


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    loaded = Settings.load(_PROJECT_ROOT / "config.yaml")
    loaded.paths.exports = str(tmp_path / "exports")
    return loaded


def _codes(items: tuple[Limitation, ...]) -> set[str]:
    return {item.code for item in items}


def test_pending_terms_are_reported(tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """Неразобранные кандидаты терминов попадают в ограничения."""
    monkeypatch.delenv("NAMES_MAP_PATH", raising=False)
    conn: sqlite3.Connection = build_m8_db(tmp_path, _spec(terms=True))
    try:
        items = pipeline_limitations(conn, "R01", settings)
    finally:
        conn.close()
    assert "terms_pending" in _codes(items)
    assert any("Кандидатов терминов без решения: 1" in str(item.text) for item in items)


def test_no_terms_no_limitation(tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """Без кандидатов ограничение о терминах не появляется."""
    monkeypatch.delenv("NAMES_MAP_PATH", raising=False)
    conn = build_m8_db(tmp_path, _spec())
    try:
        items = pipeline_limitations(conn, "R01", settings)
    finally:
        conn.close()
    assert "terms_pending" not in _codes(items)


def test_unreviewed_sample_is_reported(
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """При включённых воротах 2 отсутствие маркера даёт ограничение."""
    monkeypatch.delenv("NAMES_MAP_PATH", raising=False)
    settings.m8.quality_sample_enabled = True
    conn = build_m8_db(tmp_path, _spec())
    try:
        items = pipeline_limitations(conn, "R01", settings)
        assert "quality_sample" in _codes(items)

        exports = Path(settings.paths.exports)
        path = export_quality_sample(conn, "R01", exports / "quality_sample_R01.xlsx")
        write_marker(path.path)
        after = pipeline_limitations(conn, "R01", settings)
    finally:
        conn.close()
    assert "quality_sample" not in _codes(after)


def test_disabled_gate_adds_no_limitation(
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Выключенные ворота 2 не добавляют строку «не подтверждено»."""
    monkeypatch.delenv("NAMES_MAP_PATH", raising=False)
    assert settings.m8.quality_sample_enabled is False
    conn = build_m8_db(tmp_path, _spec())
    try:
        items = pipeline_limitations(conn, "R01", settings)
    finally:
        conn.close()
    assert "quality_sample" not in _codes(items)


def test_names_map_limitation_follows_env(
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Пустой NAMES_MAP_PATH даёт ограничение, заданный — нет."""
    conn = build_m8_db(tmp_path, _spec())
    try:
        monkeypatch.delenv("NAMES_MAP_PATH", raising=False)
        assert "names_map" in _codes(pipeline_limitations(conn, "R01", settings))

        monkeypatch.setenv("NAMES_MAP_PATH", str(tmp_path / "names.csv"))
        assert "names_map" not in _codes(pipeline_limitations(conn, "R01", settings))
    finally:
        conn.close()


def test_unknown_interview_gives_nothing(tmp_path: Path, settings: Settings) -> None:
    """Для неизвестного интервью состояние конвейера ничего не добавляет."""
    conn = build_m8_db(tmp_path, _spec())
    try:
        assert pipeline_limitations(conn, "R99", settings) == ()
    finally:
        conn.close()


def test_applied_manual_edits_are_reported(
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Правки, записанные в прогон L2, видны в ограничениях."""
    monkeypatch.delenv("NAMES_MAP_PATH", raising=False)
    conn = build_m8_db(tmp_path, _spec())
    try:
        run_id = int(conn.execute("SELECT MAX(id) FROM run").fetchone()[0])
        conn.execute("UPDATE text_unit SET run_id = ?", (run_id,))
        conn.execute(
            "UPDATE run SET params_json = ? WHERE id = ?",
            ('{"l2_version": 1, "manual_edits": 2, "manual_edits_hash": "abcdef0123456789"}', run_id),
        )
        conn.commit()
        items = pipeline_limitations(conn, "R01", settings)
    finally:
        conn.close()

    assert "manual_edits" in _codes(items)
    assert any("применено разовых правок: 2" in str(item.text) for item in items)
