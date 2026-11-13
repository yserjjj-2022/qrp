"""Тесты ворот 2: выборка качества L2 и маркер «просмотрено» (M8).

Проверяется то, от чего зависит смысл ворот: размер выборки, детерминизм,
привязка маркера к хешу выгрузки и подсчёт метрики «доля принятых без правок».
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from openpyxl import load_workbook

from qa.pipeline.quality_sample import (
    DECISIONS,
    export_quality_sample,
    file_hash,
    import_quality_sample,
    marker_matches,
    marker_path,
    quality_sample_path,
    sample_rows,
    write_marker,
)
from tests.fixtures.m8_db import M8Spec, UtteranceSpec, build_m8_db

SPEAKER_LABEL = "Тестовое Имя"


def _spec(count: int = 20) -> M8Spec:
    """Интервью с `count` репликами респондента и разной длиной текста."""
    utterances = [
        UtteranceSpec(
            "RESPONDENT",
            f"Реплика номер {index} про сверку данных.",
            f"Реплика номер {index} про сверку данных.",
            10 * index,
            "5",
        )
        for index in range(1, count + 1)
    ]
    return M8Spec(utterances=utterances)


@pytest.fixture()
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = build_m8_db(tmp_path, _spec())
    yield connection
    connection.close()


def test_sample_size_is_within_bounds(conn: sqlite3.Connection) -> None:
    """Из двадцати реплик берётся 10–15."""
    rows = sample_rows(conn, "R01", min_rows=10, max_rows=15)
    assert 10 <= len(rows) <= 15


def test_sample_is_deterministic(conn: sqlite3.Connection) -> None:
    """Один и тот же seed даёт одну и ту же выборку."""
    first = [row.utterance_id for row in sample_rows(conn, "R01", seed=0)]
    second = [row.utterance_id for row in sample_rows(conn, "R01", seed=0)]
    assert first == second


def test_sample_uses_both_layers(conn: sqlite3.Connection) -> None:
    """В выборке есть L1 и L2 и тайм-код реплики."""
    rows = sample_rows(conn, "R01")
    assert all(row.text_l1 and row.text_l2 for row in rows)
    assert all(row.timecode > 0 for row in rows)


def test_sample_without_l2_fails(tmp_path: Path) -> None:
    """Без слоя L2 выборка невозможна."""
    connection = build_m8_db(tmp_path, M8Spec())
    with pytest.raises(ValueError, match="нет слоя L2"):
        sample_rows(connection, "R01")
    connection.close()


def test_export_writes_xlsx_without_labels(tmp_path: Path, conn: sqlite3.Connection) -> None:
    """Выгрузка содержит реплики, но не метку говорящего."""
    path = quality_sample_path(tmp_path, "R01")
    result = export_quality_sample(conn, "R01", path)

    assert result.path.is_file()
    assert result.n_rows >= 10
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        values = [str(cell) for row in wb[wb.sheetnames[0]].iter_rows(values_only=True) for cell in row if cell]
    finally:
        wb.close()
    assert not any(SPEAKER_LABEL in value for value in values)


def _decide(path: Path, decisions: list[str]) -> None:
    """Проставить решения в выгрузке: колонка «решение» шестая."""
    wb = load_workbook(path)
    ws = wb[wb.sheetnames[0]]
    header_row = None
    for index, row in enumerate(ws.iter_rows(values_only=True), start=1):
        if row and "id_реплики" in [str(value) for value in row if value is not None]:
            header_row = index
            break
    assert header_row is not None
    for offset, decision in enumerate(decisions, start=1):
        ws.cell(row=header_row + offset, column=6, value=decision)
    wb.save(path)


def test_import_counts_decisions(tmp_path: Path, conn: sqlite3.Connection) -> None:
    """Импорт считает решения и долю принятых без правок."""
    path = quality_sample_path(tmp_path, "R01")
    export_quality_sample(conn, "R01", path)
    _decide(path, ["принято", "принято", "исправлено", "отклонено"])

    result = import_quality_sample(conn, path)
    assert (result.approved, result.edited, result.rejected) == (2, 1, 1)
    assert result.skipped >= 1
    assert result.approved_share == pytest.approx(0.5)


def test_import_rejects_unknown_decision(tmp_path: Path, conn: sqlite3.Connection) -> None:
    """Неизвестное решение — ошибка, а не тихий пропуск."""
    path = quality_sample_path(tmp_path, "R01")
    export_quality_sample(conn, "R01", path)
    _decide(path, ["может быть"])

    with pytest.raises(ValueError, match="Неизвестное решение"):
        import_quality_sample(conn, path)


def test_marker_binds_to_file_hash(tmp_path: Path, conn: sqlite3.Connection) -> None:
    """Маркер действует только для той выгрузки, к которой привязан."""
    path = quality_sample_path(tmp_path, "R01")
    export_quality_sample(conn, "R01", path)
    assert not marker_matches(path)

    write_marker(path)
    assert marker_matches(path)
    assert marker_path(path).read_text(encoding="utf-8").strip() == file_hash(path)

    export_quality_sample(conn, "R01", path, seed=1)
    assert not marker_matches(path)


def test_marker_without_sample_does_not_apply(tmp_path: Path) -> None:
    """Маркер без выгрузки не действует."""
    path = quality_sample_path(tmp_path, "R01")
    marker_path(path).write_text("что-то\n", encoding="utf-8")
    assert not marker_matches(path)


def test_decisions_map_is_russian() -> None:
    """Решения в файле — русские слова: коллега заполняет таблицу вручную."""
    assert set(DECISIONS) == {"принято", "исправлено", "отклонено"}
