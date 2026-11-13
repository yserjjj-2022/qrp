"""Тесты разовых правок «в этой реплике имелось в виду Z» (M8).

Правка применяется на границе L1 -> L2: проверяется разбор файла, замена по
вхождению и отказ от молчаливой «правки в пустоту».
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qa.pipeline.manual_edits import (
    ManualEditsError,
    load_manual_edits,
    manual_edits_path,
)


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_missing_file_is_empty() -> None:
    """Отсутствие файла — норма: правок нет, ошибки нет."""
    edits = load_manual_edits(Path("/tmp/нет-такого-файла-qrp.json"))
    assert edits.count == 0
    assert edits.file_hash == ""


def test_apply_replaces_inside_utterance(tmp_path: Path) -> None:
    """Замена идёт только в своей реплике и не зависит от регистра."""
    path = _write(
        tmp_path / "edits.json",
        {"edits": [{"utterance_id": 7, "from": "сенжин", "to": "S&P 500"}]},
    )
    edits = load_manual_edits(path)

    text, applied = edits.apply(7, "Смотрю на Сенжин и ещё раз на сенжин")
    assert text == "Смотрю на S&P 500 и ещё раз на S&P 500"
    assert len(applied) == 1
    assert applied[0].target == "S&P 500"


def test_apply_leaves_other_utterances(tmp_path: Path) -> None:
    """Правка не действует на другие реплики: она разовая."""
    path = _write(tmp_path / "edits.json", {"edits": [{"utterance_id": 7, "from": "а", "to": "б"}]})
    edits = load_manual_edits(path)

    text, applied = edits.apply(8, "а")
    assert (text, applied) == ("а", ())


def test_missing_source_is_not_applied(tmp_path: Path) -> None:
    """Не найденная исходная форма не попадает в журнал применённых правок."""
    path = _write(tmp_path / "edits.json", {"edits": [{"utterance_id": 7, "from": "нет такого", "to": "x"}]})
    edits = load_manual_edits(path)

    text, applied = edits.apply(7, "совсем другой текст")
    assert text == "совсем другой текст"
    assert applied == ()


def test_hash_changes_with_content(tmp_path: Path) -> None:
    """Хеш файла меняется вместе с правками: прогон видит другое состояние."""
    first = load_manual_edits(_write(tmp_path / "a.json", {"edits": [{"utterance_id": 1, "from": "a", "to": "b"}]}))
    second = load_manual_edits(_write(tmp_path / "b.json", {"edits": [{"utterance_id": 1, "from": "a", "to": "c"}]}))
    assert first.file_hash and second.file_hash
    assert first.file_hash != second.file_hash


def test_manual_edits_path_is_next_to_exports(tmp_path: Path) -> None:
    """Файл правок лежит рядом с выгрузками и назван по коду интервью."""
    assert manual_edits_path(tmp_path, "R01") == tmp_path / "quote_edits_R01.json"


@pytest.mark.parametrize(
    "payload",
    [
        ["не объект"],
        {"edits": "не массив"},
        {"edits": [{"utterance_id": 1, "from": "", "to": "b"}]},
        {"edits": [{"utterance_id": "abc", "from": "a", "to": "b"}]},
        {"edits": [{"from": "a", "to": "b"}]},
    ],
)
def test_invalid_payload_is_reported(tmp_path: Path, payload: object) -> None:
    """Некорректный файл даёт понятную ошибку, а не тихий пропуск."""
    path = _write(tmp_path / "bad.json", payload)
    with pytest.raises(ManualEditsError):
        load_manual_edits(path)


def test_broken_json_is_reported(tmp_path: Path) -> None:
    """Битый JSON — ошибка формата."""
    path = tmp_path / "broken.json"
    path.write_text("{не json", encoding="utf-8")
    with pytest.raises(ManualEditsError):
        load_manual_edits(path)


def test_normalize_applies_edit_before_model(tmp_path: Path) -> None:
    """Нормализация видит исправленный текст, L1 остаётся исходным."""
    from qa.db.workbook_repo import RunRepository
    from qa.pipeline.normalize import normalize_interview
    from qa.settings import Settings
    from tests.fixtures.m8_db import M8Spec, UtteranceSpec, build_m8_db

    project_root = Path(__file__).resolve().parents[1]
    settings = Settings.load(project_root / "config.yaml")
    settings.paths.exports = str(tmp_path / "exports")

    conn = build_m8_db(
        tmp_path,
        M8Spec(
            utterances=[
                UtteranceSpec(
                    "RESPONDENT",
                    "Смотрю на сенжин каждый день и всё устраивает",
                    "Смотрю на сенжин каждый день и всё устраивает",
                    30,
                    "5",
                )
            ]
        ),
    )
    conn.execute("DELETE FROM annotation")
    conn.execute("DELETE FROM fragment_coding")
    conn.execute("DELETE FROM fragment")
    conn.execute("DELETE FROM text_unit")
    conn.commit()
    edits = load_manual_edits(
        _write(tmp_path / "edits.json", {"edits": [{"utterance_id": 1, "from": "сенжин", "to": "S&P 500"}]})
    )
    try:
        result = normalize_interview(conn, "R01", settings, None, manual_edits=edits)
        text = conn.execute("SELECT text FROM text_unit").fetchone()[0]
        l1 = conn.execute("SELECT text FROM utterance").fetchone()[0]
        params = RunRepository(conn).params(int(result.run_id or 0))
    finally:
        conn.close()

    assert "S&P 500" in text
    assert "сенжин" in l1
    assert params["manual_edits"] == 1
    assert params["manual_edits_hash"] == edits.file_hash
