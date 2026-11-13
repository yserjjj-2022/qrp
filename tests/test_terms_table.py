"""Тесты таблицы терминов (M8): выгрузка, блок одиночных форм, импорт.

Проверяется то, что видно только в файле: сортировка, отдельный блок форм с
одним вхождением, отсутствие меток говорящих и бэкканалов, пополнение
лексикона решением оператора. Тексты синтетические.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from openpyxl import load_workbook

from qa.pipeline.terms_table import (
    CANONICAL_COLUMN,
    COLUMNS,
    CONTEXT_COLUMN,
    FORM_COLUMN,
    SINGLE_BLOCK_TITLE,
    build_term_rows,
    export_terms_table,
    format_timecode,
    import_terms_table,
)
from tests.fixtures.m8_db import FAKE_SPEAKER_LABEL, M8Spec, TermSpec, UtteranceSpec, build_m8_db


def _spec() -> M8Spec:
    return M8Spec(
        utterances=[
            UtteranceSpec("INTERVIEWER", "Что ИИ мог бы делать сам?", "Что ИИ мог бы делать сам?", 5, "5"),
            UtteranceSpec(
                "RESPONDENT",
                "Комиссию смотрю всегда, комиссию видно сразу в отчёте брокера",
                "Комиссию смотрю всегда, комиссию видно сразу в отчёте брокера.",
                40,
                "5",
            ),
            UtteranceSpec("RESPONDENT", "угу", "угу", 60, backchannel=True),
            UtteranceSpec(
                "RESPONDENT",
                "Про ребалансировку скажу коротко: делаю раз в год по календарю",
                "Про ребалансировку скажу коротко: делаю раз в год по календарю.",
                75,
                "18",
            ),
        ],
        terms=[
            TermSpec("комиссию", 1),
            TermSpec("комиссию", 1),
            TermSpec("ребалансировку", 3),
        ],
    )


def _sheet(path: Path) -> list[tuple[object, ...]]:
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb[wb.sheetnames[0]]
        return [tuple(row) for row in ws.iter_rows(values_only=True)]
    finally:
        wb.close()


def test_duplicates_are_merged(tmp_path: Path) -> None:
    """Одна форма из двух кандидатов даёт одну строку с суммой вхождений."""
    conn = build_m8_db(tmp_path, _spec())
    rows = build_term_rows(conn, "R01")
    forms = [row.form for row in rows]
    assert forms.count("комиссию") == 1
    commission = next(row for row in rows if row.form == "комиссию")
    assert commission.frequency == 2
    conn.close()


def test_export_sorts_and_separates_single_forms(tmp_path: Path) -> None:
    """Частые формы выше, одиночные — под заголовком-разделителем."""
    conn = build_m8_db(tmp_path, _spec())
    result = export_terms_table(conn, "R01", tmp_path / "terms.xlsx")
    conn.close()
    assert result.n_rows == 2
    assert result.n_single == 1

    rows = _sheet(result.path)
    assert rows[0][0] is not None  # инструкция для коллег
    assert tuple(rows[2])[: len(COLUMNS)] == tuple(COLUMNS)
    titles = [row[0] for row in rows]
    assert SINGLE_BLOCK_TITLE in titles
    divider = titles.index(SINGLE_BLOCK_TITLE)
    before = [row[0] for row in rows[3:divider] if row[0]]
    after = [row[0] for row in rows[divider + 1 :] if row[0]]
    assert before == sorted(before, key=lambda value: (-_frequency(rows, value), str(value).casefold()))
    assert after == ["ребалансировку"]


def _frequency(rows: list[tuple[object, ...]], form: object) -> int:
    """Частота формы в выгрузке."""
    position = COLUMNS.index("частота")
    for row in rows:
        if row[0] == form:
            return int(row[position])  # type: ignore[arg-type]
    return 0


def test_export_has_no_speaker_labels_or_backchannels(tmp_path: Path) -> None:
    """В выгрузке нет меток говорящих, а бэкканал не даёт контекста."""
    conn = build_m8_db(tmp_path, _spec())
    result = export_terms_table(conn, "R01", tmp_path / "terms.xlsx")
    conn.close()
    text = " ".join(str(cell) for row in _sheet(result.path) for cell in row if cell is not None)
    assert FAKE_SPEAKER_LABEL not in text
    assert "угу" not in text.lower()


def test_context_uses_role_and_anonymized_text(tmp_path: Path) -> None:
    """Контекст подписан ролью и взят из utterance.text."""
    conn = build_m8_db(tmp_path, _spec())
    result = export_terms_table(conn, "R01", tmp_path / "terms.xlsx")
    conn.close()
    rows = _sheet(result.path)
    position = COLUMNS.index(CONTEXT_COLUMN)
    contexts = [str(row[position]) for row in rows[3:] if row[position]]
    assert contexts
    roles = ("респондент: ", "интервьюер: ", "наблюдатель: ")
    for context in contexts:
        for part in context.split(" | "):
            assert part.startswith(roles)
    assert any(context.startswith("респондент: ") for context in contexts)


def test_import_adds_lexicon_variant_with_review_source(tmp_path: Path) -> None:
    """Заполненная ячейка пополняет лексикон с source='review' и закрывает кандидата."""
    conn = build_m8_db(tmp_path, _spec())
    path = tmp_path / "terms.xlsx"
    export_terms_table(conn, "R01", path)

    wb = load_workbook(path)
    try:
        ws = wb[wb.sheetnames[0]]
        header = next(row for row in ws.iter_rows(values_only=True) if FORM_COLUMN in row)
        form_index = header.index(FORM_COLUMN) + 1
        canonical_index = header.index(CANONICAL_COLUMN) + 1
        for row in ws.iter_rows(min_row=4):
            cell = row[form_index - 1]
            if cell.value == "комиссию":
                row[canonical_index - 1].value = "комиссия"
                break
        wb.save(path)
    finally:
        wb.close()

    result = import_terms_table(conn, path)
    assert result.n_variants == 1
    assert result.variants == (("комиссию", "комиссия"),)

    source = conn.execute("SELECT source FROM lexicon WHERE canonical = 'комиссия'").fetchone()
    assert source is not None and source[0] == "review"
    status = conn.execute("SELECT status FROM new_term WHERE surface_form = 'комиссию'").fetchone()
    assert status is not None and status[0] == "approved"
    conn.close()


def test_import_skips_empty_cells_without_blocking(tmp_path: Path) -> None:
    """Пустая ячейка — решения нет: строка учтена, но лексикон не меняется."""
    conn = build_m8_db(tmp_path, _spec())
    path = tmp_path / "terms.xlsx"
    export_terms_table(conn, "R01", path)
    result = import_terms_table(conn, path)
    assert result.n_rows == 3
    assert result.n_variants == 0
    assert result.n_skipped == 3
    lexicon = int(conn.execute("SELECT COUNT(*) FROM lexicon").fetchone()[0])
    assert lexicon == 1
    conn.close()


def test_import_missing_file_raises(tmp_path: Path) -> None:
    """Отсутствующий файл — явная ошибка."""
    conn = build_m8_db(tmp_path, _spec())
    with pytest.raises(FileNotFoundError):
        import_terms_table(conn, tmp_path / "нет.xlsx")
    conn.close()


def test_export_without_candidates_raises(tmp_path: Path) -> None:
    """Без кандидатов таблица не выгружается: пустой файл коллеге не нужен."""
    conn = build_m8_db(tmp_path, M8Spec(utterances=_spec().utterances))
    with pytest.raises(ValueError, match="нет кандидатов"):
        export_terms_table(conn, "R01", tmp_path / "terms.xlsx")
    conn.close()


def test_unknown_interview_raises(tmp_path: Path) -> None:
    """Интервью вне БД — явная ошибка."""
    conn = build_m8_db(tmp_path, _spec())
    with pytest.raises(ValueError, match="не найдено"):
        build_term_rows(conn, "R99")
    conn.close()


def test_format_timecode() -> None:
    """Тайм-код — минуты и секунды; отсутствие времени даёт пустую строку."""
    assert format_timecode(0) == "00:00"
    assert format_timecode(65) == "01:05"
    assert format_timecode(3600) == "60:00"
    assert format_timecode(None) == ""


def test_context_does_not_leak_turn_table(tmp_path: Path) -> None:
    """Контекст строится из utterance, а не из turn с метками."""
    conn = build_m8_db(tmp_path, _spec())
    labels = int(conn.execute("SELECT COUNT(*) FROM turn WHERE speaker_label = ?", (FAKE_SPEAKER_LABEL,)).fetchone()[0])
    assert labels == len(_spec().utterances)
    rows = build_term_rows(conn, "R01")
    assert all(FAKE_SPEAKER_LABEL not in row.context for row in rows)
    conn.close()


def test_term_without_context_utterance(tmp_path: Path) -> None:
    """Кандидат без реплики-контекста не ломает выгрузку."""
    spec = _spec()
    spec.terms = [TermSpec("пустая", None)]
    conn: sqlite3.Connection = build_m8_db(tmp_path, spec)
    rows = build_term_rows(conn, "R01")
    assert rows[0].context == ""
    assert rows[0].frequency == 0
    conn.close()
