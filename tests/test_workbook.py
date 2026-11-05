"""Тесты чтения рабочей книги: списки, диапазоны, классы листов, хеши, ошибки."""

from __future__ import annotations

from pathlib import Path

import pytest

from qa.pipeline.workbook import (
    WorkbookError,
    is_identifier_like,
    normalize_cell,
    parse_list,
    read_workbook,
)
from qa.settings import Settings
from tests.fixtures.workbook_factory import build_workbook

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def settings() -> Settings:
    return Settings.load(_PROJECT_ROOT / "config.yaml")


class TestParseList:
    """Разбор списков в ячейках."""

    def test_comma_separated(self) -> None:
        assert parse_list("8, 9, 12Т, 17") == ["8", "9", "12Т", "17"]

    def test_semicolon_separated(self) -> None:
        assert parse_list("R1; R2") == ["R1", "R2"]

    @pytest.mark.parametrize(
        ("cell", "expected"),
        [
            ("R1–R8", [f"R{n}" for n in range(1, 9)]),
            ("E0–E7", [f"E{n}" for n in range(0, 8)]),
            ("A1–A8", [f"A{n}" for n in range(1, 9)]),
            ("N1–N4", [f"N{n}" for n in range(1, 5)]),
            ("C1–C4", [f"C{n}" for n in range(1, 5)]),
        ],
    )
    def test_ranges_expanded(self, cell: str, expected: list[str]) -> None:
        assert parse_list(cell) == expected

    def test_range_with_hyphen_variants(self) -> None:
        assert parse_list("R1-R3") == ["R1", "R2", "R3"]
        assert parse_list("R1—R2") == ["R1", "R2"]

    def test_mixed_list_with_range(self) -> None:
        assert parse_list("R1–R2, индуктивно") == ["R1", "R2", "индуктивно"]

    def test_parentheses_not_split(self) -> None:
        assert parse_list("индуктивно (влияние, доверие); связь с A") == [
            "индуктивно (влияние, доверие)",
            "связь с A",
        ]

    def test_empty(self) -> None:
        assert parse_list(None) == []
        assert parse_list("   ") == []

    def test_broken_range_kept_as_is(self) -> None:
        assert parse_list("R8–R1") == ["R8–R1"]

    def test_normalize_cell_numbers(self) -> None:
        assert normalize_cell(7.0) == "7"
        assert normalize_cell(" 7Т ") == "7Т"
        assert normalize_cell(None) is None
        assert normalize_cell("") is None

    def test_identifier_like(self) -> None:
        assert is_identifier_like("R1")
        assert is_identifier_like("12Т")
        assert is_identifier_like("14Т-б")
        assert not is_identifier_like("индуктивно")
        assert not is_identifier_like("атрибут: стаж")
        assert not is_identifier_like("оценка понимания")


class TestReadWorkbook:
    """Классы листов, обязательные колонки, хеши."""

    def test_reads_required_and_optional(self, tmp_path: Path, settings: Settings) -> None:
        path = build_workbook(tmp_path / "book.xlsx")
        workbook = read_workbook(path, settings.workbook)
        assert "Вопросы_и_коды" in workbook.sheets
        assert "Коды" in workbook.sheets
        assert "Атрибуты" in workbook.sheets
        assert "Темы" in workbook.sheets
        assert len(workbook.sheets["Коды"].rows) == 10
        assert len(workbook.sheets["Вопросы_и_коды"].rows) == 8
        assert workbook.sheets["Вопросы_и_коды"].row_numbers[0] == 2

    def test_question_ids_stay_strings(self, tmp_path: Path, settings: Settings) -> None:
        path = build_workbook(tmp_path / "book.xlsx")
        workbook = read_workbook(path, settings.workbook)
        ids = [row["question_id"] for row in workbook.sheets["Вопросы_и_коды"].rows]
        assert "14Т-а" in ids
        assert all(isinstance(value, str) for value in ids)

    def test_empty_cells_are_none(self, tmp_path: Path, settings: Settings) -> None:
        workbook = read_workbook(build_workbook(tmp_path / "book.xlsx"), settings.workbook)
        first = workbook.sheets["Вопросы_и_коды"].rows[0]
        assert first["notes"] == "—"
        codes_row = workbook.sheets["Коды"].rows[1]
        assert codes_row["tiebreak_rule"] == "—"

    def test_ignored_sheets_skipped(self, tmp_path: Path, settings: Settings) -> None:
        path = build_workbook(
            tmp_path / "book.xlsx",
            extra_sheets={"Справка": [["лист", "назначение"]], "_Служебный": [["a"]]},
        )
        workbook = read_workbook(path, settings.workbook)
        assert "Справка" not in workbook.sheets
        assert "_Служебный" not in workbook.sheets

    def test_unknown_sheet_warns_not_errors(self, tmp_path: Path, settings: Settings) -> None:
        path = build_workbook(tmp_path / "book.xlsx", extra_sheets={"Что_то_новое": [["a"]]})
        workbook = read_workbook(path, settings.workbook)
        assert "Что_то_новое" not in workbook.sheets
        assert any("Неизвестный лист" in warning for warning in workbook.warnings)

    def test_facts_prefix_loaded(self, tmp_path: Path, settings: Settings) -> None:
        path = build_workbook(
            tmp_path / "book.xlsx",
            extra_sheets={"Факты_Финам": [["F1", "Права", "только чтение", None, "черновик"]]},
        )
        workbook = read_workbook(path, settings.workbook)
        assert len(workbook.sheets["Факты_Финам"].rows) == 1

    def test_missing_required_sheet(self, tmp_path: Path, settings: Settings) -> None:
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        assert ws is not None
        ws.title = "Коды"
        ws.append(["код", "семья"])
        ws.append(["R1", "Риски"])
        path = tmp_path / "broken.xlsx"
        wb.save(path)
        with pytest.raises(WorkbookError, match="отсутствуют обязательные листы"):
            read_workbook(path, settings.workbook)

    def test_missing_required_column(self, tmp_path: Path, settings: Settings) -> None:
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        assert ws is not None
        ws.title = "Коды"
        ws.append(["код", "название"])  # нет «семья»
        ws.append(["R1", "Ошибка"])
        wb.create_sheet("Вопросы_и_коды").append(["id_вопроса"])
        wb.create_sheet("Атрибуты").append(["поле"])
        path = tmp_path / "broken.xlsx"
        wb.save(path)
        with pytest.raises(WorkbookError, match="не найдены обязательные колонки"):
            read_workbook(path, settings.workbook)

    def test_file_not_found(self, tmp_path: Path, settings: Settings) -> None:
        with pytest.raises(FileNotFoundError):
            read_workbook(tmp_path / "none.xlsx", settings.workbook)

    def test_hashes_stable_and_sensitive(self, tmp_path: Path, settings: Settings) -> None:
        path = build_workbook(tmp_path / "book.xlsx")
        first = read_workbook(path, settings.workbook)
        second = read_workbook(path, settings.workbook)
        assert first.file_hash == second.file_hash
        assert first.codebook_hash == second.codebook_hash
        assert set(first.sheet_hashes) >= {"Коды", "Вопросы_и_коды", "Атрибуты"}

    def test_codebook_hash_ignores_optional_sheets(self, tmp_path: Path, settings: Settings) -> None:
        base = build_workbook(tmp_path / "a.xlsx", with_themes=False, with_optional=False)
        extended = build_workbook(
            tmp_path / "b.xlsx",
            with_themes=False,
            with_optional=True,
        )
        first = read_workbook(base, settings.workbook)
        second = read_workbook(extended, settings.workbook)
        assert first.codebook_hash == second.codebook_hash
        assert first.file_hash != second.file_hash
