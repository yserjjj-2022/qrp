"""Тесты загрузчика кодбука."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest
from openpyxl import Workbook

from qa.db.connection import get_connection
from qa.db.migrations import migrate
from qa.pipeline.load_codebook import load_codebook

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations"

# Имя переменной окружения для пути к реальному Приложению Б
_REAL_CODEBOOK_ENV = "QA_REAL_CODEBOOK_PATH"


@pytest.fixture()
def db(tmp_path: Path) -> sqlite3.Connection:
    """Подключение к временной БД с применёнными миграциями."""
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    migrate(conn, db_path, _MIGRATIONS_DIR, tmp_path / "backups")
    yield conn
    conn.close()


def _make_synthetic_excel(path: Path, n_codes: int = 5) -> Path:
    """Создать минимальный Excel с той же структурой, что и Приложение Б."""
    wb = Workbook()

    # Лист Коды
    ws_codes = wb.active
    ws_codes.title = "Коды"
    ws_codes.append(["Код", "Семейство", "Название", "Определение", "Включаем", "Исключаем", "Привязка к вопросу"])
    for i in range(1, n_codes + 1):
        ws_codes.append([f"R{i}", "риски", f"Риск {i}", f"Описание риска {i}", "вкл", "искл", "1"])

    # Лист Вопросы_и_коды
    ws_q = wb.create_sheet("Вопросы_и_коды")
    ws_q.append(["Код", "Идентификатор вопроса"])
    for i in range(1, n_codes + 1):
        ws_q.append([f"R{i}", f"Q{i}"])

    # Лист Атрибуты
    ws_attr = wb.create_sheet("Атрибуты")
    ws_attr.append(["Код", "Имя атрибута", "Значение"])
    for i in range(1, n_codes + 1):
        ws_attr.append([f"R{i}", "severity", f"level_{i}"])

    wb.save(path)
    return path


class TestLoadCodebook:
    """Проверка загрузки кодбука."""

    def test_synthetic(self, db: sqlite3.Connection, tmp_path: Path) -> None:
        """Синтетический Excel: корректная загрузка."""
        n_codes = 5
        xlsx_path = _make_synthetic_excel(tmp_path / "codebook.xlsx", n_codes)
        codes, questions, attrs = load_codebook(db, xlsx_path)
        assert codes == n_codes
        assert questions == n_codes
        assert attrs == n_codes

    def test_no_duplicates(self, db: sqlite3.Connection, tmp_path: Path) -> None:
        """Повторная загрузка того же файла не создаёт дублей."""
        xlsx_path = _make_synthetic_excel(tmp_path / "codebook.xlsx", 3)
        load_codebook(db, xlsx_path)
        codes2, questions2, attrs2 = load_codebook(db, xlsx_path)
        assert codes2 == 3
        assert questions2 == 3
        assert attrs2 == 3

    def test_missing_sheet(self, db: sqlite3.Connection, tmp_path: Path) -> None:
        """Отсутствие обязательного листа — понятная ошибка."""
        wb = Workbook()
        ws = wb.active
        ws.title = "Коды"
        ws.append(["Код", "Семейство", "Название", "Определение", "Включаем", "Исключаем", "Привязка к вопросу"])
        path = tmp_path / "broken.xlsx"
        wb.save(path)
        with pytest.raises(ValueError, match="отсутствует обязательный лист"):
            load_codebook(db, path)

    def test_missing_column(self, db: sqlite3.Connection, tmp_path: Path) -> None:
        """Отсутствие обязательной колонки — понятная ошибка."""
        wb = Workbook()
        ws = wb.active
        ws.title = "Коды"
        ws.append(["Код", "Название"])  # missing columns
        wb.create_sheet("Вопросы_и_коды").append(["Код", "Идентификатор вопроса"])
        wb.create_sheet("Атрибуты").append(["Код", "Имя атрибута", "Значение"])
        path = tmp_path / "broken.xlsx"
        wb.save(path)
        with pytest.raises(ValueError, match="не найдена колонка"):
            load_codebook(db, path)

    def test_file_not_found(self, db: sqlite3.Connection) -> None:
        """Несуществующий файл — FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_codebook(db, Path("/nonexistent/file.xlsx"))


@pytest.mark.real_codebook
class TestRealCodebook:
    """Тест на настоящем Приложении Б (42 кода). Пропускается без env-переменной."""

    def test_42_codes(self, db: sqlite3.Connection) -> None:
        real_path = os.environ.get(_REAL_CODEBOOK_ENV)
        if not real_path:
            pytest.skip(f"Переменная {_REAL_CODEBOOK_ENV} не задана")
        path = Path(real_path)
        if not path.exists():
            pytest.skip(f"Файл не найден: {path}")
        codes, questions, attrs = load_codebook(db, path)
        assert codes == 42, f"Ожидалось 42 кода, получено {codes}"
