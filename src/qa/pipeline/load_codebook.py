"""Загрузка кодбука из рабочей книги: обязательные листы, Темы, Словарь_замен.

Правила:
- обязательные листы «Вопросы_и_коды», «Коды», «Атрибуты»; отсутствие листа или
  колонки — ошибка с названием листа и номером строки;
- коды уникальны, у каждого кода указана семья, перекрёстные ссылки валидны;
- лист «Темы» используется, если непустой, иначе дерево строится по семьям;
- повторная загрузка того же файла не создаёт дублей.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qa.db.workbook_repo import CodebookRepository, ThemeRepository, WorkbookRepository
from qa.pipeline.workbook import SheetData, WorkbookData, WorkbookError, is_identifier_like, read_workbook
from qa.settings import GroupsSettings, WorkbookSettings

logger = logging.getLogger(__name__)

SHEET_QUESTIONS = "Вопросы_и_коды"
SHEET_CODES = "Коды"
SHEET_ATTRIBUTES = "Атрибуты"
SHEET_THEMES = "Темы"
SHEET_GLOSSARY = "Словарь_замен"


class CodebookValidationError(WorkbookError):
    """Нарушение перекрёстных ссылок или обязательных значений кодбука."""


@dataclass
class LoadCodebookResult:
    """Итог загрузки кодбука."""

    codebook_version_id: int
    workbook_version_id: int
    created: bool
    counts: dict[str, int] = field(default_factory=dict)
    tree: list[tuple[str, list[tuple[str, list[str]]]]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _text(row: dict[str, Any], key: str) -> str | None:
    """Строковое значение поля строки."""
    value = row.get(key)
    return None if value is None else str(value)


def _identifiers(row: dict[str, Any], field_name: str) -> list[str]:
    """Идентификаторы кодов/вопросов из списка строки (без служебных пометок)."""
    value = row.get(field_name)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if is_identifier_like(str(item))]


def _require_unique(values: list[str], sheet: str, what: str) -> None:
    """Проверить уникальность значений; ошибка с листом и повторами."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        msg = f"Лист '{sheet}': повторяющиеся {what}: {sorted(duplicates)}"
        raise CodebookValidationError(msg)


def _cross_check(
    codes: set[str],
    questions: set[str],
    code_rows: list[dict[str, Any]],
    question_rows: list[dict[str, Any]],
    theme_rows: list[dict[str, Any]],
    workbook: WorkbookData,
) -> None:
    """Перекрёстные ссылки: коды у вопросов, вопросы у кодов, коды в темах."""
    missing_code_links: list[str] = []
    for row, row_number in zip(question_rows, workbook.sheets[SHEET_QUESTIONS].row_numbers, strict=True):
        for item in _identifiers(row, "codes"):
            if item not in codes:
                missing_code_links.append(f"строка {row_number}: код {item!r}")
    if missing_code_links:
        msg = f"Лист '{SHEET_QUESTIONS}': нет таких кодов в листе «Коды»: {missing_code_links}"
        raise WorkbookError(msg)

    missing_question_links: list[str] = []
    for row, row_number in zip(code_rows, workbook.sheets[SHEET_CODES].row_numbers, strict=True):
        for item in _identifiers(row, "questions"):
            if item not in questions:
                missing_question_links.append(f"строка {row_number}: вопрос {item!r}")
    if missing_question_links:
        msg = f"Лист '{SHEET_CODES}': нет таких вопросов в листе «Вопросы_и_коды»: {missing_question_links}"
        raise WorkbookError(msg)

    missing_theme_codes: list[str] = []
    themes_sheet = workbook.sheets.get(SHEET_THEMES)
    row_numbers = themes_sheet.row_numbers if themes_sheet is not None else []
    for row, row_number in zip(theme_rows, row_numbers, strict=True):
        for item in _identifiers(row, "codes"):
            if item not in codes:
                missing_theme_codes.append(f"строка {row_number}: код {item!r}")
    if missing_theme_codes:
        msg = f"Лист '{SHEET_THEMES}': коды не найдены в листе «Коды»: {missing_theme_codes}"
        raise WorkbookError(msg)


def _validate_codebook_rows(
    codes_sheet: SheetData,
    questions_sheet: SheetData,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Проверить обязательные значения строк кодбука."""
    code_rows = codes_sheet.rows
    question_rows = questions_sheet.rows
    for row, row_number in zip(code_rows, codes_sheet.row_numbers, strict=True):
        if _text(row, "code") is None:
            msg = f"Лист '{SHEET_CODES}', строка {row_number}: пустой код"
            raise CodebookValidationError(msg)
        if _text(row, "family") is None:
            msg = f"Лист '{SHEET_CODES}', строка {row_number}: у кода {_text(row, 'code')!r} не указана семья"
            raise CodebookValidationError(msg)
    for row, row_number in zip(question_rows, questions_sheet.row_numbers, strict=True):
        if _text(row, "question_id") is None:
            msg = f"Лист '{SHEET_QUESTIONS}', строка {row_number}: пустой идентификатор вопроса"
            raise CodebookValidationError(msg)
    _require_unique([str(_text(row, "code")) for row in code_rows], SHEET_CODES, "коды")
    _require_unique([str(_text(row, "question_id")) for row in question_rows], SHEET_QUESTIONS, "вопросы")
    return code_rows, question_rows


def _require_attributes(attributes_sheet: SheetData | None, required: list[str]) -> None:
    """Проверить наличие обязательных атрибутов разметки в листе «Атрибуты».

    Имена атрибутов задаёт оператор в книге; список обязательных — в
    config.yaml (`workbook.required_attributes`). В коде имён нет: иначе
    переименование в книге молча не совпало бы с проверкой.
    """
    if not required:
        return
    names = {
        str(row.get("name")).strip()
        for row in (attributes_sheet.rows if attributes_sheet is not None else [])
        if row.get("name") is not None
    }
    missing = [name for name in required if name not in names]
    if not missing:
        return
    found = ", ".join(sorted(names)) or "лист «Атрибуты» пуст"
    msg = (
        f"Лист '{SHEET_ATTRIBUTES}': нет обязательных атрибутов {', '.join(missing)}. "
        f"Переименуйте поле в книге или добавьте его; сейчас есть: {found}"
    )
    raise CodebookValidationError(msg)


def _build_tree_from_themes(
    theme_rows: list[dict[str, Any]],
    code_ids: dict[str, int],
    themes: ThemeRepository,
) -> list[str]:
    """Собрать дерево из листа «Темы»: узел -> тема -> коды.

    Пустая тема и тема без кодов — заметки, а не ошибка: дерево достраивается
    оператором, а загрузка кодбука не должна останавливаться.
    """
    notes: list[str] = []
    nodes: dict[str, int] = {}
    for row in theme_rows:
        node_name = _text(row, "node")
        theme_name = _text(row, "theme")
        if node_name is None:
            notes.append("Лист 'Темы': строка без узла пропущена")
            continue
        node_id = nodes.get(node_name)
        if node_id is None:
            node_id = themes.insert_node(node_name)
            nodes[node_name] = node_id
        if theme_name is None:
            notes.append(f"Лист 'Темы': узел '{node_name}' без темы — узел создан пустым")
            continue
        theme_id = themes.insert_theme(theme_name, node_id)
        bound = 0
        for code in _identifiers(row, "codes"):
            code_id = code_ids.get(code)
            if code_id is not None:
                themes.bind(theme_id, code_id)
                bound += 1
        if bound == 0:
            notes.append(f"Лист 'Темы': тема '{theme_name}' (узел '{node_name}') без кодов")
    return notes


def _build_tree_from_families(
    code_rows: list[dict[str, Any]],
    code_ids: dict[str, int],
    family_nodes: dict[str, str],
    themes: ThemeRepository,
) -> list[str]:
    """Собрать дерево по семьям кодов с маппингом «семья -> узел»."""
    unknown_families: list[str] = []
    structure: dict[str, dict[str, list[str]]] = {}
    for row in code_rows:
        code = _text(row, "code")
        family = _text(row, "family")
        if code is None or family is None:
            continue
        node_name = family_nodes.get(family)
        if node_name is None:
            unknown_families.append(family)
            node_name = family
        structure.setdefault(node_name, {}).setdefault(family, []).append(code)
    for node_name, families in structure.items():
        node_id = themes.insert_node(node_name)
        for family, codes in families.items():
            theme_id = themes.insert_theme(family, node_id)
            for code in codes:
                code_id = code_ids.get(code)
                if code_id is not None:
                    themes.bind(theme_id, code_id)
    return sorted(set(unknown_families))


def _insert_codebook(
    workbook: WorkbookData,
    codebook_version_id: int,
    code_rows: list[dict[str, Any]],
    question_rows: list[dict[str, Any]],
    codebook_repo: CodebookRepository,
) -> None:
    """Записать коды, вопросы, атрибуты, связи и словарь замен."""
    codes_sheet = workbook.sheets[SHEET_CODES]
    questions_sheet = workbook.sheets[SHEET_QUESTIONS]

    code_ids = {str(_text(row, "code")): codebook_repo.insert_code(codebook_version_id, row) for row in code_rows}
    question_ids = {
        str(_text(row, "question_id")): codebook_repo.insert_question(codebook_version_id, row) for row in question_rows
    }
    for row in workbook.sheets[SHEET_ATTRIBUTES].rows:
        codebook_repo.insert_attribute_definition(codebook_version_id, row)

    for row in question_rows:
        question_db_id = question_ids[str(_text(row, "question_id"))]
        for code in _identifiers(row, "codes"):
            codebook_repo.link_question_code(question_db_id, code_ids[code])
    for row in code_rows:
        code_db_id = code_ids[str(_text(row, "code"))]
        for question in _identifiers(row, "questions"):
            codebook_repo.link_question_code(question_ids[question], code_db_id)

    glossary_sheet = workbook.sheet(SHEET_GLOSSARY)
    for row in glossary_sheet.rows if glossary_sheet is not None else []:
        term = _text(row, "term")
        if term is None:
            continue
        for single in term.split(","):
            item = single.strip()
            if item:
                codebook_repo.insert_glossary_term(codebook_version_id, item, _text(row, "replacement"))
    logger.debug(
        "Записаны листы %s и %s (коды=%s, вопросы=%s)",
        codes_sheet.name,
        questions_sheet.name,
        len(code_rows),
        len(question_rows),
    )


def load_codebook(
    conn: sqlite3.Connection,
    workbook_path: Path,
    workbook_settings: WorkbookSettings,
    *,
    reload: bool = False,
    groups: GroupsSettings | None = None,
) -> LoadCodebookResult:
    """Загрузить обязательные листы книги, темы и словарь замен.

    Args:
        conn: подключение к SQLite (миграции уже применены).
        workbook_path: путь к .xlsx.
        workbook_settings: классы листов и схемы колонок из config.yaml.
        reload: пересобрать темы, даже если версия уже загружена.
        groups: канонические значения групп для нормализации листа «Вопросы_и_коды».

    Raises:
        FileNotFoundError: файл не найден.
        WorkbookError: нет обязательного листа или колонки.
        CodebookValidationError: нарушены обязательные значения или ссылки.
    """
    workbook = read_workbook(workbook_path, workbook_settings)
    for warning in workbook.warnings:
        if warning.startswith("Неизвестный лист"):
            logger.warning(warning)

    # Обязательные атрибуты проверяются всегда, в том числе при повторной
    # загрузке: книга могла быть переименована, а версия уже записана.
    _require_attributes(workbook.sheet(SHEET_ATTRIBUTES), workbook_settings.required_attributes)

    workbook_repo = WorkbookRepository(conn)
    codebook_repo = CodebookRepository(conn, groups)
    themes_repo = ThemeRepository(conn)

    existing_workbook = workbook_repo.find_by_hash(workbook.file_hash)
    existing_codebook = codebook_repo.find_by_codebook_hash(workbook.codebook_hash)

    if existing_workbook is not None and existing_codebook is not None and not reload:
        logger.info("Книга уже загружена (hash=%s), дублей нет", workbook.file_hash[:12])
        return LoadCodebookResult(
            codebook_version_id=existing_codebook,
            workbook_version_id=existing_workbook,
            created=False,
            counts=codebook_repo.counts(existing_codebook),
            tree=themes_repo.tree(),
            notes=workbook.warnings,
        )

    codes_sheet = workbook.sheets[SHEET_CODES]
    questions_sheet = workbook.sheets[SHEET_QUESTIONS]
    code_rows, question_rows = _validate_codebook_rows(codes_sheet, questions_sheet)

    themes_sheet = workbook.sheet(SHEET_THEMES)
    theme_rows = themes_sheet.rows if themes_sheet is not None else []
    _cross_check(
        {str(_text(row, "code")) for row in code_rows},
        {str(_text(row, "question_id")) for row in question_rows},
        code_rows,
        question_rows,
        theme_rows,
        workbook,
    )

    workbook_version_id = existing_workbook if existing_workbook is not None else workbook_repo.insert(workbook)
    if existing_codebook is not None:
        codebook_version_id = existing_codebook
    else:
        codebook_version_id = codebook_repo.insert_version(workbook_version_id, workbook.codebook_hash)
        _insert_codebook(workbook, codebook_version_id, code_rows, question_rows, codebook_repo)
    code_ids = codebook_repo.code_ids(codebook_version_id)

    themes_repo.clear()
    notes: list[str] = []
    if theme_rows:
        notes.extend(_build_tree_from_themes(theme_rows, code_ids, themes_repo))
    else:
        unknown = _build_tree_from_families(code_rows, code_ids, workbook_settings.family_nodes, themes_repo)
        notes = [f"Семья без маппинга в узел: {family}" for family in unknown]
    for note in notes:
        logger.warning(note)

    conn.commit()
    counts = codebook_repo.counts(codebook_version_id)
    logger.info(
        "Кодбук загружен: version_id=%s, codes=%s, questions=%s, links=%s",
        codebook_version_id,
        counts["codes"],
        counts["questions"],
        counts["question_code"],
    )
    return LoadCodebookResult(
        codebook_version_id=codebook_version_id,
        workbook_version_id=workbook_version_id,
        created=existing_codebook is None,
        counts=counts,
        tree=themes_repo.tree(),
        notes=workbook.warnings + notes,
    )
