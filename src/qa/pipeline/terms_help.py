"""Таблица терминов проекта и применение решений оператора (M7b).

Один файл на проект (`data/exports/terms_<проект>.xlsx`) вместо таблицы на
интервью: оператор опознаёт слово один раз, и решение работает на всех
следующих интервью. Лист «Нужна помощь» показывает формы без решения, лист
«Применено само» — что система применила без оператора.

`apply-terms` пишет решения в лексикон и детерминированно пересобирает L2:
модель не вызывается, повторный запуск ничего не меняет. Если у интервью уже
есть аннотации, пересборка запрещена (заморозка): менять текст под цитатами
нельзя.

SQL здесь нет: состояние и записи идут через репозитории (CONSTITUTION §4).
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openpyxl import Workbook, load_workbook

from qa.db.pipeline_repo import PipelineRepository
from qa.db.workbook_repo import LexiconRepository, RunRepository, TextUnitRepository
from qa.pipeline.normalize_rules import LexiconEntry, apply_lexicon, suspect_words
from qa.pipeline.terms_memory import KEEP, UNREADABLE, TermHint, apply_choice
from qa.pipeline.terms_table import format_timecode
from qa.settings import Settings

logger = logging.getLogger(__name__)

SHEET_HELP = "Нужна помощь"
SHEET_AUTO = "Применено само"
HELP_COLUMNS = ("Форма", "Где встречается", "Контекст", "Подсказки", "ЗАМЕНА", "Пояснение")
AUTO_COLUMNS = ("Форма", "Замена", "Основание", "Где встречается")
FORM_COLUMN = "Форма"
PLACES_COLUMN = "Где встречается"
CONTEXT_COLUMN = "Контекст"
HINTS_COLUMN = "Подсказки"
CHOICE_COLUMN = "ЗАМЕНА"
NOTE_COLUMN = "Пояснение"
REPLACEMENT_COLUMN = "Замена"
BASIS_COLUMN = "Основание"

HELP_INSTRUCTION = (
    "Опознайте слова, которых нет в словаре и лексиконе. Пусто — принять первую "
    "подсказку выше порога; = — оставить как есть; ? — [неразборчиво]; текст — замена. "
    "После заполнения: qa-pipeline apply-terms."
)
MAX_CONTEXT_CHARS = 200
_WORD_RE = re.compile(r"[а-яёa-z]+(?:-[а-яёa-z]+)?", re.IGNORECASE)


class TermsTableError(Exception):
    """Таблица терминов недоступна или не содержит шапки."""


@dataclass(frozen=True)
class OpenForm:
    """Форма без решения: что показать оператору."""

    form: str
    frequency: int
    places: str
    context: str


@dataclass
class HelpRow:
    """Строка листа «Нужна помощь»."""

    form: str
    frequency: int = 0
    places: str = ""
    context: str = ""
    hints: tuple[TermHint, ...] = ()
    note: str = ""
    choice: str = ""

    def hints_text(self) -> str:
        """Подсказки одной строкой: `OsEngine (0.89); оставить (0.11)`."""
        return "; ".join(hint.label() for hint in self.hints)


@dataclass(frozen=True)
class ApplyResult:
    """Итог команды `apply-terms`."""

    path: Path
    rows: int = 0
    applied: int = 0
    auto: int = 0
    kept: int = 0
    conflicts: int = 0
    skipped: tuple[str, ...] = ()
    rebuilt: int = 0
    details: tuple[str, ...] = field(default_factory=tuple)


def terms_path(settings: Settings) -> Path:
    """Путь к таблице терминов проекта."""
    return settings.m7b.terms_path(settings.paths.exports)


def _occurrences(text: str, form: str) -> int:
    """Сколько раз форма встречается целым словом."""
    pattern = rf"(?<![а-яёa-z0-9]){re.escape(form.casefold())}(?![а-яёa-z0-9])"
    return len(re.findall(pattern, text.casefold()))


def _context(text: str, form: str) -> str:
    """Фрагмент реплики вокруг формы, не длиннее 200 символов."""
    folded = text.casefold()
    position = folded.find(form.casefold())
    if position < 0:
        return text[:MAX_CONTEXT_CHARS]
    start = max(0, position - MAX_CONTEXT_CHARS // 2)
    end = min(len(text), position + len(form) + MAX_CONTEXT_CHARS // 2)
    fragment = text[start:end].strip()
    return fragment if len(fragment) <= MAX_CONTEXT_CHARS else fragment[:MAX_CONTEXT_CHARS]


def collect_open_forms(
    conn: Any,  # noqa: ANN401 - sqlite3.Connection
    interview_code: str,
    entries: list[LexiconEntry],
) -> list[OpenForm]:
    """Формы актуального L2, которых нет ни в словаре, ни в лексиконе.

    Считаются все реплики интервью, а не только респондента: интервьюер тоже
    произносит термины, и его ошибки видны оператору.
    """
    repo = PipelineRepository(conn)
    interview_db_id = repo.interview_db_id(interview_code)
    if interview_db_id is None:
        return []
    version = repo.max_l2_version(interview_db_id)
    if not version:
        return []
    counts: Counter[str] = Counter()
    places: dict[str, list[str]] = {}
    contexts: dict[str, str] = {}
    for _unit_id, text, timecode in repo.l2_texts_with_timecodes(interview_db_id, version):
        for form in suspect_words(text, entries):
            counts[form] += _occurrences(text, form)
            places.setdefault(form, []).append(f"{interview_code}: {format_timecode(timecode)}")
            contexts.setdefault(form, _context(text, form))
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [
        OpenForm(
            form=form,
            frequency=count,
            places=", ".join(dict.fromkeys(places.get(form, []))),
            context=contexts.get(form, ""),
        )
        for form, count in ordered
    ]


def _find_header(ws: Any, required: tuple[str, ...]) -> tuple[int, dict[str, int]]:  # noqa: ANN401
    """Найти шапку листа по названиям колонок."""
    for row_index, row in enumerate(ws.iter_rows(values_only=True), start=1):
        headers = [str(value).strip() if value is not None else "" for value in row]
        if all(name in headers for name in required):
            return row_index, {name: headers.index(name) for name in headers if name}
        if row_index > 10:
            break
    msg = f"Нет шапки с колонками {required!r}"
    raise TermsTableError(msg)


def _cell(row: tuple[Any, ...], index: dict[str, int], name: str) -> str:
    """Значение ячейки по имени колонки."""
    position = index.get(name)
    if position is None or position >= len(row) or row[position] is None:
        return ""
    return str(row[position]).strip()


_HINT_RE = re.compile(r"^(?P<to>.+) \((?P<confidence>\d+(?:\.\d+)?)\)$")


def parse_hints(text: str) -> tuple[TermHint, ...]:
    """Восстановить подсказки из ячейки таблицы.

    Повторное применение таблицы не должно терять подсказки: они уже записаны
    в файл, и вызывать модель заново незачем.
    """
    hints: list[TermHint] = []
    for part in text.split(";"):
        match = _HINT_RE.match(part.strip())
        if match is None:
            continue
        hints.append(
            TermHint(
                to=match.group("to").strip(),
                confidence=float(match.group("confidence")),
                source="table",
            )
        )
    return tuple(hints)


def read_help_rows(path: Path) -> list[HelpRow]:
    """Прочитать лист «Нужна помощь», сохраняя заполненные решения.

    Подсказки восстанавливаются из текста ячейки: файл может быть дополнен
    другим прогоном, и повторно вызывать модель незачем.
    """
    if not path.exists():
        return []
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        if SHEET_HELP not in wb.sheetnames:
            return []
        ws = wb[SHEET_HELP]
        header_row, index = _find_header(ws, (FORM_COLUMN, CHOICE_COLUMN))
        rows: list[HelpRow] = []
        for row in ws.iter_rows(min_row=header_row + 1, values_only=True):
            form = _cell(row, index, FORM_COLUMN)
            if not form:
                continue
            frequency_text = _cell(row, index, "Частота")
            try:
                frequency = int(float(frequency_text)) if frequency_text else 0
            except ValueError:
                frequency = 0
            rows.append(
                HelpRow(
                    form=form,
                    frequency=frequency,
                    places=_cell(row, index, PLACES_COLUMN),
                    context=_cell(row, index, CONTEXT_COLUMN),
                    hints=parse_hints(_cell(row, index, HINTS_COLUMN)),
                    note=_cell(row, index, NOTE_COLUMN),
                    choice=_cell(row, index, CHOICE_COLUMN),
                )
            )
        return rows
    finally:
        wb.close()


def write_terms_table(path: Path, rows: list[HelpRow]) -> None:
    """Переписать лист «Нужна помощь» по частоте, сохранив решения.

    Лист «Применено само» не теряется: он дописывается отдельно.
    """
    auto_rows = read_auto_rows(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    help_ws = wb.active
    if help_ws is None:
        msg = "Не удалось создать лист таблицы терминов"
        raise TermsTableError(msg)
    help_ws.title = SHEET_HELP
    help_ws.append([HELP_INSTRUCTION])
    help_ws.append([*HELP_COLUMNS[:3], "Частота", *HELP_COLUMNS[3:]])
    for row in sorted(rows, key=lambda item: (-item.frequency, item.form.casefold())):
        help_ws.append(
            [
                row.form,
                row.places,
                row.context,
                row.frequency,
                row.hints_text(),
                row.choice,
                row.note,
            ]
        )
    auto_ws = wb.create_sheet(SHEET_AUTO)
    auto_ws.append(list(AUTO_COLUMNS))
    for form, replacement, basis, places in auto_rows:
        auto_ws.append([form, replacement, basis, places])
    wb.save(path)


def read_auto_rows(path: Path) -> list[tuple[str, str, str, str]]:
    """Прочитать лист «Применено само»."""
    if not path.exists():
        return []
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        if SHEET_AUTO not in wb.sheetnames:
            return []
        ws = wb[SHEET_AUTO]
        header_row, index = _find_header(ws, (FORM_COLUMN, REPLACEMENT_COLUMN))
        rows: list[tuple[str, str, str, str]] = []
        for row in ws.iter_rows(min_row=header_row + 1, values_only=True):
            form = _cell(row, index, FORM_COLUMN)
            if not form:
                continue
            rows.append(
                (
                    form,
                    _cell(row, index, REPLACEMENT_COLUMN),
                    _cell(row, index, BASIS_COLUMN),
                    _cell(row, index, PLACES_COLUMN),
                )
            )
        return rows
    finally:
        wb.close()


def merge_rows(existing: list[HelpRow], opened: list[HelpRow]) -> list[HelpRow]:
    """Дополнить таблицу новыми формами, не перезаписывая заполненное.

    Уже известная форма сохраняет решение и подсказки: оператор мог ответить
    вчера, и новый прогон не должен стирать ответ.
    """
    by_form = {row.form.casefold(): row for row in existing}
    for row in opened:
        by_form.setdefault(row.form.casefold(), row)
    return sorted(by_form.values(), key=lambda item: (-item.frequency, item.form.casefold()))


def upsert_terms_table(path: Path, opened: list[HelpRow]) -> list[HelpRow]:
    """Дописать новые формы в таблицу проекта и вернуть её строки."""
    existing = read_help_rows(path)
    merged = merge_rows(existing, opened)
    write_terms_table(path, merged)
    return merged


def _hint_choice(row: HelpRow, settings: Settings) -> tuple[str, str, str]:
    """Решение по строке: (цель, статус, основание).

    Пустая ячейка принимает первую подсказку выше порога, иначе оставляет
    слово как есть. `=`, `?` и текст — решения оператора.
    """
    choice = row.choice.strip()
    if choice == "=":
        return row.form, "keep", "оператор: оставить"
    if choice == "?":
        return row.form, "unreadable", "оператор: неразборчиво"
    if choice:
        return choice, "review", "оператор: замена"
    if settings.m7b.auto_apply_defaults and row.hints:
        best = row.hints[0]
        if best.confidence >= settings.m7b.hint_confidence_threshold and best.to not in (KEEP, UNREADABLE):
            basis = f"подсказка {best.confidence:.2f} (порог {settings.m7b.hint_confidence_threshold:.2f})"
            return best.to, "auto", basis
    return row.form, "keep", "автозамена: подсказка ниже порога"


def _rebuild_interview(
    conn: Any,  # noqa: ANN401
    interview_code: str,
    entries: list[LexiconEntry],
    statuses: dict[str, str] | None = None,
) -> int:
    """Детерминированно пересобрать L2 интервью по решениям лексикона.

    Args:
        conn: соединение с БД.
        interview_code: код интервью.
        entries: лексикон с решениями.
        statuses: карта «канон -> статус решения» для журнала (`review`/`auto`).

    Returns:
        Число изменённых единиц; -1 — пересборка запрещена (есть аннотации).
    """
    repo = PipelineRepository(conn)
    interview_db_id = repo.interview_db_id(interview_code)
    if interview_db_id is None:
        return 0
    if repo.annotation_count(interview_db_id):
        return -1
    version = repo.max_l2_version(interview_db_id)
    if not version:
        return 0
    accepted = statuses or {}
    units = TextUnitRepository(conn)
    changed = 0
    for unit_id, text, changes_json in units.rows_for_interview(interview_db_id, version):
        new_text, changes, _counts = apply_lexicon(text, entries)
        if new_text == text or not changes:
            continue
        existing: list[dict[str, Any]] = json.loads(changes_json or "[]")
        for change in changes:
            source = str(change.get("from") or "")
            position = text.casefold().find(source.casefold())
            existing.append(
                {
                    "type": "decision",
                    "from": source,
                    "to": change.get("to"),
                    "position": max(0, position),
                    "accepted": accepted.get(str(change.get("to")), "review"),
                    "confidence": 1.0,
                    "corrected_word": True,
                }
            )
        units.update_text(unit_id, new_text, json.dumps(existing, ensure_ascii=False))
        changed += 1
    return changed


def apply_terms(
    conn: Any,  # noqa: ANN401 - sqlite3.Connection
    settings: Settings,
    interview_code: str | None = None,
) -> ApplyResult:
    """Применить решения таблицы к лексикону и пересобрать L2.

    Args:
        conn: соединение с БД.
        settings: настройки проекта (порог, флаг автозамены, путь таблицы).
        interview_code: ограничить пересборку одним интервью; None — все
            интервью с L2, у которых нет аннотаций.

    Raises:
        TermsTableError: таблицы нет или в ней нет шапки.
    """
    path = terms_path(settings)
    if not path.exists():
        msg = f"Таблица терминов не найдена: {path}. Её создаёт стадия normalize."
        raise TermsTableError(msg)
    rows = read_help_rows(path)
    if not rows:
        return ApplyResult(path=path, rows=0)

    repo = LexiconRepository(conn)
    applied = 0
    auto = 0
    kept = 0
    conflicts = 0
    done: list[HelpRow] = []
    auto_rows: list[tuple[str, str, str, str]] = []
    details: list[str] = []
    decided_forms: list[str] = []
    statuses: dict[str, str] = {}
    for row in rows:
        target, status, basis = _hint_choice(row, settings)
        outcome = repo.set_form_decision(row.form, target, status)
        if outcome == "conflict":
            conflicts += 1
            details.append(f"{row.form}: конфликт — форма уже привязана к другому термину")
            continue
        if outcome == "operator_kept":
            kept += 1
            details.append(f"{row.form}: решение оператора сохранено")
            continue
        applied += 1
        if status in ("auto", "keep", "unreadable"):
            auto += 1
        decided_forms.append(row.form)
        statuses[target] = "auto" if status == "auto" else "review"
        replacement = UNREADABLE if status == "unreadable" else ("оставить" if status == "keep" else target)
        auto_rows.append((row.form, replacement, basis, row.places))
        done.append(row)

    remaining = [row for row in rows if id(row) not in {id(item) for item in done}]
    write_terms_table(path, remaining)
    if auto_rows:
        _append_auto(path, auto_rows)

    entries = _lexicon_entries(conn)
    skipped: list[str] = []
    rebuilt = 0
    targets = (
        [interview_code] if interview_code else [code for _id, code in PipelineRepository(conn).interviews_with_l2()]
    )
    for code in targets:
        changed = _rebuild_interview(conn, code, entries, statuses)
        if changed < 0:
            skipped.append(code)
            continue
        rebuilt += changed

    runs = RunRepository(conn)
    runs.start(
        model="deterministic",
        prompt_hash="",
        codebook_version_id=runs.latest_codebook_version_id(),
        code_version="",
        params_json=json.dumps(
            {"stage": "apply-terms", "decisions": applied, "forms": decided_forms},
            ensure_ascii=False,
        ),
    )
    conn.commit()
    logger.info(
        "apply-terms: строк=%d, применено=%d, автозамен=%d, конфликтов=%d, пропущено интервью=%s",
        len(rows),
        applied,
        auto,
        conflicts,
        skipped,
    )
    return ApplyResult(
        path=path,
        rows=len(rows),
        applied=applied,
        auto=auto,
        kept=kept,
        conflicts=conflicts,
        skipped=tuple(skipped),
        rebuilt=rebuilt,
        details=tuple(details),
    )


def _append_auto(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    """Дописать применённые решения в лист «Применено само»."""
    wb = load_workbook(path)
    try:
        ws = wb[SHEET_AUTO] if SHEET_AUTO in wb.sheetnames else wb.create_sheet(SHEET_AUTO)
        for row in rows:
            ws.append(list(row))
        wb.save(path)
    finally:
        wb.close()


def _lexicon_entries(conn: Any) -> list[LexiconEntry]:  # noqa: ANN401
    """Термины лексикона для пересборки L2."""
    from qa.pipeline.normalize import load_lexicon

    return load_lexicon(conn)


def decide_from_table(conn: Any, form: str, choice: str) -> str:  # noqa: ANN401
    """Записать одно решение оператора по форме (для точечных сценариев)."""
    return apply_choice(conn, form, choice)


__all__ = [
    "AUTO_COLUMNS",
    "HELP_COLUMNS",
    "ApplyResult",
    "HelpRow",
    "OpenForm",
    "SHEET_AUTO",
    "SHEET_HELP",
    "TermsTableError",
    "apply_terms",
    "collect_open_forms",
    "decide_from_table",
    "merge_rows",
    "read_auto_rows",
    "read_help_rows",
    "terms_path",
    "upsert_terms_table",
    "write_terms_table",
]
