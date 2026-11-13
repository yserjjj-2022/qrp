"""Таблица терминов интервью: выгрузка для коллег и импорт решений (M8).

Выгрузка собирается только из анонимизированного текста реплик
(`utterance.text`). Метки говорящих (`turn.speaker_label`) содержат настоящие
имена и в файл не попадают: в колонке контекста стоит роль (CONSTITUTION §5).

Импорт знает одно правило — «всегда заменять X на Y»: оно пополняет лексикон
как решение оператора (`source='review'`), поэтому действует при следующей
нормализации и не затирается повторной загрузкой книги. Разовая правка
«в этой реплике имелось в виду Z» живёт в отдельном JSON-файле
(`data/exports/quote_edits_<код>.json`) и применяется на этапе L2 до модели:
в таблицу её не записать, не смешав со всеобщей заменой термина.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from openpyxl import Workbook, load_workbook

from qa.db.pipeline_repo import PipelineRepository, UtteranceRow
from qa.db.workbook_repo import LexiconRepository, NewTermRepository

logger = logging.getLogger(__name__)

COLUMNS = [
    "форма",
    "частота",
    "тайм-код",
    "реплика_контекста",
    "правильное_написание_или_смысл",
    "комментарий",
]
FORM_COLUMN = "форма"
FREQUENCY_COLUMN = "частота"
TIMECODE_COLUMN = "тайм-код"
CONTEXT_COLUMN = "реплика_контекста"
CANONICAL_COLUMN = "правильное_написание_или_смысл"
COMMENT_COLUMN = "комментарий"

# Инструкция для коллег идёт первой строкой файла, до шапки таблицы: файл
# отправляют человеку, который не читал SPEC.
INSTRUCTION = (
    "Таблица терминов. Имена людей и названия организаций не вписывать: "
    "использовать обозначение из таблицы имён. Если слушали запись — указать тайм-код. "
    "Пустая ячейка «правильное_написание_или_смысл» означает, что решения нет. "
    "Заполненная — «всегда заменять форму на это написание»."
)
SINGLE_BLOCK_TITLE = "формы с одним вхождением"

# Реплика с содержимым: не бэкканал и не короче четырёх слов. Короткие
# подтверждения («угу», «да, конечно») контекстом термина не являются.
MIN_CONTEXT_WORDS = 4

_WORD_RE = re.compile(r"[а-яёa-z]+(?:-[а-яёa-z]+)?", re.IGNORECASE)
_ROLE_LABELS = {
    "RESPONDENT": "респондент",
    "INTERVIEWER": "интервьюер",
    "OBSERVER": "наблюдатель",
}


@dataclass(frozen=True)
class TermRow:
    """Строка таблицы терминов."""

    form: str
    frequency: int
    timecode: int | None
    context: str


@dataclass(frozen=True)
class TermsExportResult:
    """Итог выгрузки таблицы терминов."""

    path: Path
    n_rows: int
    n_single: int


@dataclass(frozen=True)
class TermsImportResult:
    """Итог импорта таблицы терминов."""

    path: Path
    n_rows: int
    n_variants: int
    n_skipped: int
    variants: tuple[tuple[str, str], ...] = ()


def format_timecode(seconds: int | None) -> str:
    """Тайм-код вида `mm:ss`; None даёт пустую строку."""
    if seconds is None:
        return ""
    minutes, rest = divmod(max(0, int(seconds)), 60)
    return f"{minutes:02d}:{rest:02d}"


def _word_count(text: str) -> int:
    """Число слов в тексте."""
    return len(_WORD_RE.findall(text))


def _occurrences(text: str, form: str) -> int:
    """Сколько раз форма встречается в тексте целым словом."""
    if not form:
        return 0
    pattern = rf"(?<![а-яёa-z0-9]){re.escape(form.casefold())}(?![а-яёa-z0-9])"
    return len(re.findall(pattern, text.casefold()))


def build_term_rows(conn: sqlite3.Connection, interview_code: str) -> list[TermRow]:
    """Собрать строки таблицы терминов по кандидатам `new_term`.

    Дубли формы схлопываются: `new_term` хранит одну строку на реплику, и
    коллега не должен видеть одну форму дважды. Частота суммируется по
    вхождениям в репликах респондента, а не по числу реплик.

    Raises:
        ValueError: интервью не найдено в БД.
    """
    repo = PipelineRepository(conn)
    interview_db_id = repo.interview_db_id(interview_code)
    if interview_db_id is None:
        msg = f"Интервью {interview_code} не найдено в БД"
        raise ValueError(msg)

    utterances = repo.utterances_with_timecodes(interview_db_id)
    respondents = [item for item in utterances if item.speaker_role == "RESPONDENT"]
    content = [item for item in utterances if not item.is_backchannel and _word_count(item.text) >= MIN_CONTEXT_WORDS]

    merged: dict[str, TermRow] = {}
    order: list[str] = []
    for candidate in repo.proposed_terms(interview_db_id):
        key = candidate.surface_form.casefold()
        if key in merged:
            # Форма уже посчитана целиком: частота берётся по вхождениям в
            # репликах, а не по числу строк `new_term`. Иначе дубль кандидата
            # удвоил бы частоту.
            continue
        order.append(key)
        merged[key] = TermRow(
            form=candidate.surface_form,
            frequency=sum(_occurrences(item.text, candidate.surface_form) for item in respondents),
            timecode=_first_timecode(respondents, candidate.surface_form),
            context=_context_around(content, respondents, candidate.surface_form),
        )
    return [merged[key] for key in order]


def _first_timecode(respondents: list[UtteranceRow], form: str) -> int | None:
    """Тайм-код первой реплики респондента с вхождением формы."""
    for item in respondents:
        if _occurrences(item.text, form):
            return item.timecode
    return None


def _context_around(content: list[UtteranceRow], respondents: list[UtteranceRow], form: str) -> str:
    """1–2 содержательные реплики вокруг первого вхождения формы.

    Берётся реплика с вхождением и предыдущая содержательная реплика: по ним
    видно, о чём шла речь. Метки говорящих не используются — только роль.
    Бэкканал контекстом не считается: «угу» не показывает, о чём речь.
    """
    target_id: int | None = None
    for item in respondents:
        if _occurrences(item.text, form):
            target_id = item.id
            break
    if target_id is None:
        return ""
    position = next((index for index, item in enumerate(content) if item.id == target_id), None)
    if position is None:
        # Реплика с вхождением короче порога контекста: показываем её саму.
        # Бэкканал не показываем вовсе: он не несёт содержания.
        for item in respondents:
            if item.id == target_id:
                return "" if item.is_backchannel else _render_context_line(item)
        return ""
    start = max(0, position - 1)
    return " | ".join(_render_context_line(item) for item in content[start : position + 1])


def _render_context_line(item: UtteranceRow) -> str:
    """Строка контекста: роль и текст, без метки говорящего."""
    label = _ROLE_LABELS.get(item.speaker_role, item.speaker_role.lower() or "реплика")
    return f"{label}: {item.text}"


def export_terms_table(
    conn: sqlite3.Connection,
    interview_code: str,
    output_path: Path,
) -> TermsExportResult:
    """Записать xlsx с таблицей терминов интервью.

    Raises:
        ValueError: интервью не найдено или кандидатов нет.
    """
    rows = build_term_rows(conn, interview_code)
    if not rows:
        msg = f"У интервью {interview_code} нет кандидатов терминов: таблица не выгружена"
        raise ValueError(msg)

    frequent = sorted((row for row in rows if row.frequency > 1), key=lambda row: (-row.frequency, row.form.casefold()))
    single = sorted((row for row in rows if row.frequency <= 1), key=lambda row: row.form.casefold())

    wb = Workbook()
    ws = wb.active
    if ws is None:
        msg = "Не удалось создать лист для таблицы терминов"
        raise RuntimeError(msg)
    ws.title = "Термины"
    ws.append([INSTRUCTION])
    ws.append([])
    ws.append(COLUMNS)
    for row in frequent:
        ws.append([row.form, row.frequency, format_timecode(row.timecode), row.context, None, None])
    if single:
        ws.append([])
        ws.append([SINGLE_BLOCK_TITLE])
        for row in single:
            ws.append([row.form, row.frequency, format_timecode(row.timecode), row.context, None, None])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    logger.info(
        "Таблица терминов %s: строк=%d (одиночных=%d) -> %s",
        interview_code,
        len(rows),
        len(single),
        output_path,
    )
    return TermsExportResult(path=output_path, n_rows=len(rows), n_single=len(single))


def _find_header_row(ws: object) -> tuple[int, dict[str, int]]:
    """Найти строку шапки таблицы и индексы колонок.

    Шапка ищется по колонкам, а не по номеру строки: над ней стоит инструкция,
    и её длина может меняться.
    """
    for row_index, row in enumerate(ws.iter_rows(values_only=True), start=1):  # type: ignore[attr-defined]
        headers = [str(value).strip() if value is not None else "" for value in row]
        if FORM_COLUMN in headers and CANONICAL_COLUMN in headers:
            return row_index, {name: headers.index(name) for name in headers if name}
        if row_index > 20:
            break
    msg = f"В файле таблицы терминов нет шапки с колонками {FORM_COLUMN!r} и {CANONICAL_COLUMN!r}"
    raise ValueError(msg)


def import_terms_table(conn: sqlite3.Connection, path: Path) -> TermsImportResult:
    """Прочитать заполненную таблицу и пополнить лексикон решениями оператора.

    Заполненная колонка «правильное_написание_или_смысл» означает «всегда
    заменять форму на это написание»: форма становится вариантом термина, а
    запись получает `source='review'`. Пустая ячейка — решения нет, строка
    пропускается и ничего не блокирует.

    Raises:
        FileNotFoundError: файл не найден.
        ValueError: нет шапки таблицы.
    """
    if not path.exists():
        msg = f"Файл таблицы терминов не найден: {path}"
        raise FileNotFoundError(msg)
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb[wb.sheetnames[0]]
        header_row, index = _find_header_row(ws)
        lexicon = LexiconRepository(conn)
        terms = NewTermRepository(conn)
        known = {form.casefold(): term_id for term_id, form, _ctx in terms.proposed()}
        rows = 0
        variants: list[tuple[str, str]] = []
        skipped = 0
        for row in ws.iter_rows(min_row=header_row + 1, values_only=True):
            if row is None or all(value is None for value in row):
                continue
            form = _cell(row, index, FORM_COLUMN)
            if not form:
                continue
            rows += 1
            canonical = _cell(row, index, CANONICAL_COLUMN)
            if not canonical:
                skipped += 1
                continue
            if lexicon.add_variant(canonical, form):
                variants.append((form, canonical))
            term_id = known.get(form.casefold())
            if term_id is not None:
                terms.decide(term_id, "approved")
    finally:
        wb.close()
    conn.commit()
    logger.info(
        "Таблица терминов импортирована: строк=%d, вариантов=%d, без решения=%d",
        rows,
        len(variants),
        skipped,
    )
    return TermsImportResult(
        path=path,
        n_rows=rows,
        n_variants=len(variants),
        n_skipped=skipped,
        variants=tuple(variants),
    )


def _cell(row: tuple[object, ...], index: dict[str, int], name: str) -> str | None:
    """Значение ячейки по имени колонки."""
    position = index.get(name)
    if position is None or position >= len(row) or row[position] is None:
        return None
    value = str(row[position]).strip()
    return value or None


__all__ = [
    "COLUMNS",
    "INSTRUCTION",
    "TermRow",
    "TermsExportResult",
    "TermsImportResult",
    "build_term_rows",
    "export_terms_table",
    "format_timecode",
    "import_terms_table",
]
