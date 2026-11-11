"""Агрегация покрытия дерева тем (M4).

Чистые функции над строками `qa.db.analytics_repo`: SQL и HTML здесь
отсутствуют (SPEC-M6, «Разделение слоёв»). Модуль также держит модель
покрытия, которую принимает рендер дашборда и модель отчёта.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from qa.db.analytics_repo import (
    AcceptedAnnotationRow,
    AnalyticsRepository,
    CodeRow,
    QuestionCodeRow,
    QuestionRow,
    ThemeRow,
)
from qa.settings import GroupsSettings, normalize_group

STATUS_NO_DATA = "no_data"
STATUS_FEW = "few"
STATUS_HAS_DATA = "has_data"

STATUS_LABELS: dict[str, str] = {
    "no_data": "нет данных",
    "few": "мало",
    "has_data": "есть данные",
}

# Подпись, обязательная рядом с каждым эвристическим показателем
# (CONSTITUTION §6): шкала посчитана на одном интервью.
HEURISTIC_CAPTION = "эвристика, 1 интервью"


class AggregateError(Exception):
    """Агрегация невозможна: интервью нет в БД или нет версии кодбука."""


@dataclass(frozen=True)
class QuoteItem:
    """Цитата темы: дословный срез L2 или вырезка с многоточием."""

    code_name: str
    text: str
    question_identifier: str | None


@dataclass(frozen=True)
class ThemeCoverage:
    """Покрытие одной темы принятыми аннотациями."""

    node: str
    theme: str
    status: str
    annotations: int
    distinct_codes: int
    theme_codes: int
    covered_fragments: int
    cumulative: tuple[int, ...]
    quotes: tuple[QuoteItem, ...]


@dataclass(frozen=True)
class NodeCoverage:
    """Узел дерева и его темы."""

    node: str
    themes: tuple[ThemeCoverage, ...]
    annotations: int
    distinct_codes: int


@dataclass(frozen=True)
class UnusedCode:
    """Неиспользованные коды одной семьи с общим признаком применимости."""

    family: str
    names: tuple[str, ...]
    applicable: bool


@dataclass(frozen=True)
class CodebookCoverage:
    """Покрытие кодбука: использованные и неиспользованные коды."""

    used: int
    total: int
    unused: tuple[UnusedCode, ...]
    unused_applicable: int
    unused_not_applicable: int


@dataclass(frozen=True)
class DataQuality:
    """Качество данных: статусы, надёжность, доля на doubtful."""

    by_status: dict[str, int]
    coded: int
    unmarked: int
    oth: int
    skipped_garbled: int
    failed: int
    doubtful: int
    reliability: dict[int, int]
    questions_without_answer: tuple[str, ...]
    accepted_on_doubtful: int
    fragments_with_accepted_doubtful: int


@dataclass(frozen=True)
class CoverageReport:
    """Полная модель покрытия интервью для дашборда и отчёта."""

    interview_code: str
    group: str
    run_id: int | None
    l2_version: int | None
    codebook_version_id: int | None
    nodes: tuple[NodeCoverage, ...]
    codebook: CodebookCoverage
    quality: DataQuality
    fragments_total: int
    covered_fragments: int


def theme_status(annotations: int, low_min: int, high_min: int) -> str:
    """Статус листа по числу принятых аннотаций темы.

    Шкала отвечает только на вопрос «есть ли чем показать тему»: на одном
    интервью она не измеряет раскрытие темы.
    """
    if annotations < low_min:
        return STATUS_NO_DATA
    if annotations < high_min:
        return STATUS_FEW
    return STATUS_HAS_DATA


def truncate_quote(text: str, limit: int) -> str:
    """Обрезать цитату по границе слова и отметить вырезку многоточием."""
    if limit <= 0 or len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    if space > 0:
        cut = cut[:space]
    return cut.rstrip(" ,;:—-") + " …"


def quote_text(annotation: AcceptedAnnotationRow, limit: int) -> str:
    """Текст цитаты: срез L2 по позициям аннотации, иначе сохранённая цитата.

    Позиции относятся к тексту единицы L2, поэтому срез по ним и есть дословная
    цитата (SPEC-M6, критерий 9). Сохранённая `annotation.quote` используется
    только если позиций нет.
    """
    if annotation.quote_start is not None and annotation.quote_end is not None:
        sliced = annotation.l2_text[annotation.quote_start : annotation.quote_end]
        if sliced:
            return truncate_quote(sliced, limit)
    return truncate_quote(annotation.quote, limit)


def cumulative_codes(annotations: list[AcceptedAnnotationRow]) -> tuple[int, ...]:
    """Накопление различных кодов по порядку реплик.

    Шаг — одна принятая аннотация; значение — сколько различных кодов
    встретилось до этого шага включительно.
    """
    seen: set[int] = set()
    points: list[int] = []
    for annotation in annotations:
        seen.add(annotation.code_id)
        points.append(len(seen))
    return tuple(points)


def _applicable_code_ids(
    interview_group: str,
    questions: list[QuestionRow],
    question_codes: list[QuestionCodeRow],
    groups: GroupsSettings,
) -> tuple[set[int], set[int]]:
    """Коды, применимые к группе интервью, и коды с вопросами вообще.

    Код применим, если хотя бы один его вопрос адресован всем
    (`groups.universal`) или этой группе. Код без вопросов применим: у него нет
    ограничения по группе.
    """
    universal = normalize_group(groups.universal, groups)
    target = normalize_group(interview_group, groups)
    allowed_question_ids = {
        question.id for question in questions if normalize_group(question.group or "", groups) in (universal, target)
    }
    with_questions: set[int] = {link.code_id for link in question_codes}
    applicable: set[int] = {link.code_id for link in question_codes if link.question_id in allowed_question_ids}
    return applicable, with_questions


def _codebook_coverage(
    codes: list[CodeRow],
    used_code_ids: set[int],
    applicable_code_ids: set[int],
    codes_with_questions: set[int],
    family_order: dict[str, int],
) -> CodebookCoverage:
    """Покрытие кодбука: использованные коды и неиспользованные по семьям."""
    unused_by_key: dict[tuple[str, bool], list[str]] = {}
    for code in codes:
        if code.id in used_code_ids:
            continue
        family = code.family or "—"
        applicable = code.id in applicable_code_ids or code.id not in codes_with_questions
        unused_by_key.setdefault((family, applicable), []).append(code.name)

    order = sorted(
        unused_by_key.items(),
        key=lambda item: (family_order.get(item[0][0], len(family_order)), item[0][0], not item[0][1]),
    )
    unused = tuple(
        UnusedCode(family=family, names=tuple(names), applicable=applicable) for (family, applicable), names in order
    )
    not_applicable = sum(len(entry.names) for entry in unused if not entry.applicable)
    return CodebookCoverage(
        used=len(used_code_ids & {code.id for code in codes}),
        total=len(codes),
        unused=unused,
        unused_applicable=sum(len(entry.names) for entry in unused if entry.applicable),
        unused_not_applicable=not_applicable,
    )


def _questions_without_answer(
    questions: list[QuestionRow],
    answered: set[int],
    interview_group: str,
    groups: GroupsSettings,
) -> tuple[str, ...]:
    """Вопросы гайда этой группы, на которые респондент не ответил."""
    universal = normalize_group(groups.universal, groups)
    target = normalize_group(interview_group, groups)
    missing = [
        question.identifier
        for question in questions
        if normalize_group(question.group or "", groups) in (universal, target) and question.id not in answered
    ]
    return tuple(sorted(missing, key=_identifier_sort_key))


def _identifier_sort_key(identifier: str) -> tuple[int, str]:
    """Порядок идентификаторов: числовые по возрастанию, затем остальные."""
    digits = identifier.rstrip("Т-абв")
    return (int(digits), identifier) if digits.isdigit() else (10**6, identifier)


def build_coverage(
    conn: sqlite3.Connection,
    interview_code: str,
    groups: GroupsSettings,
    *,
    low_min: int,
    high_min: int,
    quotes_per_theme: int,
    quote_max_chars: int,
) -> CoverageReport:
    """Собрать модель покрытия интервью.

    Raises:
        AggregateError: интервью нет в БД или не загружена версия кодбука.
    """
    repo = AnalyticsRepository(conn)
    interview = repo.interview_by_code(interview_code)
    if interview is None:
        msg = f"Интервью {interview_code} не найдено в БД"
        raise AggregateError(msg)
    run = repo.latest_coding_run(interview.id)
    codebook_version_id = (
        run.codebook_version_id if run is not None and run.codebook_version_id is not None else None
    ) or repo.latest_codebook_version_id()
    if codebook_version_id is None:
        msg = "Не загружена ни одна версия кодбука: сначала выполните load-codebook"
        raise AggregateError(msg)

    codes = repo.codes(codebook_version_id)
    themes = repo.themes()
    theme_codes = repo.theme_codes()
    questions = repo.questions(codebook_version_id)
    question_codes = repo.question_codes(codebook_version_id)
    annotations = repo.accepted_annotations(interview.id)

    group = interview.group or ""
    applicable_code_ids, codes_with_questions = _applicable_code_ids(group, questions, question_codes, groups)
    family_order = _family_order(themes, theme_codes, codes)
    code_names = {code.id: code.name for code in codes}

    annotations_by_code: dict[int, list[AcceptedAnnotationRow]] = {}
    for annotation in annotations:
        annotations_by_code.setdefault(annotation.code_id, []).append(annotation)

    themes_by_node: dict[int, list[ThemeCoverage]] = {}
    node_names: dict[int, str] = {}
    for theme in themes:
        if theme.level == 0:
            node_names[theme.id] = theme.name
            themes_by_node.setdefault(theme.id, [])
    for theme in themes:
        if theme.level != 1 or theme.parent_id is None:
            continue
        node_name = node_names.get(theme.parent_id, "")
        theme_code_ids = [code_id for theme_id, code_id in theme_codes if theme_id == theme.id]
        theme_annotations = [
            annotation for code_id in theme_code_ids for annotation in annotations_by_code.get(code_id, [])
        ]
        theme_annotations.sort(key=lambda item: (item.utterance_id, item.start, item.id))
        themes_by_node.setdefault(theme.parent_id, []).append(
            ThemeCoverage(
                node=node_name,
                theme=theme.name,
                status=theme_status(len(theme_annotations), low_min, high_min),
                annotations=len(theme_annotations),
                distinct_codes=len({annotation.code_id for annotation in theme_annotations}),
                theme_codes=len(theme_code_ids),
                covered_fragments=len({annotation.fragment_id for annotation in theme_annotations}),
                cumulative=cumulative_codes(theme_annotations),
                quotes=tuple(
                    QuoteItem(
                        code_name=code_names.get(annotation.code_id, ""),
                        text=quote_text(annotation, quote_max_chars),
                        question_identifier=annotation.question_identifier,
                    )
                    for annotation in theme_annotations[:quotes_per_theme]
                ),
            )
        )

    nodes = tuple(
        NodeCoverage(
            node=name,
            themes=tuple(themes_by_node.get(node_id, [])),
            annotations=sum(theme.annotations for theme in themes_by_node.get(node_id, [])),
            distinct_codes=sum(theme.distinct_codes for theme in themes_by_node.get(node_id, [])),
        )
        for node_id, name in node_names.items()
    )

    used_code_ids = {annotation.code_id for annotation in annotations}
    codebook = _codebook_coverage(codes, used_code_ids, applicable_code_ids, codes_with_questions, family_order)

    run_id = run.id if run is not None else None
    coding_counts = repo.fragment_coding_counts(interview.id, run_id) if run_id is not None else {}
    quality_counts = repo.fragment_quality_counts(interview.id)
    status_counts = repo.annotation_status_counts(interview.id)
    oth_code_ids = {code.id for code in codes if code.name == "OTH"}
    doubtful_annotations = [item for item in annotations if item.fragment_quality == "doubtful"]
    reliability: dict[int, int] = {}
    for annotation in annotations:
        if annotation.reliability is not None:
            reliability[annotation.reliability] = reliability.get(annotation.reliability, 0) + 1

    quality = DataQuality(
        by_status=status_counts,
        coded=coding_counts.get("coded", 0),
        unmarked=coding_counts.get("unmarked", 0),
        oth=sum(1 for annotation in annotations if annotation.code_id in oth_code_ids),
        skipped_garbled=coding_counts.get("skipped_garbled", 0),
        failed=coding_counts.get("failed", 0),
        doubtful=quality_counts.get("doubtful", 0),
        reliability=dict(sorted(reliability.items())),
        questions_without_answer=_questions_without_answer(
            questions, repo.answered_question_ids(interview.id), group, groups
        ),
        accepted_on_doubtful=len(doubtful_annotations),
        fragments_with_accepted_doubtful=len({item.fragment_id for item in doubtful_annotations}),
    )

    return CoverageReport(
        interview_code=interview.code,
        group=normalize_group(group, groups) if group else "",
        run_id=run_id,
        l2_version=(run.l2_version if run is not None else None) or repo.l2_version_of_fragments(interview.id),
        codebook_version_id=codebook_version_id,
        nodes=nodes,
        codebook=codebook,
        quality=quality,
        fragments_total=repo.fragments_total(interview.id),
        covered_fragments=len({annotation.fragment_id for annotation in annotations}),
    )


def _family_order(themes: list[ThemeRow], theme_codes: list[tuple[int, int]], codes: list[CodeRow]) -> dict[str, int]:
    """Порядок семей по дереву тем: неиспользованные коды идут как темы."""
    code_family = {code.id: code.family or "—" for code in codes}
    order: dict[str, int] = {}
    for theme in themes:
        if theme.level != 1:
            continue
        for theme_id, code_id in theme_codes:
            if theme_id != theme.id:
                continue
            family = code_family.get(code_id)
            if family is not None and family not in order:
                order[family] = len(order)
    for code in codes:
        family = code.family or "—"
        if family not in order:
            order[family] = len(order)
    return order
