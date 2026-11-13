"""Агрегация покрытия дерева тем (M4).

Чистые функции над строками `qa.db.analytics_repo`: SQL и HTML здесь
отсутствуют (SPEC-M6, «Разделение слоёв»). Модуль также держит модель
покрытия, которую принимает рендер дашборда и модель отчёта.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from qa.db.analytics_repo import (
    AnalyticsRepository,
    AnnotationRow,
    CodeRow,
    QuestionRow,
    ThemeRow,
)
from qa.pipeline.quote_view import (
    PREFERRED_MIN_CHARS,
    Quote,
    QuoteEditor,
    build_quote,
    quote_source_text,
    select_annotations,
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
class ThemeCoverage:
    """Покрытие одной темы аннотациями (все, кроме отклонённых)."""

    node: str
    theme: str
    status: str
    annotations: int
    verified: int
    distinct_codes: int
    theme_codes: int
    covered_fragments: int
    cumulative: tuple[int, ...]
    quotes: tuple[Quote, ...]
    annotations_detail: tuple[AnnotationRow, ...] = ()


@dataclass(frozen=True)
class NodeCoverage:
    """Узел дерева и его темы."""

    node: str
    themes: tuple[ThemeCoverage, ...]
    annotations: int
    verified: int
    distinct_codes: int


@dataclass(frozen=True)
class UnusedCode:
    """Неиспользованные коды одной семьи."""

    family: str
    names: tuple[str, ...]


@dataclass(frozen=True)
class CodebookCoverage:
    """Покрытие кодбука: использованные и неиспользованные коды по семьям."""

    used: int
    total: int
    unused: tuple[UnusedCode, ...]


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
    interpretation_only: int
    reliability: dict[int, int]
    questions_without_answer: tuple[str, ...]
    accepted_on_doubtful: int
    fragments_with_accepted_doubtful: int


@dataclass(frozen=True)
class CoverageReport:
    """Полная модель покрытия интервью для дашборда и отчёта."""

    interview_code: str
    interview_db_id: int
    group: str
    run_id: int | None
    l2_version: int | None
    codebook_version_id: int | None
    nodes: tuple[NodeCoverage, ...]
    codebook: CodebookCoverage
    quality: DataQuality
    fragments_total: int
    covered_fragments: int
    annotations_total: int
    annotations_verified: int
    annotations: tuple[AnnotationRow, ...] = ()
    code_names: dict[int, str] = field(default_factory=dict)


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
    """Обрезать цитату по границе слова и отметить вырезку `[…]`.

    Обёртка над `qa.pipeline.quote_view.truncate_quote`: правило одно на
    дашборд и отчёт, поэтому живёт в одном месте.
    """
    from qa.pipeline.quote_view import truncate_quote as cut

    return cut(text, limit)[0]


def quote_text(annotation: AnnotationRow, limit: int) -> str:
    """Текст цитаты: срез L2 по позициям аннотации, иначе сохранённая цитата.

    Позиции относятся к тексту единицы L2, поэтому срез по ним и есть дословная
    цитата (SPEC-M6, критерий 9). Сохранённая `annotation.quote` используется
    только если позиций нет.
    """
    from qa.pipeline.quote_view import quote_source_text

    return truncate_quote(quote_source_text(annotation), limit)


def cumulative_codes(annotations: list[AnnotationRow]) -> tuple[int, ...]:
    """Накопление различных кодов по порядку реплик.

    Шаг — одна аннотация; значение — сколько различных кодов встретилось до
    этого шага включительно.
    """
    seen: set[int] = set()
    points: list[int] = []
    for annotation in annotations:
        seen.add(annotation.code_id)
        points.append(len(seen))
    return tuple(points)


def _codebook_coverage(
    codes: list[CodeRow],
    used_code_ids: set[int],
    family_order: dict[str, int],
) -> CodebookCoverage:
    """Покрытие кодбука: использованные коды и неиспользованные по семьям.

    Деления на «не применимо к группе» и «применимо, но не найдено» нет:
    торговые вопросы фактически задавались и другим группам, а при кодировании
    код по группе не исключался. Поэтому список один.
    """
    unused_by_family: dict[str, list[str]] = {}
    for code in codes:
        if code.id in used_code_ids:
            continue
        unused_by_family.setdefault(code.family or "—", []).append(code.name)

    unused = tuple(
        UnusedCode(family=family, names=tuple(names))
        for family, names in sorted(
            unused_by_family.items(),
            key=lambda item: (family_order.get(item[0], len(family_order)), item[0]),
        )
    )
    return CodebookCoverage(
        used=len(used_code_ids & {code.id for code in codes}),
        total=len(codes),
        unused=unused,
    )


def _questions_without_answer(questions: list[QuestionRow], answered: set[int]) -> tuple[str, ...]:
    """Вопросы гайда, на которые респондент не ответил.

    Фильтра по группе нет: считаются фактически заданные вопросы. Отсутствие
    ответа на вопрос другой группы — такой же пропуск, как и на «своей».
    """
    missing = [question.identifier for question in questions if question.id not in answered]
    return tuple(sorted(missing, key=_identifier_sort_key))


def _identifier_sort_key(identifier: str) -> tuple[int, str]:
    """Порядок идентификаторов: числовые по возрастанию, затем остальные."""
    digits = identifier.rstrip("Т-абв")
    return (int(digits), identifier) if digits.isdigit() else (10**6, identifier)


def _theme_quotes(
    annotations: list[AnnotationRow],
    code_names: dict[int, str],
    limit: int,
    max_chars: int,
    editor: QuoteEditor | None,
    min_chars: int,
) -> tuple[Quote, ...]:
    """Отобрать и оформить цитаты темы.

    Отбор — `quote_view.select_annotations`: `garbled` не попадает, `doubtful`
    идёт после `ok`. Правка моделью — только для отобранных цитат и только если
    её включил оператор.
    """
    quotes: list[Quote] = []
    for annotation in select_annotations(annotations, limit, min_chars=min_chars):
        source = quote_source_text(annotation)
        edited_text: str | None = None
        rule = ""
        if editor is not None:
            edited_text, _reason, rule = editor(source)
        quotes.append(
            build_quote(
                annotation,
                code_names.get(annotation.code_id, ""),
                max_chars,
                edited_text=edited_text,
                edit_rule=rule,
            )
        )
    return tuple(quotes)


def build_coverage(
    conn: sqlite3.Connection,
    interview_code: str,
    groups: GroupsSettings,
    *,
    low_min: int,
    high_min: int,
    quotes_per_theme: int,
    quote_max_chars: int,
    quote_editor: QuoteEditor | None = None,
    quote_min_chars: int = PREFERRED_MIN_CHARS,
) -> CoverageReport:
    """Собрать модель покрытия интервью.

    База агрегатов — все аннотации, кроме `rejected`. Выборочный контроль не
    фильтр: проверенные вручную (`accepted`/`edited`) выводятся отдельным
    счётчиком «из них проверено», но не определяют покрытие.

    Args:
        quote_editor: необязательная правка цитаты моделью; None — без правки.
            Правка применяется только к уже отобранным цитатам (SPEC-M8, C.3).

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
    annotations = repo.annotations(interview.id)

    group = interview.group or ""
    family_order = _family_order(themes, theme_codes, codes)
    code_names = {code.id: code.name for code in codes}

    annotations_by_code: dict[int, list[AnnotationRow]] = {}
    for annotation in annotations:
        annotations_by_code.setdefault(annotation.code_id, []).append(annotation)

    themes_by_node: dict[int, list[ThemeCoverage]] = {}
    node_names: dict[int, str] = {}
    node_code_ids: dict[int, set[int]] = {}
    for theme in themes:
        if theme.level == 0:
            node_names[theme.id] = theme.name
            themes_by_node.setdefault(theme.id, [])
            node_code_ids.setdefault(theme.id, set())
    for theme in themes:
        if theme.level != 1 or theme.parent_id is None:
            continue
        node_name = node_names.get(theme.parent_id, "")
        theme_code_ids = [code_id for theme_id, code_id in theme_codes if theme_id == theme.id]
        node_code_ids.setdefault(theme.parent_id, set()).update(theme_code_ids)
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
                verified=sum(1 for annotation in theme_annotations if annotation.verified),
                distinct_codes=len({annotation.code_id for annotation in theme_annotations}),
                theme_codes=len(theme_code_ids),
                covered_fragments=len({annotation.fragment_id for annotation in theme_annotations}),
                cumulative=cumulative_codes(theme_annotations),
                quotes=_theme_quotes(
                    theme_annotations,
                    code_names,
                    quotes_per_theme,
                    quote_max_chars,
                    quote_editor,
                    quote_min_chars,
                ),
                annotations_detail=tuple(theme_annotations),
            )
        )

    nodes = tuple(
        NodeCoverage(
            node=name,
            themes=tuple(themes_by_node.get(node_id, [])),
            annotations=sum(theme.annotations for theme in themes_by_node.get(node_id, [])),
            verified=sum(theme.verified for theme in themes_by_node.get(node_id, [])),
            distinct_codes=len(
                {
                    annotation.code_id
                    for code_id in node_code_ids.get(node_id, set())
                    for annotation in annotations_by_code.get(code_id, [])
                }
            ),
        )
        for node_id, name in node_names.items()
    )

    used_code_ids = {annotation.code_id for annotation in annotations}
    codebook = _codebook_coverage(codes, used_code_ids, family_order)

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
        interpretation_only=sum(1 for annotation in annotations if annotation.interpretation_only),
        reliability=dict(sorted(reliability.items())),
        questions_without_answer=_questions_without_answer(questions, repo.answered_question_ids(interview.id)),
        accepted_on_doubtful=len(doubtful_annotations),
        fragments_with_accepted_doubtful=len({item.fragment_id for item in doubtful_annotations}),
    )

    return CoverageReport(
        interview_code=interview.code,
        interview_db_id=interview.id,
        group=normalize_group(group, groups) if group else "",
        run_id=run_id,
        l2_version=(run.l2_version if run is not None else None) or repo.l2_version_of_fragments(interview.id),
        codebook_version_id=codebook_version_id,
        nodes=nodes,
        codebook=codebook,
        quality=quality,
        fragments_total=repo.fragments_total(interview.id),
        covered_fragments=len({annotation.fragment_id for annotation in annotations}),
        annotations_total=len(annotations),
        annotations_verified=sum(1 for annotation in annotations if annotation.verified),
        annotations=tuple(annotations),
        code_names=code_names,
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
