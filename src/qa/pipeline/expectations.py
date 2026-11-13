"""Раздел отчёта «Ожидания и рекомендации респондентов» (M8).

Раздел состоит из двух подразделов, которые не объединяются и чьи счётчики не
суммируются (SPEC-M8, решение 2):

- «Границы автономии ИИ» — аннотации, чьи коды перечислены в
  `m8.autonomy_codes`;
- «Советы и условия согласия» — аннотации к вопросам гайда из
  `m8.advice_questions`, без отбора по коду.

В кодбуке нет признака «ожидание / рекомендация / барьер», поэтому отбор идёт
по двум разным признакам: код отвечает на вопрос «о чём реплика», вопрос гайда
— «на что отвечал респондент». Одна аннотация может попасть в оба подраздела.

Группировка внутри подраздела детерминированная: по теме, к которой относится
код, а если темы нет — по коду. Нарратива модели нет: заголовки, число и
цитаты.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qa.db.analytics_repo import AnnotationRow
from qa.pipeline.aggregate import CoverageReport
from qa.pipeline.quote_view import (
    PREFERRED_MIN_CHARS,
    Quote,
    QuoteEditor,
    build_quote,
    quote_source_text,
    select_annotations,
)
from qa.settings import M8Settings

AUTONOMY_TITLE = "Границы автономии ИИ"
ADVICE_TITLE = "Советы и условия согласия"


@dataclass(frozen=True)
class ExpectationGroup:
    """Группа цитат внутри подраздела: тема или код."""

    title: str
    annotations: int
    codes: tuple[str, ...]
    quotes: tuple[Quote, ...]


@dataclass(frozen=True)
class ExpectationSection:
    """Подраздел «Ожиданий и рекомендаций»."""

    title: str
    description: str
    selector: str
    annotations: int
    groups: tuple[ExpectationGroup, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Expectations:
    """Раздел целиком: два подраздела и признак «счётчики не суммируются»."""

    sections: tuple[ExpectationSection, ...]
    overlap: int

    @property
    def total(self) -> int:
        """Сумма аннотаций по подразделам (не число уникальных аннотаций)."""
        return sum(section.annotations for section in self.sections)


def _theme_of_code(coverage: CoverageReport) -> dict[str, str]:
    """Карта «код -> тема» по дереву: первый узел → тема, где встречается код."""
    result: dict[str, str] = {}
    for node in coverage.nodes:
        for theme in node.themes:
            for annotation in theme.annotations_detail:
                name = coverage.code_names.get(annotation.code_id)
                if name is not None and name not in result:
                    result[name] = f"{node.node} → {theme.theme}"
    return result


def _group_annotations(
    annotations: list[AnnotationRow],
    coverage: CoverageReport,
    theme_of_code: dict[str, str],
    quotes_per_group: int,
    quote_max_chars: int,
    editor: QuoteEditor | None,
    min_chars: int,
) -> tuple[ExpectationGroup, ...]:
    """Сгруппировать аннотации по теме кода, иначе по коду."""
    buckets: dict[str, list[AnnotationRow]] = {}
    for annotation in annotations:
        code_name = coverage.code_names.get(annotation.code_id, "")
        key = theme_of_code.get(code_name) or (f"код {code_name}" if code_name else "без кода")
        buckets.setdefault(key, []).append(annotation)

    groups: list[ExpectationGroup] = []
    for title in sorted(buckets):
        items = sorted(buckets[title], key=lambda item: (item.utterance_id, item.start, item.id))
        codes = sorted({coverage.code_names.get(item.code_id, "") for item in items} - {""})
        quotes: list[Quote] = []
        for annotation in select_annotations(items, quotes_per_group, min_chars=min_chars):
            edited_text: str | None = None
            rule = ""
            if editor is not None:
                edited_text, _reason, rule = editor(quote_source_text(annotation))
            quotes.append(
                build_quote(
                    annotation,
                    coverage.code_names.get(annotation.code_id, ""),
                    quote_max_chars,
                    edited_text=edited_text,
                    edit_rule=rule,
                )
            )
        groups.append(
            ExpectationGroup(
                title=title,
                annotations=len(items),
                codes=tuple(codes),
                quotes=tuple(quotes),
            )
        )
    return tuple(groups)


def build_expectations(
    coverage: CoverageReport,
    settings: M8Settings,
    *,
    quotes_per_group: int = 2,
    quote_max_chars: int = 240,
    quote_min_chars: int = PREFERRED_MIN_CHARS,
    editor: QuoteEditor | None = None,
) -> Expectations:
    """Собрать раздел «Ожидания и рекомендации».

    Args:
        coverage: покрытие интервью (все аннотации, кроме отклонённых).
        settings: секция `m8` — списки кодов и вопросов.
        quotes_per_group: сколько цитат показывать в группе.
        quote_max_chars: предел длины цитаты.
        quote_min_chars: нижняя граница предпочтительной длины цитаты.
        editor: необязательная правка цитаты моделью.

    Returns:
        Два подраздела с детерминированной группировкой и числом пересечений:
        столько аннотаций попало в оба подраздела.
    """
    annotations = list(coverage.annotations)
    code_names = coverage.code_names
    autonomy_codes = {name.strip() for name in settings.autonomy_codes if name.strip()}
    advice_questions = {value.strip() for value in settings.advice_questions if value.strip()}
    theme_of_code = _theme_of_code(coverage)

    autonomy = [item for item in annotations if code_names.get(item.code_id, "") in autonomy_codes]
    advice = [
        item
        for item in annotations
        if item.question_identifier is not None and item.question_identifier in advice_questions
    ]
    overlap = len({item.id for item in autonomy} & {item.id for item in advice})

    sections = (
        ExpectationSection(
            title=AUTONOMY_TITLE,
            description="Аннотации с кодами из m8.autonomy_codes: что респондент готов отдать ИИ и на каких условиях.",
            selector="коды: " + (", ".join(sorted(autonomy_codes)) or "—"),
            annotations=len(autonomy),
            groups=_group_annotations(
                autonomy,
                coverage,
                theme_of_code,
                quotes_per_group,
                quote_max_chars,
                editor,
                quote_min_chars,
            ),
        ),
        ExpectationSection(
            title=ADVICE_TITLE,
            description="Аннотации к вопросам гайда из m8.advice_questions, без отбора по коду.",
            selector="вопросы: " + (", ".join(sorted(advice_questions)) or "—"),
            annotations=len(advice),
            groups=_group_annotations(
                advice,
                coverage,
                theme_of_code,
                quotes_per_group,
                quote_max_chars,
                editor,
                quote_min_chars,
            ),
        ),
    )
    return Expectations(sections=sections, overlap=overlap)


__all__ = [
    "ADVICE_TITLE",
    "AUTONOMY_TITLE",
    "ExpectationGroup",
    "ExpectationSection",
    "Expectations",
    "build_expectations",
]
