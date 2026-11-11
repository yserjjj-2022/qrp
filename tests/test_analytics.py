"""Тесты агрегации покрытия (M4).

Проверяется то, что нельзя увидеть в БД без агрегации: попадание в счётчики
только принятых аннотаций, границы статусов из конфига, применимость кодов по
группе и список вопросов без ответа. Все данные синтетические.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from qa.pipeline.aggregate import (
    STATUS_FEW,
    STATUS_HAS_DATA,
    STATUS_NO_DATA,
    AggregateError,
    build_coverage,
    cumulative_codes,
    theme_status,
    truncate_quote,
)
from qa.settings import Settings
from tests.fixtures.analytics_db import (
    AnnotationSpec,
    CodeSpec,
    InterviewSpec,
    QuestionSpec,
    build_analytics_db,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

CODES = [
    CodeSpec("R1", "Риски", ("8",)),
    CodeSpec("V1", "Проверка", ("9",)),
    CodeSpec("E0", "Ошибки исполнения", ("9Т",)),
    CodeSpec("A1", "Ответственность", ("13",)),
    CodeSpec("OTH", "Индуктивный", ()),
]

QUESTIONS = [
    QuestionSpec("8", "Все"),
    QuestionSpec("9", "Все"),
    QuestionSpec("9Т", "ТОРГ"),
    QuestionSpec("13", "Не торгующие"),
    QuestionSpec("20", "Все"),
]

THEMES = [
    ("Риски", "Риски", ["R1"]),
    ("Риски", "Проверка", ["V1"]),
    ("Риски", "Ошибки исполнения", ["E0"]),
    ("Ответственность", "Ответственность", ["A1"]),
    ("Индуктивные", "Индуктивный", ["OTH"]),
]

FRAGMENTS = [
    "Первый ответ про риски и проверку.",
    "Второй ответ тоже про риски.",
    "Третий ответ про ответственность.",
    "Четвёртый ответ вне тем.",
]


@pytest.fixture()
def settings() -> Settings:
    return Settings.load(_PROJECT_ROOT / "config.yaml")


def _build(
    tmp_path: Path,
    *,
    annotations: list[AnnotationSpec] | None = None,
    group: str = "Не торгующие",
    interview_code: str = "R01",
    fragments: list[str] | None = None,
) -> sqlite3.Connection:
    return build_analytics_db(
        tmp_path,
        themes=THEMES,
        codes=CODES,
        questions=QUESTIONS,
        interview=InterviewSpec(
            code=interview_code,
            group=group,
            fragments=fragments if fragments is not None else FRAGMENTS,
            # Четвёртая реплика — вне гайда: вопрос не привязан.
            questions={0: "8", 1: "9", 2: "13"},
            annotations=annotations or [],
            unmarked_fragments=(3,),
            doubtful_fragments=(1,),
            proposed=2,
        ),
    )


def _coverage(conn: sqlite3.Connection, settings: Settings, **overrides: object):  # noqa: ANN202
    params = {
        "low_min": settings.m4.theme_low_min_annotations,
        "high_min": settings.m4.theme_high_min_annotations,
        "quotes_per_theme": settings.m4.quotes_per_theme,
        "quote_max_chars": settings.m4.quote_max_chars,
    }
    params.update(overrides)
    return build_coverage(conn, "R01", settings.groups, **params)  # type: ignore[arg-type]


def test_status_boundaries_follow_config(settings: Settings) -> None:
    """Границы статусов берутся из конфига, а не зашиты в код."""
    low, high = settings.m4.theme_low_min_annotations, settings.m4.theme_high_min_annotations
    assert theme_status(0, low, high) == STATUS_NO_DATA
    assert theme_status(low, low, high) == STATUS_FEW
    assert theme_status(high - 1, low, high) == STATUS_FEW
    assert theme_status(high, low, high) == STATUS_HAS_DATA


def test_only_accepted_and_edited_are_counted(tmp_path: Path, settings: Settings) -> None:
    """`proposed` не входит в покрытие и выводится отдельным числом."""
    conn = _build(
        tmp_path,
        annotations=[AnnotationSpec(0, "R1", "про риски", status="accepted")],
    )
    report = _coverage(conn, settings)
    theme = report.nodes[0].themes[0]
    assert theme.annotations == 1
    assert report.quality.by_status["proposed"] == 2
    assert report.quality.by_status["accepted"] == 1
    conn.close()


def test_edited_counts_as_accepted(tmp_path: Path, settings: Settings) -> None:
    """Исправленная человеком аннотация участвует в агрегатах наравне с принятой."""
    conn = _build(tmp_path, annotations=[AnnotationSpec(0, "R1", "про риски", status="edited")])
    report = _coverage(conn, settings)
    assert report.nodes[0].themes[0].annotations == 1
    conn.close()


def test_theme_without_codes_and_code_without_theme(tmp_path: Path, settings: Settings) -> None:
    """Тема без кодов видна как «нет данных», код без темы уходит в неиспользованные."""
    conn = build_analytics_db(
        tmp_path,
        themes=[("Риски", "Риски", ["R1"]), ("Риски", "Пустая тема", [])],
        codes=[*CODES, CodeSpec("X1", "Без темы", ("8",))],
        questions=QUESTIONS,
        interview=InterviewSpec(fragments=FRAGMENTS),
    )
    report = build_coverage(
        conn,
        "R01",
        settings.groups,
        low_min=1,
        high_min=4,
        quotes_per_theme=2,
        quote_max_chars=240,
    )
    empty = [theme for node in report.nodes for theme in node.themes if theme.theme == "Пустая тема"]
    assert empty and empty[0].status == STATUS_NO_DATA
    assert empty[0].theme_codes == 0
    unused_names = [name for entry in report.codebook.unused for name in entry.names]
    assert "X1" in unused_names
    conn.close()


def test_code_applicability_by_question_group(tmp_path: Path, settings: Settings) -> None:
    """Код с вопросами только чужой группы помечается как неприменимый."""
    conn = _build(tmp_path)
    report = _coverage(conn, settings)
    not_applicable = [entry for entry in report.codebook.unused if not entry.applicable]
    applicable = [entry for entry in report.codebook.unused if entry.applicable]
    assert [name for entry in not_applicable for name in entry.names] == ["E0"]
    assert "V1" in [name for entry in applicable for name in entry.names]
    assert report.codebook.unused_not_applicable == 1
    conn.close()


def test_oth_without_questions_is_applicable(tmp_path: Path, settings: Settings) -> None:
    """Индуктивный код без вопросов считается применимым, а не чужим."""
    conn = _build(tmp_path)
    report = _coverage(conn, settings)
    oth = [entry for entry in report.codebook.unused if "OTH" in entry.names]
    assert oth and oth[0].applicable
    conn.close()


@pytest.mark.parametrize(
    "group",
    ["Не торгующие", "не торгующие", "не_торгующие", "не торг"],
)
def test_group_spelling_does_not_change_applicability(tmp_path: Path, settings: Settings, group: str) -> None:
    """Регистр, пробелы, подчёркивания и алиас из конфига дают один результат."""
    conn = _build(tmp_path, group=group)
    report = _coverage(conn, settings)
    assert report.codebook.unused_not_applicable == 1
    assert report.quality.questions_without_answer == ("20",)
    conn.close()


def test_questions_without_answer_skip_foreign_group(tmp_path: Path, settings: Settings) -> None:
    """Вопросы чужой группы не попадают в список без ответа."""
    conn = _build(tmp_path)
    report = _coverage(conn, settings)
    assert report.quality.questions_without_answer == ("20",)
    conn.close()


def test_doubtful_share(tmp_path: Path, settings: Settings) -> None:
    """Доля принятых аннотаций на doubtful-фрагментах считается отдельно."""
    conn = _build(
        tmp_path,
        annotations=[
            AnnotationSpec(1, "R1", "про риски", status="accepted"),
            AnnotationSpec(2, "A1", "ответственность", status="accepted"),
        ],
    )
    report = _coverage(conn, settings)
    assert report.quality.doubtful == 1
    assert report.quality.accepted_on_doubtful == 1
    assert report.quality.fragments_with_accepted_doubtful == 1
    conn.close()


def test_cumulative_follows_utterance_order(tmp_path: Path, settings: Settings) -> None:
    """Накопление кодов идёт по порядку реплик и считает различные коды."""
    conn = _build(
        tmp_path,
        annotations=[
            AnnotationSpec(0, "R1", "риски", status="accepted"),
            AnnotationSpec(0, "V1", "проверку", status="accepted"),
            AnnotationSpec(1, "R1", "риски", status="accepted"),
        ],
    )
    report = _coverage(conn, settings)
    risks = report.nodes[0].themes[0]
    assert risks.annotations == 2
    assert risks.distinct_codes == 1
    assert risks.cumulative == (1, 1)
    conn.close()


def test_cumulative_codes_unit() -> None:
    """Функция накопления не зависит от БД и считает различные коды."""
    from qa.db.analytics_repo import AcceptedAnnotationRow

    def row(code_id: int) -> AcceptedAnnotationRow:
        return AcceptedAnnotationRow(
            id=code_id,
            code_id=code_id,
            fragment_id=1,
            utterance_id=1,
            start=0,
            quote="",
            quote_start=None,
            quote_end=None,
            reliability=None,
            fragment_quality="ok",
            l2_text="",
            question_identifier=None,
        )

    assert cumulative_codes([]) == ()
    assert cumulative_codes([row(1), row(1), row(2)]) == (1, 1, 2)


def test_quotes_are_slices(tmp_path: Path, settings: Settings) -> None:
    """Цитата — срез L2 по позициям аннотации, а не сохранённая строка."""
    conn = _build(tmp_path, annotations=[AnnotationSpec(0, "R1", "риски")])
    report = _coverage(conn, settings)
    quote = report.nodes[0].themes[0].quotes[0]
    assert quote.text == "риски"
    assert quote.code_name == "R1"
    conn.close()


def test_truncate_quote_marks_cut() -> None:
    """Длинная цитата режется по границе слова и получает многоточие."""
    text = "Первый ответ про риски и проверку."
    truncated = truncate_quote(FRAGMENTS[0], 20)
    assert truncated.endswith(" …")
    assert truncated[:-2].rstrip() in text


def test_quotes_limited_by_config(tmp_path: Path, settings: Settings) -> None:
    """Число цитат на тему ограничено настройкой."""
    conn = _build(
        tmp_path,
        annotations=[
            AnnotationSpec(0, "R1", "риски", status="accepted"),
            AnnotationSpec(1, "R1", "риски", status="accepted"),
        ],
    )
    report = _coverage(conn, settings, quotes_per_theme=1)
    assert len(report.nodes[0].themes[0].quotes) == 1
    conn.close()


def test_unknown_interview_raises(tmp_path: Path, settings: Settings) -> None:
    """Отсутствующее интервью — явная ошибка, а не пустой дашборд."""
    conn = _build(tmp_path)
    with pytest.raises(AggregateError):
        build_coverage(
            conn,
            "R99",
            settings.groups,
            low_min=1,
            high_min=4,
            quotes_per_theme=2,
            quote_max_chars=240,
        )
    conn.close()


def test_interview_without_annotations(tmp_path: Path, settings: Settings) -> None:
    """Интервью без принятых аннотаций агрегируется: все темы «нет данных»."""
    conn = _build(tmp_path)
    report = _coverage(conn, settings)
    statuses = [theme.status for node in report.nodes for theme in node.themes]
    assert statuses and set(statuses) == {STATUS_NO_DATA}
    assert report.codebook.used == 0
    assert report.covered_fragments == 0
    conn.close()


def test_coding_counters_from_run(tmp_path: Path, settings: Settings) -> None:
    """Статусы фрагментов берутся из прогона кодирования, а не из всех прогонов."""
    conn = _build(tmp_path)
    report = _coverage(conn, settings)
    assert report.quality.coded == 3
    assert report.quality.unmarked == 1
    assert report.fragments_total == 4
    conn.close()


def test_covered_fragments_counts_distinct(tmp_path: Path, settings: Settings) -> None:
    """Два кода на одном фрагменте дают один покрытый фрагмент."""
    conn = _build(
        tmp_path,
        annotations=[
            AnnotationSpec(0, "R1", "риски", status="accepted"),
            AnnotationSpec(0, "V1", "проверку", status="accepted"),
        ],
    )
    report = _coverage(conn, settings)
    assert report.covered_fragments == 1
    conn.close()
