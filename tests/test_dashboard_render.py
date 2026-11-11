"""Тесты рендера дашборда (M4).

Проверяется то, что видно только в готовом HTML: три плашки статусов с
подписью «эвристика, 1 интервью», один список неиспользованных кодов с
пометкой неприменимых и отсутствие JavaScript и внешних ресурсов.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from qa.dashboard.render import render_dashboard
from qa.pipeline.aggregate import CoverageReport, build_coverage
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
    CodeSpec("R2", "Риски", ("8",)),
    CodeSpec("V1", "Проверка", ("9",)),
    CodeSpec("E0", "Ошибки исполнения", ("9Т",)),
    CodeSpec("OTH", "Индуктивный", ()),
]

QUESTIONS = [
    QuestionSpec("8", "Все"),
    QuestionSpec("9", "Все"),
    QuestionSpec("9Т", "ТОРГ"),
    QuestionSpec("20", "Все"),
]

THEMES = [
    ("Риски", "Риски", ["R1", "R2"]),
    ("Риски", "Проверка", ["V1"]),
    ("Риски", "Ошибки исполнения", ["E0"]),
    ("Индуктивные", "Индуктивный", ["OTH"]),
]

FRAGMENTS = [
    "Первый ответ про риски.",
    "Второй ответ про риски.",
    "Третий ответ про проверку.",
    "Четвёртый ответ про риски.",
]


@pytest.fixture()
def settings() -> Settings:
    return Settings.load(_PROJECT_ROOT / "config.yaml")


def _report(tmp_path: Path, settings: Settings, *, annotations: list[AnnotationSpec] | None = None) -> CoverageReport:
    conn = build_analytics_db(
        tmp_path,
        themes=THEMES,
        codes=CODES,
        questions=QUESTIONS,
        interview=InterviewSpec(
            fragments=FRAGMENTS,
            questions={0: "8", 1: "8", 2: "9"},
            annotations=annotations or [],
            doubtful_fragments=(3,),
            proposed=5,
        ),
    )
    report = build_coverage(
        conn,
        "R01",
        settings.groups,
        low_min=settings.m4.theme_low_min_annotations,
        high_min=settings.m4.theme_high_min_annotations,
        quotes_per_theme=settings.m4.quotes_per_theme,
        quote_max_chars=settings.m4.quote_max_chars,
    )
    conn.close()
    return report


def test_three_status_pills_and_caption(tmp_path: Path, settings: Settings) -> None:
    """В HTML есть все три статуса и подпись эвристики рядом с каждым листом."""
    report = _report(
        tmp_path,
        settings,
        annotations=[
            AnnotationSpec(0, "R1", "риски"),
            AnnotationSpec(1, "R1", "риски"),
            AnnotationSpec(3, "R2", "риски"),
            AnnotationSpec(0, "R2", "риски"),
            AnnotationSpec(2, "V1", "проверку"),
        ],
    )
    html = render_dashboard(report, "2026-11-10")
    assert "есть данные" in html
    assert "мало" in html
    assert "нет данных" in html
    assert html.count("эвристика, 1 интервью") >= 4


def test_status_wording_is_neutral(tmp_path: Path, settings: Settings) -> None:
    """Названий, оценивающих раскрытие темы, в тексте нет."""
    html = render_dashboard(_report(tmp_path, settings), "2026-11-10")
    for word in ("насыщено", "насыщение"):
        assert word not in html


def test_unused_codes_in_one_list_with_marks(tmp_path: Path, settings: Settings) -> None:
    """Неиспользованные коды идут одним списком; неприменимые помечены отдельно."""
    html = render_dashboard(_report(tmp_path, settings), "2026-11-10")
    assert "Использовано 0 из 5 кодов. Не использовано 5." in html
    assert html.count('class="chip restricted"') == 1
    assert 'class="chip">R1<' in html
    assert "Пунктиром: 1 кодов не применимы" in html


def test_no_script_and_no_external_resources(tmp_path: Path, settings: Settings) -> None:
    """Дашборд открывается офлайн: нет JS и внешних ссылок."""
    html = render_dashboard(_report(tmp_path, settings), "2026-11-10")
    assert "<script" not in html.lower()
    assert re.findall(r'(?:src|href)="([^"]+)"', html) == []


def test_doubtful_share_rendered(tmp_path: Path, settings: Settings) -> None:
    """Доля принятых аннотаций на doubtful выводится в блоке качества."""
    report = _report(
        tmp_path,
        settings,
        annotations=[AnnotationSpec(3, "R1", "риски")],
    )
    html = render_dashboard(report, "2026-11-10")
    assert "Принятые аннотации на doubtful" in html
    assert "1 из 1" in html


def test_snapshot_and_run_in_header(tmp_path: Path, settings: Settings) -> None:
    """Шапка подписана датой снимка и номером прогона, proposed не учтены."""
    html = render_dashboard(_report(tmp_path, settings), "2026-11-10")
    assert "2026-11-10" in html
    assert "run 1" in html
    assert "5 proposed не учтены" in html


def test_theme_without_annotations_says_so(tmp_path: Path, settings: Settings) -> None:
    """У темы без принятых аннотаций стоит явная строка, а не пустой блок."""
    html = render_dashboard(_report(tmp_path, settings), "2026-11-10")
    assert "Цитат нет: принятых аннотаций по теме нет." in html
