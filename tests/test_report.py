"""Тесты отчёта (M5): модель, разделение слоёв и рендер.

Проверяется то, что видно только в собранном артефакте: порядок блоков,
формулировка про ручную проверку, наличие ограничений и отсутствие импорта
рендера из модели отчёта.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from qa.db.analytics_repo import AnalyticsRepository
from qa.pipeline.aggregate import build_coverage
from qa.report.model import InterviewReport, build_report
from qa.report.render import render_report
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
    CodeSpec("OTH", "Индуктивный", ()),
]

QUESTIONS = [
    QuestionSpec("8", "Все"),
    QuestionSpec("9", "Все"),
    QuestionSpec("9Т", "ТОРГ"),
    QuestionSpec("20", "Все"),
]

THEMES = [
    ("Риски", "Риски", ["R1"]),
    ("Риски", "Проверка", ["V1"]),
    ("Индуктивные", "Индуктивный", ["OTH"]),
]

FRAGMENTS = [
    "Первый ответ про риски.",
    "Второй ответ про проверку.",
    "Третий ответ вне тем.",
]


@pytest.fixture()
def settings() -> Settings:
    return Settings.load(_PROJECT_ROOT / "config.yaml")


def _report(
    tmp_path: Path,
    settings: Settings,
    *,
    annotations: list[AnnotationSpec] | None = None,
) -> InterviewReport:
    conn = build_analytics_db(
        tmp_path,
        themes=THEMES,
        codes=CODES,
        questions=QUESTIONS,
        interview=InterviewSpec(
            fragments=FRAGMENTS,
            questions={0: "8", 1: "9"},
            annotations=annotations or [],
            unmarked_fragments=(2,),
            proposed=1,
        ),
    )
    coverage = build_coverage(
        conn,
        "R01",
        settings.groups,
        low_min=settings.m4.theme_low_min_annotations,
        high_min=settings.m4.theme_high_min_annotations,
        quotes_per_theme=settings.m4.quotes_per_theme,
        quote_max_chars=settings.m4.quote_max_chars,
    )
    report = build_report(coverage, AnalyticsRepository(conn), snapshot_date="2026-11-11")
    conn.close()
    return report


def test_provenance_has_transcript_type_and_snapshot(tmp_path: Path, settings: Settings) -> None:
    """Происхождение содержит тип расшифровки, дату и прогон снимка."""
    report = _report(tmp_path, settings)
    provenance = report.provenance
    assert provenance.transcript_type == "машинная расшифровка с лёгкой редакцией"
    assert provenance.snapshot_date == "2026-11-11"
    assert provenance.snapshot_run_id == 1
    assert provenance.group == "Не торгующие"
    assert provenance.runs and provenance.runs[0].stage == "coding"


def test_model_marking_line_in_limitations(tmp_path: Path, settings: Settings) -> None:
    """Ограничения содержат строку о доле ручной проверки."""
    report = _report(tmp_path, settings, annotations=[AnnotationSpec(0, "R1", "риски")])
    texts = [item.text for item in report.limitations]
    assert any("Разметка модели; 1 из 2 аннотаций проверены вручную" in text for text in texts)


def test_limitations_cover_required_points(tmp_path: Path, settings: Settings) -> None:
    """Блок ограничений содержит обязательные пункты SPEC-M6."""
    codes = {item.code for item in _report(tmp_path, settings).limitations}
    assert {
        "snapshot",
        "model_marking",
        "skipped_garbled",
        "interpretation_only",
        "reliability_cap",
        "doubtful_share",
        "unmarked",
        "oth",
        "codebook_coverage",
        "one_interview",
        "reference_fact",
        "leaf_status",
    } <= codes


def test_model_does_not_import_render() -> None:
    """Модель отчёта не зависит от рендера и не содержит HTML."""
    source = (_PROJECT_ROOT / "src/qa/report/model.py").read_text(encoding="utf-8")
    imports = [line for line in source.splitlines() if line.startswith(("from ", "import "))]
    assert not any("render" in line for line in imports)
    assert "<div" not in source


def test_render_blocks_follow_order(tmp_path: Path, settings: Settings) -> None:
    """Порядок блоков отчёта: происхождение, числа, дерево, цитаты, ожидания, ограничения."""
    html = render_report(_report(tmp_path, settings))
    positions = [
        html.index("1. Происхождение"),
        html.index("2. Числа"),
        html.index("3. Дерево тем"),
        html.index("4. Цитаты"),
        html.index("5. Ожидания и рекомендации"),
        html.index("6. Ограничения"),
    ]
    assert positions == sorted(positions)


def test_report_has_no_script_and_no_external_resources(tmp_path: Path, settings: Settings) -> None:
    """Отчёт открывается офлайн: нет JS и внешних ссылок."""
    html = render_report(_report(tmp_path, settings))
    assert "<script" not in html.lower()
    assert re.findall(r'(?:src|href)="([^"]+)"', html) == []


def test_report_without_annotations_says_so(tmp_path: Path, settings: Settings) -> None:
    """Отчёт без аннотаций собирается и прямо говорит об этом."""
    conn = build_analytics_db(
        tmp_path,
        themes=THEMES,
        codes=CODES,
        questions=QUESTIONS,
        interview=InterviewSpec(fragments=FRAGMENTS, questions={0: "8", 1: "9"}),
    )
    coverage = build_coverage(
        conn,
        "R01",
        settings.groups,
        low_min=1,
        high_min=4,
        quotes_per_theme=2,
        quote_max_chars=240,
    )
    report = build_report(coverage, AnalyticsRepository(conn), snapshot_date="2026-11-11")
    html = render_report(report)
    assert "Аннотаций нет: дерево пусто." in html
    assert "Цитат нет: аннотаций по темам нет." in html
    conn.close()


def test_unused_codes_listed_without_group_split(tmp_path: Path, settings: Settings) -> None:
    """Неиспользованные коды идут одним списком по семьям, без пометки неприменимых."""
    html = render_report(_report(tmp_path, settings))
    assert "не применимо" not in html
    assert "V1" in html and "E0" in html and "OTH" in html
