"""Тесты раздела отчёта «Ожидания и рекомендации» (M8).

Раздел собран из двух подразделов с разными основаниями отбора: коды из
`m8.autonomy_codes` и вопросы гайда из `m8.advice_questions`. Счётчики
подразделов не суммируются, пересечение показывается отдельно
(SPEC-M8, решение 2). Проверяется именно это, а не вёрстка.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from qa.db.analytics_repo import AnalyticsRepository
from qa.pipeline.aggregate import CoverageReport, build_coverage
from qa.pipeline.expectations import (
    ADVICE_TITLE,
    AUTONOMY_TITLE,
    build_expectations,
)
from qa.report.model import build_report
from qa.report.render import render_report
from qa.settings import M8Settings, Settings
from tests.fixtures.m8_db import AnnotationSpec, M8Spec, UtteranceSpec, build_m8_db

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Раскладка: C1 на вопросах 17 и 22, C2 на 5, C4 на 18, B3 на 5.
# Autonomy-коды C1/C2/C4 дают 4 аннотации, вопросы 17/18/22 — 3,
# пересечение (C1·17, C4·18, C1·22) — 3 аннотации.
SPEC = M8Spec(
    utterances=[
        UtteranceSpec("RESPONDENT", "Сверять данные я готов.", "Сверять данные я готов.", 65, "17"),
        UtteranceSpec("RESPONDENT", "Черновики тоже можно.", "Черновики тоже можно.", 130, "5"),
        UtteranceSpec("RESPONDENT", "Окончательное решение за мной.", "Окончательное решение за мной.", 195, "18"),
        UtteranceSpec("RESPONDENT", "Это про основание сделки.", "Это про основание сделки.", 260, "5"),
        UtteranceSpec("RESPONDENT", "Попробовать можно на демо.", "Попробовать можно на демо.", 325, "22"),
    ],
    annotations=[
        AnnotationSpec(0, "C1", "Сверять данные"),
        AnnotationSpec(1, "C2", "Черновики"),
        AnnotationSpec(2, "C4", "Окончательное решение"),
        AnnotationSpec(3, "B3", "основание сделки"),
        AnnotationSpec(4, "C1", "Попробовать можно"),
    ],
)


def _m8() -> M8Settings:
    """Настройки раздела: три кода и три вопроса."""
    return M8Settings(autonomy_codes=["C1", "C2", "C4"], advice_questions=["17", "18", "22"])


@pytest.fixture()
def settings() -> Settings:
    return Settings.load(_PROJECT_ROOT / "config.yaml")


def _coverage(tmp_path: Path, settings: Settings) -> tuple[sqlite3.Connection, CoverageReport]:
    conn = build_m8_db(tmp_path, SPEC)
    coverage = build_coverage(
        conn,
        "R01",
        settings.groups,
        low_min=settings.m4.theme_low_min_annotations,
        high_min=settings.m4.theme_high_min_annotations,
        quotes_per_theme=settings.m4.quotes_per_theme,
        quote_max_chars=settings.m4.quote_max_chars,
    )
    return conn, coverage


def test_sections_select_by_codes_and_questions(tmp_path: Path, settings: Settings) -> None:
    """Первый подраздел — по кодам, второй — по вопросам гайда."""
    conn, coverage = _coverage(tmp_path, settings)
    expectations = build_expectations(coverage, _m8())
    conn.close()

    autonomy, advice = expectations.sections
    assert autonomy.title == AUTONOMY_TITLE
    assert advice.title == ADVICE_TITLE
    assert autonomy.annotations == 4
    assert advice.annotations == 3
    assert autonomy.selector == "коды: C1, C2, C4"
    assert advice.selector == "вопросы: 17, 18, 22"


def test_counts_are_not_summed(tmp_path: Path, settings: Settings) -> None:
    """`total` — сумма подразделов, а пересечение считается отдельно."""
    conn, coverage = _coverage(tmp_path, settings)
    expectations = build_expectations(coverage, _m8())
    conn.close()

    assert expectations.total == 7
    assert expectations.overlap == 3


def test_groups_follow_themes(tmp_path: Path, settings: Settings) -> None:
    """Группировка — по теме кода, а не по коду."""
    conn, coverage = _coverage(tmp_path, settings)
    expectations = build_expectations(coverage, _m8())
    conn.close()

    autonomy = expectations.sections[0]
    assert [group.title for group in autonomy.groups] == ["Ответственность → Контроль"]
    control = autonomy.groups[0]
    assert control.annotations == 4
    assert control.codes == ("C1", "C2", "C4")
    assert control.quotes and control.quotes[0].code_name in {"C1", "C2", "C4"}


def test_advice_groups_include_all_codes(tmp_path: Path, settings: Settings) -> None:
    """Во втором подразделе нет отбора по коду: в группах разные коды."""
    conn, coverage = _coverage(tmp_path, settings)
    expectations = build_expectations(coverage, _m8())
    conn.close()

    advice = expectations.sections[1]
    codes = {code for group in advice.groups for code in group.codes}
    assert codes == {"C1", "C4"}


def test_empty_lists_give_empty_sections(tmp_path: Path, settings: Settings) -> None:
    """Пустые списки — подразделы на месте, но без аннотаций."""
    conn, coverage = _coverage(tmp_path, settings)
    expectations = build_expectations(coverage, M8Settings(autonomy_codes=[], advice_questions=[]))
    conn.close()

    assert [section.annotations for section in expectations.sections] == [0, 0]
    assert all(not section.groups for section in expectations.sections)
    assert expectations.overlap == 0


def test_report_without_settings_has_no_section(tmp_path: Path, settings: Settings) -> None:
    """Без настроек раздел пуст: базовый отчёт M5 не меняется."""
    conn, coverage = _coverage(tmp_path, settings)
    report = build_report(coverage, AnalyticsRepository(conn), snapshot_date="2026-11-11")
    conn.close()

    assert report.expectations.sections == ()
    assert "Раздел не настроен" in render_report(report)


def test_report_renders_two_sections_and_overlap(tmp_path: Path, settings: Settings) -> None:
    """В отчёте два подраздела, число каждого и строка о пересечении."""
    configured = settings.model_copy(update={"m8": _m8()})
    conn, coverage = _coverage(tmp_path, settings)
    report = build_report(
        coverage,
        AnalyticsRepository(conn),
        snapshot_date="2026-11-11",
        settings=configured,
    )
    conn.close()

    html = render_report(report)
    assert f"{AUTONOMY_TITLE} — 4 аннотации" in html
    assert f"{ADVICE_TITLE} — 3 аннотации" in html
    assert "3 аннотации входят в оба подраздела" in html
    assert "7 аннотаций" not in html


def test_report_quotes_carry_timecode(tmp_path: Path, settings: Settings) -> None:
    """Цитаты подразделов показывают тайм-код, как и цитаты тем."""
    configured = settings.model_copy(update={"m8": _m8()})
    conn, coverage = _coverage(tmp_path, settings)
    report = build_report(
        coverage,
        AnalyticsRepository(conn),
        snapshot_date="2026-11-11",
        settings=configured,
    )
    conn.close()

    quotes = [quote for section in report.expectations.sections for group in section.groups for quote in group.quotes]
    assert quotes
    assert all(quote.timecode is not None for quote in quotes)
    assert all(quote.original_l2 for quote in quotes)
