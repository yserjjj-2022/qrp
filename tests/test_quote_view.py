"""Тесты представления цитат (M8): отбор, обрезка, правка, показ оригинала.

Синтетические аннотации без БД: проверяются чистые функции и рендер отчёта.
Модель подменяется заглушкой, сети нет (CONSTITUTION §9).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from qa.db.analytics_repo import AnalyticsRepository, AnnotationRow
from qa.llm.base import LlmProvider, LlmResult
from qa.pipeline.aggregate import build_coverage
from qa.pipeline.quote_view import (
    CUT_MARKER,
    build_quote,
    check_quote_edit,
    edit_quote,
    quote_editor_from_settings,
    quote_source_text,
    select_annotations,
    truncate_quote,
)
from qa.report.model import QuotePolicy, build_report
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

LONG_TEXT = (
    "Я бы доверил искусственному интеллекту сверять данные и готовить черновики, "
    "но окончательное решение о сделке оставляю за собой."
)


@pytest.fixture()
def settings() -> Settings:
    return Settings.load(_PROJECT_ROOT / "config.yaml")


def _annotation(
    *,
    quote: str = "сверять данные",
    quality: str = "ok",
    reliability: int | None = 2,
    interpretation_only: bool = False,
    annotation_id: int = 1,
    timecode: int | None = 65,
    l1: str = "исходная реплика",
) -> AnnotationRow:
    """Аннотация с заданными свойствами отбора."""
    return AnnotationRow(
        id=annotation_id,
        code_id=1,
        fragment_id=1,
        utterance_id=1,
        start=0,
        quote=quote,
        quote_start=0,
        quote_end=len(quote),
        reliability=reliability,
        fragment_quality=quality,
        l2_text=quote,
        question_identifier="17",
        verified=True,
        interpretation_only=interpretation_only,
        timecode=timecode,
        l1_text=l1,
    )


def test_garbled_is_excluded() -> None:
    """Искажённый фрагмент не даёт цитату: текст нечитаем."""
    items = [_annotation(quality="garbled"), _annotation(quality="ok", annotation_id=2)]
    selected = select_annotations(items, 5)
    assert [item.id for item in selected] == [2]


def test_doubtful_goes_after_ok() -> None:
    """Сомнительный фрагмент допустим, но идёт после `ok`."""
    items = [
        _annotation(quality="doubtful", annotation_id=1, quote=LONG_TEXT),
        _annotation(quality="ok", annotation_id=2, quote="сверять"),
    ]
    selected = select_annotations(items, 5)
    assert [item.id for item in selected] == [2, 1]


def test_reliability_and_interpretation_order() -> None:
    """Надёжные и не «только интерпретация» идут раньше при прочих равных."""
    items = [
        _annotation(annotation_id=1, reliability=1, quote="сверять"),
        _annotation(annotation_id=2, reliability=3, interpretation_only=True, quote="сверять"),
        _annotation(annotation_id=3, reliability=3, quote="сверять"),
    ]
    selected = select_annotations(items, 5)
    assert [item.id for item in selected] == [3, 2, 1]


def test_limit_zero_gives_empty() -> None:
    """Ноль цитат — пустой список, без обращения к данным."""
    assert select_annotations([_annotation()], 0) == []


def test_truncate_marks_cut_with_brackets() -> None:
    """Обрезка помечается `[…]`, а не многоточием: это вырезка, а не авторский знак."""
    text, truncated = truncate_quote(LONG_TEXT, 40)
    assert truncated
    assert text.endswith(CUT_MARKER)
    assert "…" not in text.replace(CUT_MARKER, "")
    assert text.removesuffix(" " + CUT_MARKER) in LONG_TEXT


def test_truncate_keeps_short_text() -> None:
    """Короткий текст не трогается."""
    text, truncated = truncate_quote("коротко", 100)
    assert (text, truncated) == ("коротко", False)


def test_quote_source_prefers_positions() -> None:
    """Цитата — срез L2 по позициям, а не сохранённая строка."""
    annotation = AnnotationRow(
        id=1,
        code_id=1,
        fragment_id=1,
        utterance_id=1,
        start=0,
        quote="обрезанная строка",
        quote_start=2,
        quote_end=7,
        reliability=2,
        fragment_quality="ok",
        l2_text="текст единицы",
        question_identifier=None,
        verified=True,
        interpretation_only=False,
    )
    assert quote_source_text(annotation) == "кст е"


def test_build_quote_keeps_original_and_timecode() -> None:
    """У цитаты есть оригинал L2, оригинал L1 и тайм-код."""
    quote = build_quote(_annotation(quote=LONG_TEXT), "C1", 40)
    assert quote.code_name == "C1"
    assert quote.truncated
    assert quote.original_l2 == LONG_TEXT
    assert quote.original_l1 == "исходная реплика"
    assert quote.timecode == 65
    assert not quote.edited


def test_check_edit_rejects_changed_numbers() -> None:
    """Изменённое число — отказ: цифры в цитате менять нельзя."""
    assert check_quote_edit("ставка 12 процентов", "ставка 13 процентов") is not None


def test_check_edit_rejects_lost_negation() -> None:
    """Потерянное отрицание переворачивает смысл."""
    assert check_quote_edit("я не доверяю этому", "я доверяю этому") is not None


def test_check_edit_allows_bracket_insertions() -> None:
    """Пояснение в квадратных скобках не считается переписыванием текста."""
    source = (
        "Я бы доверил искусственному интеллекту сверять данные и готовить черновики, "
        "но окончательное решение оставляю за собой."
    )
    assert check_quote_edit(source, source + " [курсив мой]") is None


def test_check_edit_rejects_new_latin() -> None:
    """Латиница, которой не было в оригинале, — отказ."""
    assert check_quote_edit("сверять данные", "сверять data") is not None


def test_check_edit_rejects_too_much_change() -> None:
    """Переписывание текста сверх порога — отказ."""
    source = "сверять данные и готовить черновики"
    assert check_quote_edit(source, "полностью другой текст без связи с исходным") is not None


def test_edit_quote_without_provider() -> None:
    """Без модели правка не выполняется и оригинал остаётся."""
    edited, reason, _rule = edit_quote("сверять данные", None)
    assert edited is None
    assert reason


class _Provider(LlmProvider):
    """Заглушка: отдаёт заданный ответ один раз."""

    def __init__(self, edited: str) -> None:
        self._edited = edited
        self.calls = 0

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        return self.complete_json_detailed(prompt, schema).data

    def complete_json_detailed(self, prompt: str, schema: dict[str, Any]) -> LlmResult:
        self.calls += 1
        return LlmResult(data={"edited_text": self._edited, "reason": "пунктуация"})

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


def test_edit_quote_applies_valid_edit() -> None:
    """Прошедшая проверки правка возвращается вместе с названием правила."""
    provider = _Provider("Сверять данные.")
    edited, reason, rule = edit_quote("Сверять данные", provider)
    assert edited == "Сверять данные."
    assert reason == ""
    assert "промпт" in rule
    assert provider.calls == 1


def test_edit_quote_drops_invalid_edit() -> None:
    """Не прошедшая проверку правка отбрасывается."""
    provider = _Provider("Совсем другой смысл без отрицания")
    edited, reason, _rule = edit_quote("я не доверяю этому решению", provider)
    assert edited is None
    assert reason


def test_quote_policy_note_without_changes() -> None:
    """Без обрезки и правки отчёт говорит, что цитаты дословные."""
    note = QuotePolicy().note
    assert "дословно" in note


def test_quote_policy_note_lists_rules() -> None:
    """С правкой отчёт называет правило."""
    note = QuotePolicy(truncated=2, edited=1, rules=("правило A",)).note
    assert "обрезка отмечена […]: 2" in note
    assert "правило A" in note


def _render_single_quote(tmp_path: Path, settings: Settings, *, text: str, doubtful: bool) -> str:
    """Собрать отчёт с одной цитатой и вернуть HTML."""
    conn = build_analytics_db(
        tmp_path,
        themes=[("Ответственность", "Контроль", ["C1"])],
        codes=[CodeSpec("C1", "Контроль", ("17",))],
        questions=[QuestionSpec("17", "Все")],
        interview=InterviewSpec(
            fragments=[text],
            questions={0: "17"},
            annotations=[AnnotationSpec(0, "C1", text)],
            doubtful_fragments=(0,) if doubtful else (),
        ),
    )
    coverage = build_coverage(
        conn,
        "R01",
        settings.groups,
        low_min=1,
        high_min=4,
        quotes_per_theme=2,
        quote_max_chars=40,
    )
    report = build_report(coverage, AnalyticsRepository(conn), snapshot_date="2026-11-11")
    conn.close()
    return render_report(report)


def test_render_shows_timecode_and_original_when_cut(tmp_path: Path, settings: Settings) -> None:
    """У обрезанной цитаты в отчёте есть тайм-код и дословный оригинал."""
    html = _render_single_quote(tmp_path, settings, text=LONG_TEXT, doubtful=False)
    assert "тайм-код" in html
    assert "оригинал L2" in html
    assert CUT_MARKER in html
    assert LONG_TEXT in html


def test_render_marks_doubtful(tmp_path: Path, settings: Settings) -> None:
    """Сомнительный фрагмент помечен в отчёте."""
    html = _render_single_quote(tmp_path, settings, text="сверять данные", doubtful=True)
    assert "фрагмент doubtful" in html


def test_quote_edit_disabled_by_default(settings: Settings) -> None:
    """Правка цитат выключена по умолчанию: редактор не создаётся."""
    assert settings.m8.quote_edit_enabled is False
    assert quote_editor_from_settings(settings) is None
