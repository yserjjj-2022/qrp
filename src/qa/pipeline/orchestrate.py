"""Команда `run`: выбор следующего шага конвейера и остановка на воротах (M8).

`run` не содержит SQL и не дублирует логику стадий: состояние вычисляется
репозиторием `qa.db.pipeline_repo`, а работа выполняется существующими
функциями стадий. Отдельной таблицы статусов нет — повторный запуск безопасен
именно потому, что каждая стадия сама проверяет, сделана ли она.

С M7b в пути по умолчанию нет ворот терминов, шага таблицы терминов, выборки
качества и разовых правок: термины опознаёт память решений (`apply-terms`).
Осталась одна остановка — перед сегментацией, если после нормализации есть
слова для оператора: `Помогите опознать N слов: <файл>`. Флаг `--no-wait`
проходит дальше и применяет правила пустой ячейки.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from qa.db.pipeline_repo import PipelineRepository
from qa.llm.base import LlmProvider
from qa.settings import Settings

logger = logging.getLogger(__name__)

STATUS_DONE = "done"
STATUS_SKIPPED = "skipped"
STATUS_GATE = "gate"
STATUS_BLOCKED = "blocked"


@dataclass
class StepResult:
    """Итог одного шага конвейера."""

    number: int
    title: str
    status: str
    detail: str = ""
    lines: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    """Итог команды `run`."""

    interview_code: str
    steps: list[StepResult]
    stopped_at_gate: int | None = None
    all_done: bool = False

    @property
    def performed(self) -> list[StepResult]:
        """Шаги, которые действительно что-то сделали."""
        return [step for step in self.steps if step.status == STATUS_DONE]


class RunError(Exception):
    """Шаг конвейера невозможен: нет файла, ошибка стадии."""


def _fmt_timecode(seconds: int | None) -> str:
    """Тайм-код вида `mm:ss` для сообщений."""
    from qa.pipeline.terms_table import format_timecode

    return format_timecode(seconds)


def _mtime(path: Path) -> float:
    """Время изменения файла или 0, если файла нет."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _parse_started_at(value: str | None) -> float:
    """Время последнего прогона в epoch (SQLite хранит UTC)."""
    if not value:
        return 0.0
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    except ValueError:
        return 0.0
    return parsed.timestamp()


def _artifact_is_fresh(path: Path, run_started_at: str | None) -> bool:
    """Файл новее последнего прогона (или прогонов ещё не было)."""
    if not path.is_file():
        return False
    last_run = _parse_started_at(run_started_at)
    if last_run <= 0:
        return True
    return _mtime(path) >= last_run


def _transcript_path(settings: Settings, interview_code: str) -> Path | None:
    """Файл транскрипта интервью по каталогу из настроек."""
    directory = Path(settings.paths.transcripts)
    for suffix in (".txt", ".TXT"):
        candidate = directory / f"{interview_code}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _provider(settings: Settings) -> LlmProvider | None:
    """Провайдер LLM из настроек; None для заглушки."""
    if settings.llm.provider == "stub":
        return None
    from qa.llm import provider_from_settings

    return provider_from_settings(settings)


class PipelineRunner:
    """Выполнение шагов конвейера по состоянию БД."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        interview_code: str,
        settings: Settings,
        *,
        skip_terms: bool = False,
        transcripts_dir: Path | None = None,
        no_wait: bool = False,
    ) -> None:
        self._conn = conn
        self._code = interview_code
        self._settings = settings
        self._repo = PipelineRepository(conn)
        # Флаг сохранён для совместимости вызовов: ворот терминов в run больше
        # нет, значение ни на что не влияет.
        self._skip_terms = skip_terms
        self._transcripts_dir = transcripts_dir
        self._no_wait = no_wait
        self._open_forms = 0
        self._terms_path = ""
        self._interview_db_id: int | None = None
        self._results: list[StepResult] = []

    @property
    def interview_db_id(self) -> int | None:
        """id интервью в БД (None, пока интервью не загружено)."""
        if self._interview_db_id is None:
            self._interview_db_id = self._repo.interview_db_id(self._code)
        return self._interview_db_id

    def plan(self) -> list[StepResult]:
        """План шагов без выполнения: что уже сделано и что предстоит."""
        plan: list[StepResult] = []
        for number, title in enumerate(self._titles(), start=1):
            status, detail = self._evaluate(number)
            plan.append(StepResult(number=number, title=title, status=status, detail=detail))
        return plan

    @staticmethod
    def _titles() -> tuple[str, ...]:
        """Названия шагов в порядке выполнения."""
        return (
            "книга",
            "проект",
            "интервью",
            "L2 (нормализация)",
            "фрагменты",
            "кодирование",
            "дашборд",
            "отчёт",
        )

    def _evaluate(self, number: int) -> tuple[str, str]:
        """Состояние шага: сделан, предстоит, ворота или заблокирован."""
        repo = self._repo
        if number == 1:
            count = repo.codebook_versions()
            return (STATUS_SKIPPED, f"версий кодбука: {count}") if count else (STATUS_DONE, "загрузить кодбук")
        if number == 2:
            count = repo.lexicon_terms()
            return (STATUS_SKIPPED, f"терминов в лексиконе: {count}") if count else (STATUS_DONE, "загрузить проект")
        if number == 3:
            interview_db_id = self.interview_db_id
            if interview_db_id is None:
                return STATUS_DONE, "интервью нет в БД"
            turns = repo.turn_count(interview_db_id)
            return (STATUS_SKIPPED, f"записей транскрипта: {turns}") if turns else (STATUS_DONE, "загрузить транскрипт")
        if number == 4:
            interview_db_id = self.interview_db_id
            if interview_db_id is None:
                return STATUS_BLOCKED, "сначала загрузите интервью"
            units = repo.text_unit_count(interview_db_id)
            return (STATUS_SKIPPED, f"единиц L2: {units}") if units else (STATUS_DONE, "нормализовать")
        if number == 5:
            interview_db_id = self.interview_db_id
            if interview_db_id is None:
                return STATUS_BLOCKED, "сначала загрузите интервью"
            fragments = repo.fragment_count(interview_db_id)
            return (STATUS_SKIPPED, f"фрагментов: {fragments}") if fragments else (STATUS_DONE, "разбить на фрагменты")
        if number == 6:
            interview_db_id = self.interview_db_id
            if interview_db_id is None:
                return STATUS_BLOCKED, "сначала загрузите интервью"
            annotations = repo.annotation_count(interview_db_id)
            return (STATUS_SKIPPED, f"аннотаций: {annotations}") if annotations else (STATUS_DONE, "закодировать")
        if number == 7:
            path = Path(self._settings.paths.exports) / f"dashboard_{self._code}.html"
            if _artifact_is_fresh(path, self._repo.latest_run_started_at()):
                return STATUS_SKIPPED, "файл новее последнего прогона"
            return STATUS_DONE, "собрать дашборд"
        path = Path(self._settings.paths.reports) / f"report_{self._code}.html"
        if _artifact_is_fresh(path, self._repo.latest_run_started_at()):
            return STATUS_SKIPPED, "файл новее последнего прогона"
        return STATUS_DONE, "собрать отчёт"

    def execute(self) -> RunResult:
        """Выполнять шаги по порядку, остановившись на блокирующих воротах."""
        titles = self._titles()
        for number in range(1, len(titles) + 1):
            title = titles[number - 1]
            status, detail = self._evaluate(number)
            if status in (STATUS_SKIPPED, STATUS_BLOCKED):
                self._results.append(StepResult(number, title, status, detail))
                if status == STATUS_BLOCKED and number in (3, 4, 5, 6):
                    # Дальше идти нельзя: шаг невозможен без предыдущего.
                    return RunResult(self._code, self._results, stopped_at_gate=None, all_done=False)
                continue
            result = self._perform(number, title)
            self._results.append(result)
            if result.status == STATUS_BLOCKED:
                return RunResult(self._code, self._results, stopped_at_gate=None, all_done=False)
            if number == 4 and result.status == STATUS_DONE and self._should_wait():
                # Остановка перед сегментацией: фрагменты строятся по тексту,
                # и опознавать слова после их появления уже поздно.
                self._results.append(
                    StepResult(
                        number + 1,
                        titles[number],
                        STATUS_GATE,
                        f"Помогите опознать {self._open_forms} слов: {self._terms_path}",
                    )
                )
                return RunResult(self._code, self._results, stopped_at_gate=number + 1, all_done=False)
        return RunResult(self._code, self._results, stopped_at_gate=None, all_done=True)

    def _should_wait(self) -> bool:
        """Есть ли причина остановиться перед сегментацией."""
        return bool(self._open_forms) and self._settings.m7b.wait_on_terms and not self._no_wait

    def _perform(self, number: int, title: str) -> StepResult:
        """Выполнить шаг через существующую функцию стадии."""
        try:
            if number == 1:
                return self._step_codebook(number, title)
            if number == 2:
                return self._step_project(number, title)
            if number == 3:
                return self._step_interview(number, title)
            if number == 4:
                return self._step_normalize(number, title)
            if number == 5:
                return self._step_structure(number, title)
            if number == 6:
                return self._step_code(number, title)
            if number == 7:
                return self._step_dashboard(number, title)
            return self._step_report(number, title)
        except Exception as exc:  # noqa: BLE001 - стадии бросают разные типы ошибок
            logger.info("run: шаг %d не выполнен: %s", number, exc)
            return StepResult(number, title, STATUS_BLOCKED, str(exc))

    def _step_codebook(self, number: int, title: str) -> StepResult:
        """Загрузить книгу и кодбук."""
        from qa.pipeline.load_codebook import load_codebook

        result = load_codebook(
            self._conn,
            Path(self._settings.paths.workbook),
            self._settings.workbook,
            reload=False,
            groups=self._settings.groups,
        )
        counts = result.counts
        detail = (
            f"кодов={counts.get('codes', 0)}, вопросов={counts.get('questions', 0)}, "
            f"словарь={counts.get('glossary', 0)}"
        )
        return StepResult(number, title, STATUS_DONE, detail)

    def _step_project(self, number: int, title: str) -> StepResult:
        """Загрузить необязательные листы проекта."""
        from qa.pipeline.load_project import load_project

        result = load_project(
            self._conn,
            Path(self._settings.paths.workbook),
            self._settings.workbook,
            groups=self._settings.groups,
            lexicon_status_map=self._settings.lexicon_status_map,
            pipeline=self._settings.pipeline,
            transcripts_dir=Path(self._settings.paths.transcripts),
        )
        return StepResult(number, title, STATUS_DONE, f"записей: {sum(result.counts.values())}")

    def _step_interview(self, number: int, title: str) -> StepResult:
        """Определить роли и загрузить транскрипт."""
        from qa.pipeline.init_interview import apply_plan, build_plan, plan_to_meta, resolve_names_map
        from qa.pipeline.load_transcript import build_replacement_map, load_transcript

        path = _transcript_path(self._settings, self._code)
        if path is None:
            return StepResult(
                number,
                title,
                STATUS_BLOCKED,
                f"файл транскрипта {self._code}.txt не найден в {self._settings.paths.transcripts}",
            )
        plan = build_plan(self._conn, path, None, self._settings.groups)
        if plan.needs_confirmation:
            # run не спрашивает подтверждения: роли уже должны быть в книге или в БД.
            return StepResult(
                number,
                title,
                STATUS_BLOCKED,
                "роли не подтверждены: выполните init-interview и подтвердите роли, затем повторите run",
            )
        apply_plan(self._conn, plan, self._settings.groups)
        meta = plan_to_meta(plan, resolve_names_map())
        replacement_map = build_replacement_map(meta, resolve_names_map())
        result = load_transcript(
            self._conn,
            path,
            meta,
            replacement_map,
            self._settings.pipeline,
            groups=self._settings.groups,
        )
        if result.skipped:
            return StepResult(number, title, STATUS_SKIPPED, "транскрипт уже загружен")
        return StepResult(
            number,
            title,
            STATUS_DONE,
            f"записей={result.n_turns}, реплик={result.n_utterances}",
        )

    def _step_normalize(self, number: int, title: str) -> StepResult:
        """Нормализовать реплики в L2 и собрать слова для оператора."""
        from qa.pipeline.normalize import normalize_interview
        from qa.pipeline.terms_help import terms_path
        from qa.version import code_version

        result = normalize_interview(
            self._conn,
            self._code,
            self._settings,
            _provider(self._settings),
            version=1,
            code_version=code_version(),
        )
        self._open_forms = result.open_forms
        self._terms_path = str(terms_path(self._settings))
        if result.skipped:
            return StepResult(number, title, STATUS_SKIPPED, "L2 уже записан")
        stats = result.stats
        return StepResult(
            number,
            title,
            STATUS_DONE,
            f"единиц={stats.total}, вызовов={stats.model_calls}, слов для оператора: {result.open_forms}",
        )

    def _step_structure(self, number: int, title: str) -> StepResult:
        """Привязать реплики к вопросам и разбить L2 на фрагменты."""
        from qa.pipeline.structure import structure_interview
        from qa.version import code_version

        result = structure_interview(
            self._conn,
            self._code,
            self._settings,
            _provider(self._settings),
            code_version=code_version(),
        )
        if result.skipped:
            return StepResult(number, title, STATUS_SKIPPED, "фрагменты уже есть")
        return StepResult(
            number,
            title,
            STATUS_DONE,
            f"блоков={result.stats.blocks}, фрагментов={result.stats.fragments}",
        )

    def _step_code(self, number: int, title: str) -> StepResult:
        """Закодировать фрагменты кодами кодбука."""
        from qa.pipeline.coding import code_interview
        from qa.version import code_version

        result = code_interview(
            self._conn,
            self._code,
            self._settings,
            _provider(self._settings),
            code_version=code_version(),
        )
        stats = result.stats
        return StepResult(
            number,
            title,
            STATUS_DONE,
            f"фрагментов={stats.fragments}, coded={stats.coded}, unmarked={stats.unmarked}, "
            f"аннотаций={stats.annotations}",
        )

    def _step_dashboard(self, number: int, title: str) -> StepResult:
        """Собрать HTML-дашборд покрытия."""
        from datetime import date

        from qa.dashboard.render import render_dashboard
        from qa.pipeline.aggregate import build_coverage
        from qa.pipeline.quote_view import quote_editor_from_settings

        coverage = build_coverage(
            self._conn,
            self._code,
            self._settings.groups,
            low_min=self._settings.m4.theme_low_min_annotations,
            high_min=self._settings.m4.theme_high_min_annotations,
            quotes_per_theme=self._settings.m4.quotes_per_theme,
            quote_max_chars=self._settings.m4.quote_max_chars,
            quote_editor=quote_editor_from_settings(self._settings),
            quote_min_chars=self._settings.m8.quote_min_chars,
        )
        html = render_dashboard(coverage, date.today().isoformat())
        output = Path(self._settings.paths.exports) / f"dashboard_{self._code}.html"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html, encoding="utf-8")
        return StepResult(number, title, STATUS_DONE, f"аннотаций={coverage.annotations_total} -> {output.name}")

    def _step_report(self, number: int, title: str) -> StepResult:
        """Собрать HTML-отчёт."""
        from datetime import date

        from qa.db.analytics_repo import AnalyticsRepository
        from qa.pipeline.aggregate import build_coverage
        from qa.pipeline.quote_view import quote_editor_from_settings
        from qa.report.limitations import pipeline_limitations
        from qa.report.model import build_report
        from qa.report.render import render_report

        coverage = build_coverage(
            self._conn,
            self._code,
            self._settings.groups,
            low_min=self._settings.m4.theme_low_min_annotations,
            high_min=self._settings.m4.theme_high_min_annotations,
            quotes_per_theme=self._settings.m4.quotes_per_theme,
            quote_max_chars=self._settings.m4.quote_max_chars,
            quote_editor=quote_editor_from_settings(self._settings),
            quote_min_chars=self._settings.m8.quote_min_chars,
        )
        snapshot = date.today().isoformat()
        report = build_report(
            coverage,
            AnalyticsRepository(self._conn),
            snapshot_date=snapshot,
            settings=self._settings,
            extra_limitations=pipeline_limitations(self._conn, self._code, self._settings),
            quote_editor=quote_editor_from_settings(self._settings),
        )
        html = render_report(report)
        output = Path(self._settings.paths.reports) / f"report_{self._code}.html"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html, encoding="utf-8")
        return StepResult(
            number,
            title,
            STATUS_DONE,
            f"аннотаций={coverage.annotations_total}, ограничений={len(report.limitations)} -> {output.name}",
        )


def format_run_result(result: RunResult) -> list[str]:
    """Строки отчёта команды `run` для оператора."""
    lines: list[str] = []
    for step in result.steps:
        mark = {
            STATUS_DONE: "выполнено",
            STATUS_SKIPPED: "пропущено",
            STATUS_GATE: "ворота",
            STATUS_BLOCKED: "остановка",
        }.get(step.status, step.status)
        lines.append(f"{step.number}. {step.title}: {mark}" + (f" — {step.detail}" if step.detail else ""))
        for extra in step.lines:
            lines.append(f"   {extra}")
    if result.all_done:
        lines.append("Всё сделано: следующих шагов нет.")
    elif result.stopped_at_gate is not None:
        lines.append(
            "Остановка перед сегментацией: заполните лист «Нужна помощь» в таблице "
            "терминов и выполните apply-terms, затем повторите run. "
            "Флаг --no-wait проходит дальше, применив правила пустой ячейки."
        )
    return lines


def warn_names_map() -> str | None:
    """Предупреждение о пустой таблице имён в момент выгрузки для коллег.

    `NAMES_MAP_PATH` — данные, а не код: пока он пуст, псевдонимизация при
    загрузке фактически не работает, и в тексте остаются названия организаций
    (SPEC-M8, решение 5).
    """
    value = (os.environ.get("NAMES_MAP_PATH") or "").strip()
    if value:
        return None
    return (
        "NAMES_MAP_PATH не задан: в тексте реплик остаются имена и названия организаций. "
        "Перед отправкой таблицы коллегам заполните таблицу имён (CONSTITUTION §5)"
    )


__all__ = [
    "PipelineRunner",
    "RunError",
    "RunResult",
    "StepResult",
    "format_run_result",
    "warn_names_map",
]
