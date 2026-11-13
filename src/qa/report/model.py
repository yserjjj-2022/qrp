"""Модель данных отчёта (M5, расширена в M8).

Dataclasses без HTML и без SQL: рендер отчёта принимает эту модель и не
считает чисел сам (SPEC-M6, «Разделение слоёв»). Модуль не импортирует
`qa.report.render`.

Модель готова к агрегации нескольких интервью (SPEC-M8, D.3): происхождение
хранится списком `interviews` (сейчас из одного элемента), а темы адресуются
ключом `(узел, тема)` через `theme_index`. Одиночное поле `provenance`
оставлено как свойство: отчёт по одному интервью читается как раньше.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qa.db.analytics_repo import AnalyticsRepository
from qa.pipeline.aggregate import HEURISTIC_CAPTION, CoverageReport, ThemeCoverage
from qa.pipeline.expectations import Expectations, build_expectations
from qa.pipeline.quote_view import QuoteEditor
from qa.settings import Settings

# Тип расшифровки фиксирован CONSTITUTION §6: в проекте один источник
# транскриптов — машинная расшифровка с лёгкой редакцией.
TRANSCRIPT_TYPE = "машинная расшифровка с лёгкой редакцией"


@dataclass(frozen=True)
class RunInfo:
    """Прогон, который затрагивал интервью.

    Стадии в таблице `run` нет: она восстанавливается из `params_json`
    (кодирование — ключ `fragments`, структура — `blocks`), поэтому поле
    приходит уже вычисленным из репозитория.
    """

    id: int
    stage: str
    model: str | None
    prompt_hash: str | None
    code_version: str | None
    started_at: str | None


@dataclass(frozen=True)
class Provenance:
    """Происхождение чисел: кто, когда и на какой версии их получил."""

    interview_code: str
    group: str
    transcript_type: str
    l2_version: int | None
    codebook_version_id: int | None
    snapshot_date: str
    snapshot_run_id: int | None
    runs: tuple[RunInfo, ...]


@dataclass(frozen=True)
class Limitation:
    """Ограничение отчёта: машинный ключ и формулировка для читателя."""

    code: str
    text: str


@dataclass(frozen=True)
class QuotePolicy:
    """Что произошло с цитатами: обрезка, правка и по какому правилу.

    Строка о правках в начале блока цитат собирается из этих чисел, а не
    пишется вручную: если правка выключена, отчёт не утверждает, что она была.
    """

    truncated: int = 0
    edited: int = 0
    rules: tuple[str, ...] = ()

    @property
    def note(self) -> str:
        """Строка для читателя: что именно сделано с текстом цитат."""
        if not self.truncated and not self.edited:
            return (
                "Цитаты приведены дословно: вырезки не было, правка модели не применялась. "
                "Под каждой цитатой — оригинал с тайм-кодом."
            )
        parts = ["Цитаты частично отредактированы"]
        if self.truncated:
            parts.append(f"обрезка отмечена […]: {self.truncated}")
        if self.edited:
            rules = ", ".join(self.rules) if self.rules else "без названия правила"
            parts.append(f"правка модели: {self.edited} (правило: {rules})")
        return "; ".join(parts) + ". Оригинал приведён под каждой цитатой."


@dataclass(frozen=True)
class InterviewReport:
    """Полный отчёт: происхождение, покрытие, ожидания, ограничения.

    `interviews` — список интервью отчёта. Сейчас он всегда из одного
    элемента, но поля `provenance` больше нет: при агрегации нескольких
    интервью отчёт не придётся переделывать (SPEC-M8, D.3).
    """

    interviews: tuple[Provenance, ...]
    coverage: CoverageReport
    limitations: tuple[Limitation, ...]
    expectations: Expectations = field(default_factory=lambda: Expectations(sections=(), overlap=0))
    quote_policy: QuotePolicy = QuotePolicy()

    @property
    def provenance(self) -> Provenance:
        """Происхождение единственного интервью отчёта."""
        return self.interviews[0]

    @property
    def theme_index(self) -> dict[tuple[str, str], ThemeCoverage]:
        """Темы по ключу `(узел, тема)` — ключ будущей агрегации интервью."""
        return {(node.node, theme.theme): theme for node in self.coverage.nodes for theme in node.themes}

    @property
    def annotations_total(self) -> int:
        """Все аннотации интервью, кроме отклонённых."""
        return self.coverage.annotations_total

    @property
    def annotations_verified(self) -> int:
        """Аннотации, проверенные вручную (`accepted`/`edited`)."""
        return self.coverage.annotations_verified


def _limitations(coverage: CoverageReport) -> tuple[Limitation, ...]:
    """Ограничения отчёта: детерминированные формулировки из чисел модели.

    Текст не пишет модель: отчёт на шаблоне не может соврать о числе, потому
    что каждое число подставляется из `CoverageReport`.
    """
    quality = coverage.quality
    doubtful_share = (
        f"{quality.accepted_on_doubtful} из {coverage.annotations_total} аннотаций и "
        f"{quality.fragments_with_accepted_doubtful} из {quality.doubtful} doubtful-фрагментов"
    )
    return (
        Limitation(
            code="snapshot",
            text=(
                "Числа привязаны к дате и прогону: снимок на дату отчёта, "
                f"run {coverage.run_id if coverage.run_id is not None else '—'}. "
                "После повторного кодирования или ревью они изменятся."
            ),
        ),
        Limitation(
            code="model_marking",
            text=(
                f"Разметка модели; {coverage.annotations_verified} из {coverage.annotations_total} "
                "аннотаций проверены вручную, остальные не проверялись."
            ),
        ),
        Limitation(
            code="skipped_garbled",
            text=(
                f"Искажённых фрагментов, пропущенных при кодировании: {quality.skipped_garbled}. "
                "Такой фрагмент не кодируется и не считается случаем «код не найден»."
            ),
        ),
        Limitation(
            code="interpretation_only",
            text=(
                f"Аннотаций, которые держатся только на прочтении модели: {quality.interpretation_only}. "
                "Такие места требуют ручной проверки."
            ),
        ),
        Limitation(
            code="reliability_cap",
            text=(
                f"Фрагментов с сомнительной расшифровкой: {quality.doubtful}. "
                "Для них надёжность кода ограничена сверху правилом кодирования."
            ),
        ),
        Limitation(
            code="doubtful_share",
            text=f"Доля выводов на сомнительном тексте: {doubtful_share}.",
        ),
        Limitation(
            code="unmarked",
            text=(
                f"Фрагментов без разметки: {quality.unmarked} из {coverage.fragments_total}. "
                "Их отсутствие в покрытии — пропуск кодирования, а не отсутствие темы."
            ),
        ),
        Limitation(
            code="oth",
            text=(
                f"Аннотаций с кодом OTH: {quality.oth}. Кластеризация отложена (M3 отложен), "
                "поэтому новые темы по OTH не предлагаются."
            ),
        ),
        Limitation(
            code="codebook_coverage",
            text=(
                f"Использовано {coverage.codebook.used} из {coverage.codebook.total} кодов кодбука. "
                "Неиспользованные коды перечислены одним списком по семьям: код не найден, "
                "если по нему нет ни одной аннотации."
            ),
        ),
        Limitation(
            code="one_interview",
            text=(
                f"Одно интервью, группа «{coverage.group or '—'}». Пустые темы означают отсутствие "
                "данных в этом разговоре, а не дефект кодирования; выводов о группе отчёт не делает."
            ),
        ),
        Limitation(
            code="reference_fact",
            text="Эталонных фактов нет, поэтому оценка понимания не делается.",
        ),
        Limitation(
            code="leaf_status",
            text=(
                f"Статусы тем — {HEURISTIC_CAPTION}: шкала показывает, есть ли чем показать тему, а не раскрыта ли она."
            ),
        ),
    )


def build_report(
    coverage: CoverageReport,
    repo: AnalyticsRepository,
    *,
    snapshot_date: str,
    settings: Settings | None = None,
    extra_limitations: tuple[Limitation, ...] = (),
    quote_editor: QuoteEditor | None = None,
) -> InterviewReport:
    """Собрать модель отчёта по готовому покрытию.

    Args:
        coverage: покрытие интервью, уже посчитанное агрегацией.
        repo: репозиторий для списка прогонов (единственный источник SQL).
        snapshot_date: дата снимка чисел (ISO).
        settings: настройки проекта; нужны для раздела «Ожидания и
            рекомендации» (`m8.autonomy_codes`, `m8.advice_questions`). Без них
            раздел пуст, а базовый отчёт M5 собирается как раньше.
        extra_limitations: ограничения, которые знает только вызывающая
            сторона (состояние конвейера: термины, выборка качества, правки).
        quote_editor: правка цитат моделью; None — без правки. Тот же редактор,
            что применён к цитатам тем: правило одно на весь отчёт.

    Returns:
        Модель отчёта для рендера.
    """
    runs = tuple(
        RunInfo(
            id=run.id,
            stage=run.stage,
            model=run.model,
            prompt_hash=run.prompt_hash,
            code_version=run.code_version,
            started_at=run.started_at,
        )
        for run in repo.interview_runs(coverage.interview_db_id)
    )
    provenance = Provenance(
        interview_code=coverage.interview_code,
        group=coverage.group,
        transcript_type=TRANSCRIPT_TYPE,
        l2_version=coverage.l2_version,
        codebook_version_id=coverage.codebook_version_id,
        snapshot_date=snapshot_date,
        snapshot_run_id=coverage.run_id,
        runs=runs,
    )
    expectations = (
        build_expectations(
            coverage,
            settings.m8,
            quote_max_chars=settings.m4.quote_max_chars,
            quote_min_chars=settings.m8.quote_min_chars,
            editor=quote_editor,
        )
        if settings is not None
        else Expectations(sections=(), overlap=0)
    )
    quotes = tuple(quote for node in coverage.nodes for theme in node.themes for quote in theme.quotes)
    policy = QuotePolicy(
        truncated=sum(1 for quote in quotes if quote.truncated),
        edited=sum(1 for quote in quotes if quote.edited),
        rules=tuple(sorted({quote.edit_rule for quote in quotes if quote.edited and quote.edit_rule})),
    )
    return InterviewReport(
        interviews=(provenance,),
        coverage=coverage,
        limitations=_limitations(coverage) + tuple(extra_limitations),
        expectations=expectations,
        quote_policy=policy,
    )
