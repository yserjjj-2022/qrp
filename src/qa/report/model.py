"""Модель данных отчёта (M5).

Dataclasses без HTML и без SQL: рендер отчёта принимает эту модель и не
считает чисел сам (SPEC-M6, «Разделение слоёв»). Модуль не импортирует
`qa.report.render`.
"""

from __future__ import annotations

from dataclasses import dataclass

from qa.db.analytics_repo import AnalyticsRepository
from qa.pipeline.aggregate import HEURISTIC_CAPTION, CoverageReport

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
class InterviewReport:
    """Полный отчёт: происхождение, покрытие, ограничения."""

    provenance: Provenance
    coverage: CoverageReport
    limitations: tuple[Limitation, ...]

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
) -> InterviewReport:
    """Собрать модель отчёта по готовому покрытию.

    Args:
        coverage: покрытие интервью, уже посчитанное агрегацией.
        repo: репозиторий для списка прогонов (единственный источник SQL).
        snapshot_date: дата снимка чисел (ISO).

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
    return InterviewReport(
        provenance=provenance,
        coverage=coverage,
        limitations=_limitations(coverage),
    )
