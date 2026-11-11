"""CLI entry point: qa-pipeline."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from qa.db.connection import get_connection
from qa.db.migrations import migrate
from qa.log import setup_logging
from qa.settings import Settings, load_env_file


def _load_settings() -> Settings:
    """Загрузить .env (если есть) и настройки проекта."""
    load_env_file()
    return Settings.load()


def _migrations_dir() -> Path:
    """Каталог миграций: из корня репозитория, иначе из текущего каталога."""
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "migrations"
    if candidate.is_dir():
        return candidate
    return Path.cwd() / "migrations"


def _open_db(settings: Settings) -> sqlite3.Connection:
    """Открыть БД из настроек и применить миграции."""
    settings.ensure_dirs()
    db_path = Path(settings.db_path)
    backup_dir = Path(settings.paths.backups)
    conn = get_connection(db_path)
    migrate(conn, db_path, _migrations_dir(), backup_dir)
    return conn


def _workbook_path(args: argparse.Namespace, settings: Settings) -> Path:
    """Путь к книге: аргумент команды или paths.workbook."""
    explicit = getattr(args, "workbook", None)
    return Path(explicit) if explicit else Path(settings.paths.workbook)


def _print_tree(tree: list[tuple[str, list[tuple[str, list[str]]]]]) -> None:
    """Показать дерево тем оператору."""
    print("Дерево тем:")
    for node, themes in tree:
        print(f"  {node}")
        for theme, codes in themes:
            print(f"    {theme}: {', '.join(codes) if codes else '—'}")


def _print_notes(notes: list[str]) -> None:
    """Показать предупреждения и заметки разбора."""
    for note in notes:
        print(f"  ! {note}")


def cmd_load_codebook(args: argparse.Namespace) -> int:
    """Загрузить обязательные листы книги, темы и словарь замен."""
    settings = _load_settings()
    run_id = setup_logging(settings.log_level)
    from qa.pipeline.load_codebook import load_codebook

    conn = _open_db(settings)
    try:
        result = load_codebook(
            conn,
            _workbook_path(args, settings),
            settings.workbook,
            reload=args.reload,
            groups=settings.groups,
        )
    finally:
        conn.close()
    if not result.created:
        print(f"[{run_id}] книга уже загружена, дублей нет")
    counts = result.counts
    print(
        f"[{run_id}] кодбук: кодов={counts.get('codes', 0)}, вопросов={counts.get('questions', 0)}, "
        f"связей вопрос-код={counts.get('question_code', 0)}, словарь={counts.get('glossary', 0)}, "
        f"атрибутов={counts.get('attributes', 0)}"
    )
    _print_tree(result.tree)
    _print_notes(result.notes)
    return 0


def cmd_load_project(args: argparse.Namespace) -> int:
    """Загрузить необязательные листы проекта и умолчания."""
    settings = _load_settings()
    run_id = setup_logging(settings.log_level)
    from qa.pipeline.load_project import load_project

    conn = _open_db(settings)
    try:
        result = load_project(
            conn,
            _workbook_path(args, settings),
            settings.workbook,
            groups=settings.groups,
            lexicon_status_map=settings.lexicon_status_map,
            pipeline=settings.pipeline,
            transcripts_dir=Path(settings.paths.transcripts),
        )
    finally:
        conn.close()
    print(f"[{run_id}] проект: {result.counts}")
    if result.missing_sheets:
        print(f"  листы отсутствуют, применены умолчания: {', '.join(result.missing_sheets)}")
    _print_notes(result.discrepancies)
    return 0


def cmd_init_interview(args: argparse.Namespace) -> int:
    """Определить говорящих, показать роли, подтвердить и записать интервью."""
    settings = _load_settings()
    run_id = setup_logging(settings.log_level)
    from qa.pipeline.init_interview import apply_plan, build_plan, create_names_map_skeleton, render_plan

    conn = _open_db(settings)
    try:
        plan = build_plan(conn, Path(args.transcript), args.group, settings.groups)
        print(render_plan(plan))
        if plan.needs_confirmation:
            answer = input("Записать роли в БД? [y/N]: ").strip().lower()
            if answer not in ("y", "н", "да"):
                print("Отменено оператором, в БД ничего не записано")
                return 1
        interview_db_id = apply_plan(conn, plan, settings.groups)
        names_map_path = create_names_map_skeleton(plan)
    finally:
        conn.close()
    print(f"[{run_id}] интервью {plan.interview_id} записано (id={interview_db_id})")
    if names_map_path is not None:
        print(f"  заготовка таблицы имён: {names_map_path}")
    else:
        print("  NAMES_MAP_PATH не задан: заготовка таблицы имён не создавалась")
    return 0


def cmd_load_transcript(args: argparse.Namespace) -> int:
    """Загрузить транскрипт интервью (M1a)."""
    settings = _load_settings()
    run_id = setup_logging(settings.log_level)
    from qa.pipeline.init_interview import build_plan, plan_to_meta, resolve_names_map
    from qa.pipeline.load_transcript import build_replacement_map, load_meta, load_transcript

    transcript_path = Path(args.transcript)
    conn = _open_db(settings)
    try:
        if args.meta:
            meta = load_meta(Path(args.meta))
        else:
            plan = build_plan(conn, transcript_path, args.group, settings.groups)
            if args.interview_id and args.interview_id != plan.interview_id:
                print(
                    f"Интервью {args.interview_id} не совпадает с кодом из имени файла ({plan.interview_id}). "
                    "Сначала выполните init-interview с этим файлом"
                )
                return 1
            meta = plan_to_meta(plan, resolve_names_map())
        replacement_map = build_replacement_map(meta, resolve_names_map())
        result = load_transcript(
            conn, transcript_path, meta, replacement_map, settings.pipeline, groups=settings.groups
        )
    finally:
        conn.close()
    if result.skipped:
        print(f"[{run_id}] интервью {meta.interview_id} уже загружено, дублей нет")
    else:
        print(
            f"[{run_id}] интервью {meta.interview_id}: записей={result.n_turns}, "
            f"реплик={result.n_utterances}, pii_flag={result.n_pii_flagged}"
        )
    return 0


def cmd_review_terms(args: argparse.Namespace) -> int:
    """Интерактивно утвердить или отклонить кандидатов new_term."""
    settings = _load_settings()
    run_id = setup_logging(settings.log_level)
    from qa.pipeline.review_terms import review_terms

    conn = _open_db(settings)
    try:
        result = review_terms(conn)
    finally:
        conn.close()
    print(
        f"[{run_id}] review-terms: утверждено={len(result.approved)}, "
        f"вариантов={len(result.variants)}, отклонено={len(result.rejected)}, пропущено={len(result.skipped)}, "
        f"вне актуального L2={result.stale}"
    )
    for surface_form, canonical in sorted(result.variants.items()):
        print(f"  вариант: {surface_form!r} -> {canonical}")
    return 0


def cmd_export_golden_sample(args: argparse.Namespace) -> int:
    """Выгрузить реплики L1 в xlsx для ручной нормализации."""
    settings = _load_settings()
    run_id = setup_logging(settings.log_level)
    from qa.pipeline.golden import GOLDEN_FILENAME, export_golden_sample

    output = Path(args.output) if args.output else Path(settings.paths.golden) / GOLDEN_FILENAME
    conn = _open_db(settings)
    try:
        result = export_golden_sample(conn, output, n=args.n, interview_id=args.interview_id)
    finally:
        conn.close()
    print(f"[{run_id}] золотой набор: {result.n_rows} строк -> {result.path}")
    return 0


def cmd_import_golden(args: argparse.Namespace) -> int:
    """Загрузить заполненный золотой набор (файл или лист книги)."""
    settings = _load_settings()
    run_id = setup_logging(settings.log_level)
    from qa.pipeline.golden import GOLDEN_FILENAME, import_golden

    path = Path(args.file) if args.file else Path(settings.paths.golden) / GOLDEN_FILENAME
    conn = _open_db(settings)
    try:
        result = import_golden(conn, path)
    finally:
        conn.close()
    print(f"[{run_id}] золотой набор импортирован: строк={result.n_rows}, с L2={result.n_filled}")
    return 0


def cmd_llm_ping(args: argparse.Namespace) -> int:
    """Проверить эндпоинт и модель на синтетическом запросе."""
    from pydantic import ValidationError

    from qa.llm.errors import LlmError
    from qa.llm.ping import dry_run_llm, ping_llm

    try:
        settings = _load_settings()
    except ValidationError as exc:
        print(f"llm-ping: ошибка настроек — {_settings_error_summary(exc)}")
        return 1
    run_id = setup_logging(settings.log_level)
    try:
        outcome = (
            dry_run_llm(settings, normalize=args.normalize)
            if args.dry_run
            else ping_llm(settings, normalize=args.normalize)
        )
    except LlmError as exc:
        print(f"[{run_id}] llm-ping: ошибка — {exc}")
        return 1
    if args.dry_run:
        print(
            f"[{run_id}] llm-ping dry-run: провайдер={settings.llm.provider}, "
            f"модель={outcome.model_requested or '—'}, вызовов=1, "
            f"оценка токенов промпта={outcome.prompt_tokens}, отправлено=0"
        )
        return 0
    print(
        f"[{run_id}] llm-ping: model_used={outcome.model_used}, "
        f"prompt_tokens={outcome.prompt_tokens}, completion_tokens={outcome.completion_tokens}, "
        f"время={outcome.elapsed_ms:.0f} мс, кэш={'да' if outcome.cached else 'нет'}"
    )
    if not outcome.ok:
        print(f"[{run_id}] llm-ping: ответ модели не содержит ok=true")
        return 1
    return 0


def cmd_normalize(args: argparse.Namespace) -> int:
    """Нормализовать реплики интервью в слой L2."""
    from qa.llm.errors import LlmConfigError, LlmError
    from qa.pipeline.normalize import NormalizeError, normalize_interview, plan_dry_run
    from qa.version import code_version

    settings = _load_settings()
    run_id = setup_logging(settings.log_level)
    conn = _open_db(settings)
    try:
        if args.dry_run:
            report = plan_dry_run(conn, args.interview, settings)
            print(
                f"[{run_id}] normalize dry-run {report.interview_code}: реплик={report.utterances}, "
                f"вызовов модели={report.model_calls}, оценка токенов промпта={report.prompt_tokens}, "
                f"отправлено=0"
            )
            if report.already_done:
                print("  L2 версии 1 уже записан: без --new-version команда ничего не сделает")
            return 0

        provider = None
        if settings.llm.provider != "stub":
            from qa.llm import provider_from_settings

            try:
                provider = provider_from_settings(settings, normalize=True)
            except LlmConfigError as exc:
                print(f"[{run_id}] normalize: ошибка настроек — {exc}")
                return 1

        version = 1
        if args.new_version:
            from qa.db.workbook_repo import InterviewRepository, TextUnitRepository

            interview_db_id = InterviewRepository(conn).interview_db_id_by_code(args.interview)
            if interview_db_id is not None:
                version = TextUnitRepository(conn).max_version_for_interview(interview_db_id) + 1

        result = normalize_interview(
            conn,
            args.interview,
            settings,
            provider,
            version=version,
            code_version=code_version(),
        )
    except (NormalizeError, LlmError) as exc:
        print(f"[{run_id}] normalize: ошибка — {exc}")
        return 1
    finally:
        conn.close()

    if result.skipped:
        print(f"[{run_id}] normalize {result.interview_code}: L2 версии {result.version} уже записан, дублей нет")
        return 0

    stats = result.stats
    quality = ", ".join(f"{name}={value}" for name, value in sorted(stats.asr_quality.items())) or "—"
    rejections = ", ".join(f"{name}={value}" for name, value in sorted(stats.rejections.items())) or "нет"
    terms = ", ".join(f"{name}={value}" for name, value in sorted(stats.term_replacements.items())) or "нет"
    print(
        f"[{run_id}] normalize {result.interview_code} (L2 v{result.version}): реплик={stats.total}, "
        f"proposed={stats.proposed}, rejected={stats.rejected}, вызовов={stats.model_calls}, "
        f"модель={result.model_used or '—'}"
    )
    print(f"  asr_quality: {quality}")
    print(
        f"  паразитов удалено={stats.fillers_removed}, заиканий={stats.stutter_removed}, "
        f"неясных повторов={stats.repeat_unclear}"
    )
    print(f"  замены по лексикону: {terms}")
    print(
        f"  fuzzy-замен={stats.fuzzy_replacements}, склеек={stats.glued_replacements}, "
        f"asr_fix={stats.asr_fixes} (молчаливых={stats.implicit_asr_fixes}), "
        f"мягкая потеря={stats.soft_thought_loss}, пометок названий={stats.names_review}"
    )
    print(f"  отказы автопроверок: {rejections}")
    print(f"  new_term={stats.new_terms}, токены: prompt={stats.prompt_tokens}, completion={stats.completion_tokens}")

    if args.review_export:
        _export_review(settings, result.interview_code, result.version, run_id)
    return 0


def _export_review(settings: Settings, interview_code: str, version: int, run_id: str) -> None:
    """Выгрузить файл ручной проверки L2 в data/exports."""
    from qa.pipeline.review_export import export_manual_review

    output = Path(settings.paths.exports) / f"normalize_review_{interview_code}_v{version}.xlsx"
    conn = _open_db(settings)
    try:
        review = export_manual_review(conn, interview_code, version, output)
    except ValueError as exc:
        print(f"[{run_id}] normalize: ручная проверка не выгружена — {exc}")
        return
    finally:
        conn.close()
    print(
        f"[{run_id}] ручная проверка: {review.n_rows} строк "
        f"(флаги={review.flagged}, случайных={review.sampled}) -> {review.path}"
    )


def cmd_structure(args: argparse.Namespace) -> int:
    """Привязать реплики к вопросам гайда и разбить L2 на фрагменты (M1c)."""
    from qa.llm.errors import LlmConfigError, LlmError
    from qa.pipeline.structure import StructureError, format_report, structure_interview
    from qa.version import code_version

    settings = _load_settings()
    run_id = setup_logging(settings.log_level)

    provider = None
    if not args.no_model and settings.llm.provider != "stub":
        from qa.llm import provider_from_settings

        try:
            provider = provider_from_settings(settings)
        except LlmConfigError as exc:
            print(f"[{run_id}] structure: ошибка настроек — {exc}")
            return 1

    conn = _open_db(settings)
    try:
        result = structure_interview(
            conn,
            args.interview,
            settings,
            provider,
            l2_version=args.l2_version,
            code_version=code_version(),
            rebuild=args.rebuild,
        )
    except (StructureError, LlmError) as exc:
        print(f"[{run_id}] structure: ошибка — {exc}")
        return 1
    finally:
        conn.close()

    if result.skipped:
        print(
            f"[{run_id}] structure {result.interview_code}: фрагменты уже есть, повторная запись не нужна (--rebuild)"
        )
        return 0
    for index, line in enumerate(format_report(result)):
        print(f"[{run_id}] {line}" if index == 0 else line)
    return 0


def cmd_code(args: argparse.Namespace) -> int:
    """Закодировать фрагменты интервью кодами кодбука (M2)."""
    from qa.llm.errors import LlmConfigError, LlmError
    from qa.pipeline.coding import CodingError, code_interview, format_report, plan_dry_run
    from qa.version import code_version

    settings = _load_settings()
    run_id = setup_logging(settings.log_level)
    conn = _open_db(settings)
    try:
        if args.dry_run:
            report = plan_dry_run(conn, args.interview, settings)
            print(
                f"[{run_id}] code dry-run {report.interview_code}: реплик={report.utterances}, "
                f"фрагментов={report.fragments}, garbled={report.garbled}, doubtful={report.doubtful}, "
                f"вызовов модели={report.model_calls}, оценка токенов промпта={report.prompt_tokens}, "
                f"отправлено=0"
            )
            return 0

        provider = None
        if settings.llm.provider != "stub":
            from qa.llm import provider_from_settings

            try:
                provider = provider_from_settings(settings)
            except LlmConfigError as exc:
                print(f"[{run_id}] code: ошибка настроек — {exc}")
                return 1

        result = code_interview(
            conn,
            args.interview,
            settings,
            provider,
            code_version=code_version(),
        )
    except (CodingError, LlmError) as exc:
        print(f"[{run_id}] code: ошибка — {exc}")
        return 1
    finally:
        conn.close()

    for index, line in enumerate(format_report(result)):
        print(f"[{run_id}] {line}" if index == 0 else line)
    if args.review_export:
        _export_coding_review(settings, result, run_id)
    return 0


def _export_coding_review(settings: Settings, result: object, run_id: str) -> None:
    """Выгрузить очередь ручной проверки кодирования в data/exports."""
    from qa.pipeline.coding_review import export_coding_review

    interview_code = getattr(result, "interview_code", "")
    run_db_id = getattr(result, "run_id", None)
    if not isinstance(run_db_id, int):
        print(f"[{run_id}] code: ручная проверка не выгружена — прогон не записан")
        return
    output = Path(settings.paths.exports) / f"coding_review_{interview_code}_run{run_db_id}.xlsx"
    conn = _open_db(settings)
    try:
        review = export_coding_review(
            conn,
            interview_code,
            run_db_id,
            output,
            sample_share=settings.m2.review_sample_share,
        )
    except ValueError as exc:
        print(f"[{run_id}] code: ручная проверка не выгружена — {exc}")
        return
    finally:
        conn.close()
    print(f"[{run_id}] ручная проверка: {review.n_rows} строк -> {review.path}")


def cmd_import_coding_review(args: argparse.Namespace) -> int:
    """Записать решения ручной проверки кодирования и посчитать метрику."""
    settings = _load_settings()
    run_id = setup_logging(settings.log_level)
    from qa.pipeline.coding_review import import_coding_review

    path = Path(args.file)
    conn = _open_db(settings)
    try:
        result = import_coding_review(conn, path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[{run_id}] import-coding-review: ошибка — {exc}")
        return 1
    finally:
        conn.close()
    share = "—" if result.approved_share is None else f"{result.approved_share:.3f}"
    print(
        f"[{run_id}] решения ручной проверки: принято={result.approved}, исправлено={result.edited}, "
        f"отклонено={result.rejected}, без решения={result.skipped}, доля принятых без правок={share}"
    )
    return 0


def _settings_error_summary(exc: Exception) -> str:
    """Краткий текст ошибки настроек без значений полей."""
    from pydantic import ValidationError

    if isinstance(exc, ValidationError):
        messages = [str(error.get("msg", "")) for error in exc.errors()]
        return "; ".join(message for message in messages if message) or "некорректные настройки"
    return "некорректные настройки"


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Собрать HTML-дашборд покрытия дерева тем (M4)."""
    from datetime import date

    from qa.dashboard.render import render_dashboard
    from qa.pipeline.aggregate import AggregateError, build_coverage

    settings = _load_settings()
    run_id = setup_logging(settings.log_level)
    conn = _open_db(settings)
    try:
        report = build_coverage(
            conn,
            args.interview,
            settings.groups,
            low_min=settings.m4.theme_low_min_annotations,
            high_min=settings.m4.theme_high_min_annotations,
            quotes_per_theme=settings.m4.quotes_per_theme,
            quote_max_chars=settings.m4.quote_max_chars,
        )
    except AggregateError as exc:
        print(f"[{run_id}] dashboard: ошибка — {exc}")
        return 1
    finally:
        conn.close()

    snapshot = args.snapshot_date or date.today().isoformat()
    html = render_dashboard(report, snapshot)
    default_name = f"dashboard_{report.interview_code}.html"
    output = Path(args.output) if args.output else Path(settings.paths.exports) / default_name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")

    accepted = report.quality.by_status.get("accepted", 0) + report.quality.by_status.get("edited", 0)
    print(
        f"[{run_id}] dashboard {report.interview_code}: тем={sum(len(node.themes) for node in report.nodes)}, "
        f"аннотаций={report.annotations_total} (проверено вручную {report.annotations_verified}), "
        f"принятых={accepted}, кодов={report.codebook.used} из {report.codebook.total}, "
        f"run={report.run_id if report.run_id is not None else '—'} -> {output}"
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Собрать HTML-отчёт по интервью (M5)."""
    from datetime import date

    from qa.db.analytics_repo import AnalyticsRepository
    from qa.pipeline.aggregate import AggregateError, build_coverage
    from qa.report.model import build_report
    from qa.report.render import render_report

    settings = _load_settings()
    run_id = setup_logging(settings.log_level)
    conn = _open_db(settings)
    try:
        coverage = build_coverage(
            conn,
            args.interview,
            settings.groups,
            low_min=settings.m4.theme_low_min_annotations,
            high_min=settings.m4.theme_high_min_annotations,
            quotes_per_theme=settings.m4.quotes_per_theme,
            quote_max_chars=settings.m4.quote_max_chars,
        )
        snapshot = args.snapshot_date or date.today().isoformat()
        report = build_report(coverage, AnalyticsRepository(conn), snapshot_date=snapshot)
    except AggregateError as exc:
        print(f"[{run_id}] report: ошибка — {exc}")
        return 1
    finally:
        conn.close()

    html = render_report(report)
    default_name = f"report_{report.provenance.interview_code}.html"
    output = Path(args.output) if args.output else Path(settings.paths.reports) / default_name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")

    print(
        f"[{run_id}] report {report.provenance.interview_code}: "
        f"аннотаций={report.annotations_total} (проверено вручную {report.annotations_verified}), "
        f"ограничений={len(report.limitations)}, прогонов={len(report.provenance.runs)} -> {output}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Собрать парсер команд."""
    parser = argparse.ArgumentParser(
        prog="qa-pipeline",
        description="Анализатор интервью: тематический анализ транскриптов",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_load = subparsers.add_parser("load-codebook", help="Загрузить кодбук, темы и словарь замен")
    p_load.add_argument("workbook", nargs="?", help="Путь к книге (.xlsx); по умолчанию paths.workbook")
    p_load.add_argument("--reload", action="store_true", help="Пересобрать темы, даже если книга уже загружена")
    p_load.set_defaults(func=cmd_load_codebook)

    p_project = subparsers.add_parser("load-project", help="Загрузить необязательные листы проекта")
    p_project.add_argument("workbook", nargs="?", help="Путь к книге (.xlsx); по умолчанию paths.workbook")
    p_project.set_defaults(func=cmd_load_project)

    p_init = subparsers.add_parser("init-interview", help="Определить роли и записать интервью")
    p_init.add_argument("transcript", help="Путь к транскрипту (.txt, UTF-8), имя файла = код респондента")
    p_init.add_argument("--group", help="Группа респондента; значения и алиасы заданы в config.yaml (groups)")
    p_init.set_defaults(func=cmd_init_interview)

    p_tr = subparsers.add_parser("load-transcript", help="Загрузить транскрипт интервью (M1a)")
    p_tr.add_argument("transcript", help="Путь к транскрипту (.txt, UTF-8)")
    p_tr.add_argument("--interview-id", help="id_интервью из БД или листа «Интервью»")
    p_tr.add_argument("--group", help="Группа респондента, если интервью ещё нет в БД (значения из config.yaml)")
    p_tr.add_argument("--meta", help="Устаревший interview.yaml (совместимость)")
    p_tr.set_defaults(func=cmd_load_transcript)

    p_review = subparsers.add_parser("review-terms", help="Ревью кандидатов новых терминов (M1b)")
    p_review.set_defaults(func=cmd_review_terms)

    p_export = subparsers.add_parser("export-golden-sample", help="Выгрузить реплики L1 для золотого набора")
    p_export.add_argument("--n", type=int, default=40, help="Сколько реплик выбрать (default: 40)")
    p_export.add_argument("--interview-id", help="Только это интервью")
    p_export.add_argument("--output", help="Путь к xlsx; по умолчанию data/golden/golden_sample.xlsx")
    p_export.set_defaults(func=cmd_export_golden_sample)

    p_import = subparsers.add_parser("import-golden", help="Загрузить заполненный золотой набор")
    p_import.add_argument("file", nargs="?", help="Путь к xlsx; по умолчанию data/golden/golden_sample.xlsx")
    p_import.set_defaults(func=cmd_import_golden)

    p_ping = subparsers.add_parser("llm-ping", help="Проверить LLM-эндпоинт на синтетическом запросе")
    p_ping.add_argument("--normalize", action="store_true", help="Использовать LLM_MODEL_NORMALIZE")
    p_ping.add_argument("--dry-run", action="store_true", help="Проверить без отправки запроса")
    p_ping.set_defaults(func=cmd_llm_ping)

    p_norm = subparsers.add_parser("normalize", help="Нормализовать реплики интервью в слой L2")
    p_norm.add_argument("interview", help="Код интервью (R01), уже загруженного в БД")
    p_norm.add_argument("--dry-run", action="store_true", help="Посчитать объём работы без вызовов модели")
    p_norm.add_argument("--new-version", action="store_true", help="Создать новую версию L2")
    p_norm.add_argument("--review-export", action="store_true", help="Выгрузить файл ручной проверки L2")
    p_norm.set_defaults(func=cmd_normalize)

    p_struct = subparsers.add_parser("structure", help="Привязать реплики к вопросам гайда и разбить L2 (M1c)")
    p_struct.add_argument("interview", help="Код интервью (R01), уже загруженного в БД")
    p_struct.add_argument("--l2-version", type=int, help="Версия L2; по умолчанию последняя записанная")
    p_struct.add_argument("--rebuild", action="store_true", help="Пересобрать привязку и фрагменты")
    p_struct.add_argument("--no-model", action="store_true", help="Без вызовов модели: только правила кода")
    p_struct.set_defaults(func=cmd_structure)

    p_code = subparsers.add_parser("code", help="Закодировать фрагменты интервью кодами кодбука (M2)")
    p_code.add_argument("interview", help="Код интервью (R01), уже загруженного в БД")
    p_code.add_argument("--dry-run", action="store_true", help="Посчитать объём работы без вызовов модели")
    p_code.add_argument("--review-export", action="store_true", help="Выгрузить очередь ручной проверки")
    p_code.set_defaults(func=cmd_code)

    p_coding_review = subparsers.add_parser(
        "import-coding-review",
        help="Записать решения ручной проверки кодирования и посчитать долю принятых без правок",
    )
    p_coding_review.add_argument("file", help="Заполненный xlsx очереди ручной проверки кодирования")
    p_coding_review.set_defaults(func=cmd_import_coding_review)

    p_dash = subparsers.add_parser("dashboard", help="HTML-дашборд покрытия дерева тем (M4)")
    p_dash.add_argument("interview", help="Код интервью (R01), уже загруженного в БД")
    p_dash.add_argument("--output", help="Путь к HTML; по умолчанию data/exports/dashboard_<код>.html")
    p_dash.add_argument("--snapshot-date", help="Дата снимка чисел (ISO); по умолчанию сегодня")
    p_dash.set_defaults(func=cmd_dashboard)

    p_report = subparsers.add_parser("report", help="HTML-отчёт по интервью (M5)")
    p_report.add_argument("interview", help="Код интервью (R01), уже загруженного в БД")
    p_report.add_argument("--output", help="Путь к HTML; по умолчанию data/reports/report_<код>.html")
    p_report.add_argument("--snapshot-date", help="Дата снимка чисел (ISO); по умолчанию сегодня")
    p_report.set_defaults(func=cmd_report)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
