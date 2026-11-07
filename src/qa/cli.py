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
        f"вариантов={len(result.variants)}, отклонено={len(result.rejected)}, пропущено={len(result.skipped)}"
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
    print(f"  отказы автопроверок: {rejections}")
    print(f"  new_term={stats.new_terms}, токены: prompt={stats.prompt_tokens}, completion={stats.completion_tokens}")
    return 0


def _settings_error_summary(exc: Exception) -> str:
    """Краткий текст ошибки настроек без значений полей."""
    from pydantic import ValidationError

    if isinstance(exc, ValidationError):
        messages = [str(error.get("msg", "")) for error in exc.errors()]
        return "; ".join(message for message in messages if message) or "некорректные настройки"
    return "некорректные настройки"


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
    p_norm.set_defaults(func=cmd_normalize)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
