"""CLI entry point: qa-pipeline."""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

from qa.db.connection import get_connection
from qa.db.migrations import migrate
from qa.log import setup_logging
from qa.settings import Settings


def _open_db(settings: Settings) -> sqlite3.Connection:
    """Открыть БД из настроек и применить миграции."""
    db_path = Path(settings.db_path)
    backup_dir = Path(settings.data_dir) / "backups"
    conn = get_connection(db_path)
    migrate(conn, db_path, Path("migrations"), backup_dir)
    return conn


def cmd_load_codebook(args: argparse.Namespace) -> None:
    """Load codebook from an Excel file."""
    settings = Settings.load()
    run_id = setup_logging(settings.log_level)
    from qa.pipeline.load_codebook import load_codebook

    conn = _open_db(settings)
    n_codes, n_questions, n_attrs = load_codebook(conn, Path(args.file))
    print(f"[{run_id}] кодбуков: кодов={n_codes}, вопросов={n_questions}, атрибутов={n_attrs}")
    conn.close()


def cmd_load_transcript(args: argparse.Namespace) -> None:
    """Load interview transcript (M1a)."""
    settings = Settings.load()
    run_id = setup_logging(settings.log_level)
    from qa.pipeline.load_transcript import build_replacement_map, load_meta, load_names_map, load_transcript

    meta = load_meta(Path(args.meta))
    extra: dict[str, str] = {}
    names_map_path = os.environ.get("NAMES_MAP_PATH")
    if names_map_path:
        extra = load_names_map(names_map_path)
    replacement_map = build_replacement_map(meta, extra)

    conn = _open_db(settings)
    result = load_transcript(conn, Path(args.transcript), meta, replacement_map, settings.pipeline)
    if result.skipped:
        print(f"[{run_id}] интервью {meta.interview_id} уже загружено, дублей нет")
    else:
        print(
            f"[{run_id}] интервью {meta.interview_id}: записей={result.n_turns}, "
            f"реплик={result.n_utterances}, pii_flag={result.n_pii_flagged}"
        )
    conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="qa-pipeline",
        description="Анализатор интервью: тематический анализ транскриптов",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # load-codebook
    p_load = subparsers.add_parser("load-codebook", help="Загрузить кодбук из Excel")
    p_load.add_argument("file", help="Путь к файлу приложения Б (.xlsx)")
    p_load.set_defaults(func=cmd_load_codebook)

    # load-transcript
    p_tr = subparsers.add_parser("load-transcript", help="Загрузить транскрипт интервью (M1a)")
    p_tr.add_argument("transcript", help="Путь к файлу транскрипта (.txt, UTF-8)")
    p_tr.add_argument(
        "--meta", default="data/interview.yaml", help="Путь к interview.yaml (default: data/interview.yaml)"
    )
    p_tr.set_defaults(func=cmd_load_transcript)

    args = parser.parse_args()
    args.func(args)
    return 0
