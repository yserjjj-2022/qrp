"""CLI entry point: qa-pipeline."""

from __future__ import annotations

import argparse

from qa.log import setup_logging
from qa.settings import Settings


def cmd_load_codebook(args: argparse.Namespace) -> None:
    """Load codebook from an Excel file."""
    settings = Settings.load()
    run_id = setup_logging(settings.log_level)
    print(f"load-codebook: path={args.file}, run_id={run_id}")


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

    args = parser.parse_args()
    args.func(args)
    return 0
