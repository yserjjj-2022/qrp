"""Версия кода для записи в run.

Версия пакета не меняется между правками, поэтому к ней добавляется короткий
хеш коммита. Git читается через subprocess, новых зависимостей нет.
"""

from __future__ import annotations

import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_PACKAGE = "qa-interview-analyzer"


def package_version() -> str:
    """Версия пакета; при отсутствии метаданных — пустая строка."""
    try:
        return version(_PACKAGE)
    except PackageNotFoundError:
        return ""


def git_short_hash(start: Path | None = None) -> str | None:
    """Короткий хеш текущего коммита, если рядом есть репозиторий git."""
    directory = start or Path(__file__).resolve().parent
    try:
        result = subprocess.run(  # noqa: S603 - фиксированные аргументы
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(directory),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def code_version(start: Path | None = None) -> str:
    """Строка версии кода: `0.1.0+abc1234` или только версия без git."""
    base = package_version()
    commit = git_short_hash(start)
    if base and commit:
        return f"{base}+{commit}"
    if base:
        return base
    return commit or ""
