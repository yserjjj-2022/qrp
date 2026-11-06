"""Файловый кэш ответов LLM.

Ключ: sha256 от модели, промпта, схемы и параметров запроса. Содержимое кэша
не пишется в логи: логируются только ключ, размер и признак попадания.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def make_cache_key(model: str, prompt: str, schema: dict[str, Any], params: dict[str, Any]) -> str:
    """Посчитать ключ кэша по модели, промпту, схеме и параметрам."""
    payload = json.dumps(
        {"model": model, "prompt": prompt, "schema": schema, "params": params},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class FileCache:
    """Кэш ответов модели в одном каталоге."""

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)

    def get(self, key: str) -> dict[str, Any] | None:
        """Прочитать запись кэша; None, если записи нет или она повреждена."""
        path = self._dir / f"{key}.json"
        if not path.is_file():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            logger.warning("кэш LLM: запись key=%s повреждена, читаем как промах", key[:12])
            return None
        if not isinstance(payload, dict):
            return None
        logger.info("кэш LLM: попадание key=%s", key[:12])
        return payload

    def put(self, key: str, value: dict[str, Any]) -> None:
        """Записать результат в кэш. Ошибка записи не прерывает прогон."""
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            path = self._dir / f"{key}.json"
            tmp_path = path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(value, f, ensure_ascii=False)
            tmp_path.replace(path)
        except OSError:
            logger.warning("кэш LLM: запись key=%s не удалась", key[:12])
            return
        logger.info("кэш LLM: запись key=%s", key[:12])
