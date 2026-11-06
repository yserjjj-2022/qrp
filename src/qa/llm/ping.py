"""Служебный вызов LLM: проверка эндпоинта и модели на синтетическом тексте."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from qa.llm.base import LlmResult
from qa.llm.errors import LlmError
from qa.settings import Settings

PING_PROMPT = "Верни JSON-объект с единственным полем ok=true. Это служебная проверка."
PING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class PingOutcome:
    """Результат проверки для вывода оператору."""

    ok: bool
    model_requested: str
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_ms: float
    cached: bool = False


def ping_llm(settings: Settings, *, normalize: bool = False) -> PingOutcome:
    """Сделать один запрос к настроенному провайдеру.

    Реальные данные не отправляются: промпт синтетический.
    """
    from qa.llm import provider_from_settings

    llm = settings.llm
    requested = llm.model_normalize if normalize else llm.model
    provider = provider_from_settings(settings, normalize=normalize)
    started = time.monotonic()
    result: LlmResult = provider.complete_json_detailed(PING_PROMPT, PING_SCHEMA)
    elapsed_ms = (time.monotonic() - started) * 1000
    return PingOutcome(
        ok=bool(result.data.get("ok")),
        model_requested=requested or "",
        model_used=result.model_used,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        elapsed_ms=elapsed_ms,
        cached=result.cached,
    )


def dry_run_llm(settings: Settings, *, normalize: bool = False) -> PingOutcome:
    """Показать, какой вызов был бы сделан, без обращения к сети."""
    llm = settings.llm
    requested = llm.model_normalize if normalize else llm.model
    prompt_tokens = max(1, len(PING_PROMPT) // 4)
    return PingOutcome(
        ok=True,
        model_requested=requested or "",
        model_used="",
        prompt_tokens=prompt_tokens,
        completion_tokens=0,
        elapsed_ms=0.0,
        cached=False,
    )


__all__ = ["LlmError", "PING_PROMPT", "PING_SCHEMA", "PingOutcome", "dry_run_llm", "ping_llm"]
