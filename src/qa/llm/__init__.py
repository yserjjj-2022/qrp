"""Фабрика LLM-провайдеров по настройкам."""

from __future__ import annotations

from qa.llm.base import LlmProvider
from qa.llm.stub import StubProvider


def get_llm_provider(provider_name: str) -> LlmProvider:
    """Вернуть провайдер по имени из настроек.

    Args:
        provider_name: имя из settings.llm.provider.

    Returns:
        Экземпляр LlmProvider.
    """
    if provider_name == "stub":
        return StubProvider()
    msg = f"Неизвестный LLM-провайдер: {provider_name!r}. Доступен: stub"
    raise ValueError(msg)
