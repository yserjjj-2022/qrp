"""Фабрика LLM-провайдеров по настройкам."""

from __future__ import annotations

from typing import Any

from qa.llm.base import LlmProvider
from qa.llm.errors import LlmConfigError
from qa.llm.http_provider import OpenAICompatibleProvider
from qa.llm.stub import StubProvider
from qa.settings import LlmSettings, Settings

PROVIDER_STUB = "stub"
PROVIDER_OPENAI_COMPATIBLE = "openai_compatible"
PROVIDER_NAMES: tuple[str, ...] = (PROVIDER_STUB, PROVIDER_OPENAI_COMPATIBLE)


def get_llm_provider(
    provider_name: str,
    settings: LlmSettings | None = None,
    *,
    api_key: str | None = None,
    model: str | None = None,
    extra: dict[str, Any] | None = None,
) -> LlmProvider:
    """Вернуть провайдер по имени из настроек.

    Args:
        provider_name: имя из settings.llm.provider.
        settings: настройки LLM; обязательны для сетевого провайдера.
        api_key: ключ из окружения (в настройках не хранится).
        model: переопределить модель (например, LLM_MODEL_NORMALIZE).
        extra: дополнительные параметры запроса.

    Returns:
        Экземпляр LlmProvider.
    """
    if provider_name == PROVIDER_STUB:
        return StubProvider()
    if provider_name == PROVIDER_OPENAI_COMPATIBLE:
        if settings is None:
            msg = "Для openai_compatible нужны настройки LLM"
            raise LlmConfigError(msg)
        endpoint = settings.endpoint
        if not endpoint:
            msg = "LLM_ENDPOINT не задан в настройках"
            raise LlmConfigError(msg)
        chosen_model = model or settings.model
        if not chosen_model:
            msg = "LLM_MODEL не задан в настройках"
            raise LlmConfigError(msg)
        if not api_key:
            msg = "LLM_API_KEY не задан в окружении"
            raise LlmConfigError(msg)
        return OpenAICompatibleProvider(
            endpoint=endpoint,
            model=chosen_model,
            api_key=api_key,
            cache_dir=settings.cache_dir,
            max_calls_per_run=settings.max_calls_per_run,
            extra=extra,
        )
    msg = f"Неизвестный LLM-провайдер: {provider_name!r}. Допустимы: {', '.join(PROVIDER_NAMES)}"
    raise ValueError(msg)


def provider_from_settings(
    settings: Settings,
    *,
    normalize: bool = False,
    extra: dict[str, Any] | None = None,
) -> LlmProvider:
    """Собрать провайдера из настроек проекта.

    Args:
        settings: настройки проекта.
        normalize: использовать LLM_MODEL_NORMALIZE вместо LLM_MODEL.
        extra: дополнительные параметры запроса.
    """
    llm = settings.llm
    model = llm.model_normalize if normalize else llm.model
    return get_llm_provider(
        llm.provider,
        llm,
        api_key=settings.llm_api_key(),
        model=model,
        extra=extra,
    )
