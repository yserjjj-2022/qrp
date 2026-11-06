"""Явные ошибки LLM-слоя."""

from __future__ import annotations


class LlmError(Exception):
    """Базовая ошибка LLM-слоя."""


class LlmConfigError(LlmError):
    """Не хватает настройки для вызова провайдера."""


class LlmRequestError(LlmError):
    """Сетевой вызов не удался после повторов."""


class LlmValidationError(LlmError):
    """Ответ модели не является JSON по схеме.

    Вызывающая сторона трактует это как «не размечено».
    """


class LlmCallLimitError(LlmError):
    """Превышен лимит вызовов LLM за прогон."""
