"""Абстрактные интерфейсы LLM."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LlmProvider(ABC):
    """Интерфейс LLM-провайдера."""

    @abstractmethod
    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Отправить промпт, получить JSON по схеме.

        Args:
            prompt: текст промпта.
            schema: JSON-схема ожидаемого ответа.

        Returns:
            Валидированный JSON-ответ.
        """
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Получить векторы для списка текстов.

        Args:
            texts: список текстов.

        Returns:
            Список векторов (по одному на текст).
        """
        ...
