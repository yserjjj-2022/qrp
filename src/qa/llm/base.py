"""Абстрактные интерфейсы LLM."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LlmResult:
    """Результат вызова модели вместе со служебными данными.

    Attributes:
        data: валидированный JSON по схеме.
        model_used: значение model из ответа API (фактическая модель, а не алиас).
        prompt_tokens: токены промпта по данным API.
        completion_tokens: токены ответа по данным API.
        cached: результат взят из кэша, сетевого вызова не было.
        estimated: значения токенов оценены локально (dry-run).
    """

    data: dict[str, Any] = field(default_factory=dict)
    model_used: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    estimated: bool = False


class LlmProvider(ABC):
    """Интерфейс LLM-провайдера."""

    def complete_json_detailed(self, prompt: str, schema: dict[str, Any]) -> LlmResult:
        """Выполнить вызов и вернуть результат со служебными данными.

        Провайдеры без служебных данных возвращают пустые model_used и usage.
        """
        return LlmResult(data=self.complete_json(prompt, schema))

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
