"""Лимит вызовов LLM за прогон и подсчёт для dry-run."""

from __future__ import annotations

import logging

from qa.llm.errors import LlmCallLimitError

logger = logging.getLogger(__name__)


class CallLimiter:
    """Счётчик вызовов модели на один прогон.

    Попадание в кэш лимит не расходует: инкремент выполняется только перед
    фактической отправкой запроса.
    """

    def __init__(self, max_calls: int) -> None:
        if max_calls < 1:
            msg = "LLM_MAX_CALLS_PER_RUN должен быть не меньше 1"
            raise ValueError(msg)
        self._max_calls = max_calls
        self._used = 0

    @property
    def max_calls(self) -> int:
        """Настроенный лимит."""
        return self._max_calls

    @property
    def used(self) -> int:
        """Сколько вызовов уже израсходовано."""
        return self._used

    def reserve(self) -> int:
        """Зарезервировать вызов.

        Raises:
            LlmCallLimitError: лимит исчерпан.
        """
        if self._used >= self._max_calls:
            msg = f"Превышен лимит вызовов LLM за прогон: {self._max_calls}. Прогон остановлен"
            raise LlmCallLimitError(msg)
        self._used += 1
        logger.info("LLM: вызов %d из %d", self._used, self._max_calls)
        return self._used


class DryRunCounter:
    """Подсчёт предстоящих вызовов и оценка токенов без обращения к сети."""

    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0

    def add(self, prompt: str, expected_output_tokens: int = 0) -> None:
        """Учесть один будущий вызов.

        Оценка токенов грубая: около 4 символов на токен. Точность для dry-run
        не нужна, нужен порядок величины.
        """
        self.calls += 1
        self.prompt_tokens += max(1, len(prompt) // 4)
        self.prompt_tokens += expected_output_tokens
