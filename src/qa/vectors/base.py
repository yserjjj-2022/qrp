"""Абстрактные интерфейсы векторного поиска."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class VectorStore(ABC):
    """Интерфейс векторного хранилища."""

    @abstractmethod
    def add(self, texts: list[str], ids: list[int]) -> None:
        """Добавить тексты с идентификаторами.

        Args:
            texts: список текстов.
            ids: список идентификаторов.
        """
        ...

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Найти похожие тексты.

        Args:
            query: поисковый запрос.
            top_k: количество результатов.

        Returns:
            Список результатов с id и score.
        """
        ...
