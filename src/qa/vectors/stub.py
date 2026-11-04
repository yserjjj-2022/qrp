"""Заглушка векторного поиска."""

from __future__ import annotations

from typing import Any

from qa.vectors.base import VectorStore


class StubVectorStore(VectorStore):
    """Заглушка: хранит в памяти, поиск возвращает пусто."""

    def __init__(self) -> None:
        self._store: dict[int, str] = {}

    def add(self, texts: list[str], ids: list[int]) -> None:
        for text, id_ in zip(texts, ids, strict=True):
            self._store[id_] = text

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:  # noqa: ARG002
        return []
