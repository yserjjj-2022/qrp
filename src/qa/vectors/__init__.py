"""Фабрика векторных хранилищ по настройкам."""

from __future__ import annotations

from qa.vectors.base import VectorStore
from qa.vectors.stub import StubVectorStore


def get_vector_store(backend_name: str) -> VectorStore:
    """Вернуть хранилище по имени из настроек.

    Args:
        backend_name: имя из settings.vectors.backend.

    Returns:
        Экземпляр VectorStore.
    """
    if backend_name == "stub":
        return StubVectorStore()
    msg = f"Неизвестный векторный бэкенд: {backend_name!r}. Доступен: stub"
    raise ValueError(msg)
