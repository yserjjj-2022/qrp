"""Тесты интерфейсов llm и vectors."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from qa.llm import get_llm_provider
from qa.llm.base import LlmProvider
from qa.llm.stub import StubProvider
from qa.vectors import get_vector_store
from qa.vectors.base import VectorStore
from qa.vectors.stub import StubVectorStore


class TestLlmInterface:
    """Проверка интерфейсов LLM."""

    def test_stub_provider(self) -> None:
        stub = get_llm_provider("stub")
        assert isinstance(stub, LlmProvider)
        assert isinstance(stub, StubProvider)

    def test_stub_complete_json(self) -> None:
        stub = StubProvider()
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
                "active": {"type": "boolean"},
            },
        }
        result = stub.complete_json("test prompt", schema)
        assert isinstance(result, dict)
        assert result["name"] == ""
        assert result["count"] == 0
        assert result["active"] is False

    def test_stub_complete_json_validates_schema(self) -> None:
        stub = StubProvider()
        schema = {"type": "object", "properties": {"items": {"type": "array"}}}
        result = stub.complete_json("", schema)
        assert isinstance(result["items"], list)

    def test_stub_embed(self) -> None:
        stub = StubProvider()
        vectors = stub.embed(["hello", "world"])
        assert len(vectors) == 2
        assert all(isinstance(v, list) for v in vectors)
        assert all(len(v) == 64 for v in vectors)

    def test_unknown_provider(self) -> None:
        with pytest.raises(ValueError, match="Неизвестный LLM-провайдер"):
            get_llm_provider("unknown")

    def test_stub_isolation(self) -> None:
        """Заглушка не импортирует LLM SDK из других модулей."""
        # Проверяем, что SDK не импортирован за пределами llm/
        forbidden_packages = {"openai", "anthropic", "google", "vertexai", "litellm"}
        installed_packages = {name for name in sys.modules if any(pkg in name for pkg in forbidden_packages)}
        assert len(installed_packages) == 0


class TestVectorsInterface:
    """Проверка интерфейсов векторного поиска."""

    def test_stub_vector_store(self) -> None:
        store = get_vector_store("stub")
        assert isinstance(store, VectorStore)
        assert isinstance(store, StubVectorStore)

    def test_stub_add(self) -> None:
        store = StubVectorStore()
        store.add(["text1", "text2"], [1, 2])
        assert len(store._store) == 2  # noqa: SLF001

    def test_stub_search(self) -> None:
        store = StubVectorStore()
        store.add(["text1"], [1])
        results = store.search("query", top_k=5)
        assert results == []

    def test_unknown_backend(self) -> None:
        with pytest.raises(ValueError, match="Неизвестный векторный бэкенд"):
            get_vector_store("unknown")


def test_no_sdk_imported_outside_llm() -> None:
    """Прямых вызовов SDK вне llm/ нет (проверка импортов)."""
    # Импорт всех модулей проекта
    _project_root = Path(__file__).resolve().parent.parent
    src_dir = _project_root / "src"
    for py_file in src_dir.rglob("*.py"):
        if py_file.name == "__init__.py" and py_file.parent.name in ("llm", "vectors"):
            continue
        content = py_file.read_text(encoding="utf-8")
        # Проверяем, что не импортируются SDK напрямую
        forbidden = ["import openai", "from openai", "import anthropic", "from anthropic"]
        for pattern in forbidden:
            assert pattern not in content, f"{py_file} импортирует {pattern}"
