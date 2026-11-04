"""Заглушка LLM: не делает сетевых вызовов."""

from __future__ import annotations

from typing import Any

from qa.llm.base import LlmProvider


class StubProvider(LlmProvider):
    """Заглушка для тестов и разработки без сети."""

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        """Вернуть фиктивный JSON по схеме."""
        return self._fake_from_schema(schema)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Вернуть список нулевых векторов фиксированной длины."""
        return [[0.0] * 64 for _ in texts]

    @staticmethod
    def _fake_from_schema(schema: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN401
        """Сгенерировать минимальный JSON по JSON-схеме."""
        result: dict[str, Any] = {}
        properties = schema.get("properties", {})
        for key, prop in properties.items():
            prop_type = prop.get("type", "string")
            if prop_type == "string":
                result[key] = ""
            elif prop_type == "number" or prop_type == "integer":
                result[key] = 0
            elif prop_type == "boolean":
                result[key] = False
            elif prop_type == "array":
                result[key] = []
            elif prop_type == "object":
                result[key] = {}
        return result
