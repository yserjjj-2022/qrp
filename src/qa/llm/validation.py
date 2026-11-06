"""Проверка JSON-ответа модели по JSON-схеме через pydantic."""

from __future__ import annotations

import json
from functools import reduce
from operator import or_
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, create_model

from qa.llm.errors import LlmValidationError

_JSON_TYPES: dict[str, type[Any]] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def parse_json_object(text: str) -> dict[str, Any]:
    """Извлечь JSON-объект из текста ответа модели.

    Допускает обёртку в markdown-блок: берётся фрагмент от первой `{` до последней `}`.
    """
    candidate = text.strip()
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end <= start:
            msg = "В ответе модели нет JSON-объекта"
            raise LlmValidationError(msg) from None
        try:
            data = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            msg = "Ответ модели не является корректным JSON"
            raise LlmValidationError(msg) from None
    if not isinstance(data, dict):
        msg = "Ответ модели не является JSON-объектом"
        raise LlmValidationError(msg)
    return data


def validate_response(payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Проверить ответ по JSON-схеме и вернуть нормализованный словарь.

    Raises:
        LlmValidationError: ответ не соответствует схеме. Текст ошибки не содержит
            значений ответа, чтобы содержимое не попало в логи.
    """
    adapter: TypeAdapter[Any] = TypeAdapter(_type_from_schema(schema, "LlmResponse", [0]))
    try:
        value = adapter.validate_python(payload)
    except ValidationError as exc:
        msg = f"Ответ модели не соответствует JSON-схеме: ошибок {exc.error_count()}"
        raise LlmValidationError(msg) from None
    dumped = adapter.dump_python(value, mode="json")
    if not isinstance(dumped, dict):
        msg = "Ожидался JSON-объект"
        raise LlmValidationError(msg)
    return dumped


def _type_from_schema(schema: dict[str, Any], name: str, counter: list[int]) -> Any:  # noqa: ANN401
    """Построить python-тип для валидации по фрагменту JSON-схемы."""
    variants = schema.get("anyOf") or schema.get("oneOf")
    if isinstance(variants, list) and variants:
        types = [_type_from_schema(item, f"{name}Variant{index}", counter) for index, item in enumerate(variants)]
        return reduce(or_, types)
    if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
        # Literal с переменным набором значений: тип вычисляется во время выполнения.
        return cast(Any, Literal)[tuple(schema["enum"])]

    schema_type = schema.get("type")
    if schema_type == "object" or "properties" in schema:
        return _object_type(schema, name, counter)
    if schema_type == "array":
        items = schema.get("items")
        item_type = _type_from_schema(items, f"{name}Item", counter) if isinstance(items, dict) else Any
        return cast(Any, list)[item_type]
    return _JSON_TYPES.get(schema_type, Any) if isinstance(schema_type, str) else Any


def _object_type(schema: dict[str, Any], name: str, counter: list[int]) -> type[BaseModel]:
    """Построить pydantic-модель для JSON-объекта."""
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    fields: dict[str, Any] = {}
    for field_name, prop in properties.items():
        counter[0] += 1
        field_type = _type_from_schema(prop, f"{name}{counter[0]}", counter) if isinstance(prop, dict) else Any
        fields[field_name] = (field_type, ...) if field_name in required else (field_type | None, None)
    extra: Literal["allow", "forbid"] = "forbid" if schema.get("additionalProperties") is False else "allow"
    return create_model(name, __config__=ConfigDict(extra=extra), **fields)
