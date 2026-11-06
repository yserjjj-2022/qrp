"""LLM-провайдер для OpenAI-совместимого HTTP-эндпоинта.

HTTP-клиент на стандартной библиотеке (urllib): новых зависимостей нет.
Ключ, эндпоинт и модель приходят из настроек; в этом модуле их значений нет.
Логи содержат только статусы, размеры, хеши и задержки.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from qa.llm.base import LlmProvider, LlmResult
from qa.llm.cache import FileCache, make_cache_key
from qa.llm.errors import LlmConfigError, LlmRequestError, LlmValidationError
from qa.llm.limiter import CallLimiter
from qa.llm.validation import parse_json_object, validate_response

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 60.0
_TRANSPORT_ATTEMPTS = 3
_VALIDATION_ATTEMPTS = 3  # первый запрос и до двух повторов
_BACKOFF_BASE = 0.5
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class _ResponseFormatUnsupportedError(Exception):
    """Эндпоинт отверг response_format json_schema; нужен повтор без него."""

    def __init__(self, status: int, reason: str) -> None:
        self.status = status
        self.reason = reason
        super().__init__(f"HTTP {status}{f' ({reason})' if reason else ''}")


class OpenAICompatibleProvider(LlmProvider):
    """Провайдер OpenAI-совместимого API (chat/completions и embeddings)."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        api_key: str,
        cache_dir: str | None = None,
        max_calls_per_run: int = 100,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
        extra: dict[str, Any] | None = None,
        embedding_model: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not endpoint:
            msg = "LLM_ENDPOINT не задан в настройках"
            raise LlmConfigError(msg)
        if not model:
            msg = "LLM_MODEL не задан в настройках"
            raise LlmConfigError(msg)
        if not api_key:
            msg = "LLM_API_KEY не задан в окружении"
            raise LlmConfigError(msg)
        self._base_url = endpoint.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._cache = FileCache(cache_dir) if cache_dir else None
        self._limiter = CallLimiter(max_calls_per_run)
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._seed = seed
        self._extra = dict(extra or {})
        self._embedding_model = embedding_model
        self._timeout = timeout
        self._sleep = sleep
        self._supports_response_format = True

    @property
    def model(self) -> str:
        """Запрошенная модель (может быть алиасом)."""
        return self._model

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Отправить промпт и вернуть валидированный JSON."""
        return self.complete_json_detailed(prompt, schema).data

    def complete_json_detailed(self, prompt: str, schema: dict[str, Any]) -> LlmResult:
        """Выполнить запрос и вернуть JSON вместе с фактической моделью и usage."""
        params = self._params()
        cache_key = make_cache_key(self._model, prompt, schema, params)
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return _result_from_cache(cached)

        self._limiter.reserve()
        logger.info(
            "LLM: запрос model_hash=%s prompt_sha=%s prompt_len=%d",
            _hash(self._model),
            _hash(prompt),
            len(prompt),
        )
        last_error: LlmValidationError | None = None
        for attempt in range(1, _VALIDATION_ATTEMPTS + 1):
            envelope = self._request_chat(prompt, schema, params)
            try:
                data = _extract_json(envelope, schema)
            except LlmValidationError as exc:
                last_error = exc
                logger.warning("LLM: ответ не прошёл валидацию, попытка %d из %d", attempt, _VALIDATION_ATTEMPTS)
                continue
            result = LlmResult(
                data=data,
                model_used=_response_model(envelope) or self._model,
                prompt_tokens=_usage_value(envelope, "prompt_tokens"),
                completion_tokens=_usage_value(envelope, "completion_tokens"),
            )
            if self._cache is not None:
                self._cache.put(cache_key, _cache_payload(result))
            logger.info(
                "LLM: ответ ok model_used=%s prompt_tokens=%d completion_tokens=%d",
                result.model_used,
                result.prompt_tokens,
                result.completion_tokens,
            )
            return result
        msg = str(last_error) if last_error is not None else "Ответ модели не прошёл валидацию"
        raise LlmValidationError(msg)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Получить векторы для списка текстов."""
        if not texts:
            return []
        if not self._embedding_model:
            msg = "Модель эмбеддингов не задана в настройках"
            raise LlmConfigError(msg)
        self._limiter.reserve()
        envelope = self._post("/embeddings", {"model": self._embedding_model, "input": texts})
        data = envelope.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            msg = "Ответ эмбеддингов не соответствует числу текстов"
            raise LlmRequestError(msg)
        ordered = sorted(data, key=_embedding_index)
        vectors: list[list[float]] = []
        for item in ordered:
            if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                msg = "В ответе эмбеддингов нет вектора"
                raise LlmRequestError(msg)
            vectors.append([float(value) for value in item["embedding"]])
        return vectors

    def _params(self) -> dict[str, Any]:
        """Параметры запроса: без значений, зашитых под конкретную модель."""
        params: dict[str, Any] = {"temperature": self._temperature}
        if self._max_tokens is not None:
            params["max_tokens"] = self._max_tokens
        if self._seed is not None:
            params["seed"] = self._seed
        params.update(self._extra)
        return params

    def _build_payload(
        self,
        prompt: str,
        schema: dict[str, Any],
        params: dict[str, Any],
        use_response_format: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            **params,
        }
        if use_response_format:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "llm_response", "schema": schema, "strict": True},
            }
        return payload

    def _request_chat(self, prompt: str, schema: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """Запрос с фоллбэком на обычный JSON в тексте ответа."""
        use_response_format = self._supports_response_format
        payload = self._build_payload(prompt, schema, params, use_response_format)
        try:
            return self._post("/chat/completions", payload)
        except _ResponseFormatUnsupportedError as exc:
            if not use_response_format:
                msg = f"LLM-эндпоинт отверг запрос: {exc.reason or exc.status}"
                raise LlmRequestError(msg) from None
            self._supports_response_format = False
            logger.info(
                "LLM: response_format json_schema не поддержан (status=%d reason=%s), повтор без него",
                exc.status,
                exc.reason or "—",
            )
            payload = self._build_payload(prompt, schema, params, False)
            return self._post("/chat/completions", payload)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST с повторами на 429 и 5xx, экспоненциальной паузой."""
        url = f"{self._base_url}{path}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        use_response_format = "response_format" in payload
        last_status: int | None = None
        for attempt in range(1, _TRANSPORT_ATTEMPTS + 1):
            request = Request(  # noqa: S310 - адрес берётся из настроек
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                method="POST",
            )
            started = time.monotonic()
            try:
                with urlopen(request, timeout=self._timeout) as response:  # noqa: S310
                    raw = response.read()
                    status = int(response.status)
            except HTTPError as exc:
                status = int(exc.code)
                raw = b""
                try:
                    raw = exc.read()
                except OSError:
                    raw = b""
                exc.close()
            except (URLError, TimeoutError) as exc:
                logger.warning("LLM: сеть недоступна, попытка %d из %d", attempt, _TRANSPORT_ATTEMPTS)
                if attempt >= _TRANSPORT_ATTEMPTS:
                    msg = "Сетевая ошибка при обращении к LLM-эндпоинту"
                    raise LlmRequestError(msg) from exc
                self._sleep(_BACKOFF_BASE * 2 ** (attempt - 1))
                continue

            latency_ms = (time.monotonic() - started) * 1000
            logger.info("LLM HTTP: path=%s status=%d latency_ms=%.0f size=%d", path, status, latency_ms, len(raw))

            if status == 200:
                return _parse_envelope(raw)
            if status in _RETRYABLE_STATUS:
                last_status = status
                if attempt < _TRANSPORT_ATTEMPTS:
                    self._sleep(_BACKOFF_BASE * 2 ** (attempt - 1))
                    continue
                break
            reason = _error_reason(raw)
            if reason:
                logger.warning("LLM HTTP: status=%d reason=%s", status, reason)
            if status in (400, 422) and use_response_format and _mentions_response_format(raw):
                raise _ResponseFormatUnsupportedError(status, reason)
            msg = f"LLM-эндпоинт вернул HTTP {status}{f' ({reason})' if reason else ''}"
            raise LlmRequestError(msg)

        msg = f"LLM-эндпоинт недоступен после {_TRANSPORT_ATTEMPTS} попыток (HTTP {last_status})"
        raise LlmRequestError(msg)


def _error_body(raw: bytes) -> dict[str, Any]:
    """Тело ошибки API, если это JSON-объект."""
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _error_reason(raw: bytes) -> str:
    """Тип и код ошибки из тела ответа.

    Возвращается только машинная диагностика (`error.type`, `error.code`,
    `error.param`): текст сообщения модели и запроса не логируется.
    """
    error = _error_body(raw).get("error")
    if not isinstance(error, dict):
        return ""
    parts: list[str] = []
    for key in ("type", "code", "param"):
        value = error.get(key)
        if isinstance(value, str) and value:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _mentions_response_format(raw: bytes) -> bool:
    """Похоже ли, что эндпоинт отверг именно response_format.

    Смотрим на структурные поля тела ошибки; свободный текст не используется,
    чтобы не путать отказ по имени модели с отказом по формату ответа.
    """
    error = _error_body(raw).get("error")
    if not isinstance(error, dict):
        return False
    fields = [error.get(key) for key in ("type", "code", "param")]
    joined = " ".join(value for value in fields if isinstance(value, str)).lower()
    return "response_format" in joined or "json_schema" in joined


def _hash(value: str) -> str:
    """Короткий хеш для логов: содержимое не раскрывается."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _parse_envelope(raw: bytes) -> dict[str, Any]:
    """Разобрать конверт ответа API."""
    try:
        envelope = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg = "Ответ LLM-эндпоинта не является JSON"
        raise LlmRequestError(msg) from exc
    if not isinstance(envelope, dict):
        msg = "Ответ LLM-эндпоинта имеет неожидаемый формат"
        raise LlmRequestError(msg)
    return envelope


def _message_content(envelope: dict[str, Any]) -> str:
    """Текст ответа модели из choices[0].message.content."""
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices:
        msg = "В ответе модели нет choices"
        raise LlmValidationError(msg)
    first = choices[0]
    if not isinstance(first, dict):
        msg = "В ответе модели неожидаемый choice"
        raise LlmValidationError(msg)
    message = first.get("message")
    if not isinstance(message, dict):
        msg = "В ответе модели нет message"
        raise LlmValidationError(msg)
    parsed = message.get("parsed")
    if isinstance(parsed, dict):
        return json.dumps(parsed, ensure_ascii=False)
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, dict)]
        return "".join(part for part in parts if isinstance(part, str))
    msg = "В ответе модели нет текста"
    raise LlmValidationError(msg)


def _extract_json(envelope: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Достать JSON из текста ответа и проверить по схеме."""
    return validate_response(parse_json_object(_message_content(envelope)), schema)


def _response_model(envelope: dict[str, Any]) -> str:
    """Значение model из ответа API (фактическая модель вместо алиаса)."""
    model = envelope.get("model")
    return model if isinstance(model, str) else ""


def _usage_value(envelope: dict[str, Any], field_name: str) -> int:
    """Число токенов из блока usage."""
    usage = envelope.get("usage")
    if not isinstance(usage, dict):
        return 0
    value = usage.get(field_name)
    return int(value) if isinstance(value, int) else 0


def _embedding_index(item: Any) -> int:  # noqa: ANN401
    """Порядковый номер вектора в ответе эмбеддингов."""
    if isinstance(item, dict) and isinstance(item.get("index"), int):
        return int(item["index"])
    return 0


def _cache_payload(result: LlmResult) -> dict[str, Any]:
    """Сериализуемое содержимое записи кэша."""
    return {
        "data": result.data,
        "model_used": result.model_used,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
    }


def _result_from_cache(payload: dict[str, Any]) -> LlmResult:
    """Восстановить результат из записи кэша."""
    data = payload.get("data")
    model_used = payload.get("model_used")
    prompt_tokens = payload.get("prompt_tokens")
    completion_tokens = payload.get("completion_tokens")
    return LlmResult(
        data=data if isinstance(data, dict) else {},
        model_used=model_used if isinstance(model_used, str) else "",
        prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else 0,
        completion_tokens=completion_tokens if isinstance(completion_tokens, int) else 0,
        cached=True,
    )
