"""Тесты OpenAI-совместимого провайдера на локальном HTTP-сервере.

Сеть не используется: эндпоинт поднимается в отдельном потоке на localhost.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from qa.cli import main
from qa.llm import get_llm_provider
from qa.llm.base import LlmProvider
from qa.llm.errors import LlmCallLimitError, LlmRequestError, LlmValidationError
from qa.llm.http_provider import OpenAICompatibleProvider
from qa.llm.ping import dry_run_llm, ping_llm
from qa.llm.validation import validate_response
from qa.settings import LlmSettings, Settings, load_env_file

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}
_PROMPT = "синтетический промпт для теста"
_FAKE_KEY = "test-key-not-a-secret"
_ALIAS = "alias-under-test"
_RESOLVED = "resolved-model-under-test"


@dataclass
class _ServerState:
    """Сценарий ответов сервера и счётчики обращений."""

    responses: list[tuple[int, dict[str, Any]]] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)
    default_status: int = 200

    def next_response(self) -> tuple[int, dict[str, Any]]:
        if self.responses:
            return self.responses.pop(0)
        return self.default_status, _ok_envelope()


def _ok_envelope(*, model: str = _RESOLVED, content: dict[str, Any] | None = None) -> dict[str, Any]:
    """Конверт успешного ответа chat/completions."""
    return {
        "model": model,
        "choices": [{"message": {"content": json.dumps(content or {"ok": True})}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }


def _make_handler(state: _ServerState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - имя метода задано http.server
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                body = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                body = {}
            if isinstance(body, dict):
                body["_path"] = self.path
                body["_authorization_present"] = bool(self.headers.get("Authorization"))
                state.requests.append(body)
            status, payload = state.next_response()
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    return Handler


@pytest.fixture()
def server() -> Iterator[tuple[str, _ServerState]]:
    state = _ServerState()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = httpd.server_address[:2]
        yield f"http://{host}:{port}", state
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _provider(endpoint: str, cache_dir: Path | None = None, **kwargs: Any) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        endpoint=endpoint,
        model=_ALIAS,
        api_key=_FAKE_KEY,
        cache_dir=str(cache_dir) if cache_dir else None,
        sleep=lambda _seconds: None,
        **kwargs,
    )


class TestCompleteJson:
    """Успешный путь и служебные поля результата."""

    def test_success_uses_response_format(self, server: tuple[str, _ServerState]) -> None:
        endpoint, state = server
        result = _provider(endpoint).complete_json_detailed(_PROMPT, _SCHEMA)
        assert result.data == {"ok": True}
        assert len(state.requests) == 1
        request = state.requests[0]
        assert request["_path"] == "/chat/completions"
        assert request["_authorization_present"] is True
        assert request["temperature"] == 0
        assert request["response_format"]["type"] == "json_schema"
        assert request["response_format"]["json_schema"]["schema"] == _SCHEMA
        assert "api_key" not in json.dumps(request)

    def test_model_used_from_response(self, server: tuple[str, _ServerState]) -> None:
        endpoint, _state = server
        provider = _provider(endpoint)
        assert provider.model == _ALIAS
        result = provider.complete_json_detailed(_PROMPT, _SCHEMA)
        assert result.model_used == _RESOLVED
        assert result.prompt_tokens == 11
        assert result.completion_tokens == 7
        assert result.cached is False

    def test_falls_back_when_response_format_rejected(self, server: tuple[str, _ServerState]) -> None:
        endpoint, state = server
        state.responses = [
            (400, {"error": {"type": "invalid_request_error", "param": "response_format"}}),
            (200, _ok_envelope()),
        ]
        result = _provider(endpoint).complete_json_detailed(_PROMPT, _SCHEMA)
        assert result.data == {"ok": True}
        assert len(state.requests) == 2
        assert "response_format" in state.requests[0]
        assert "response_format" not in state.requests[1]

    def test_bad_request_with_other_reason_is_not_retried(self, server: tuple[str, _ServerState]) -> None:
        """400 по другой причине не маскируется под отказ response_format."""
        endpoint, state = server
        state.responses = [(400, {"error": {"type": "invalid_request_error", "code": "model_not_found"}})]
        with pytest.raises(LlmRequestError) as exc_info:
            _provider(endpoint).complete_json(_PROMPT, _SCHEMA)
        assert len(state.requests) == 1
        assert "model_not_found" in str(exc_info.value)
        assert "response_format" in state.requests[0]

    def test_error_text_is_not_logged(self, server: tuple[str, _ServerState], caplog: pytest.LogCaptureFixture) -> None:
        """Тело ошибки не попадает в логи: только машинные поля."""
        endpoint, _state = server
        secret_text = "внутренний-текст-запроса"
        server_state = _state
        server_state.responses = [(400, {"error": {"type": "invalid_request_error", "message": secret_text}})]
        with caplog.at_level("DEBUG"), pytest.raises(LlmRequestError):
            _provider(endpoint).complete_json(_PROMPT, _SCHEMA)
        assert secret_text not in caplog.text
        assert "invalid_request_error" in caplog.text

    def test_response_format_not_retried_after_success(self, server: tuple[str, _ServerState]) -> None:
        endpoint, state = server
        state.responses = [
            (400, {"error": {"code": "unknown_parameter", "param": "response_format"}}),
            (200, _ok_envelope()),
            (200, _ok_envelope()),
        ]
        provider = _provider(endpoint)
        provider.complete_json_detailed(_PROMPT, _SCHEMA)
        provider.complete_json_detailed(_PROMPT + "2", _SCHEMA)
        assert "response_format" not in state.requests[2]

    def test_json_extracted_from_wrapped_text(self, server: tuple[str, _ServerState]) -> None:
        endpoint, state = server
        state.responses = [
            (200, {"model": _RESOLVED, "choices": [{"message": {"content": 'Вот ответ:\n```json\n{"ok": true}\n```'}}]})
        ]
        assert _provider(endpoint).complete_json(_PROMPT, _SCHEMA) == {"ok": True}

    def test_seed_and_extra_passed_through(self, server: tuple[str, _ServerState]) -> None:
        endpoint, state = server
        provider = _provider(endpoint, seed=42, max_tokens=64, extra={"top_p": 0.9})
        provider.complete_json(_PROMPT, _SCHEMA)
        request = state.requests[0]
        assert request["seed"] == 42
        assert request["max_tokens"] == 64
        assert request["top_p"] == 0.9

    def test_optional_params_absent_by_default(self, server: tuple[str, _ServerState]) -> None:
        endpoint, state = server
        _provider(endpoint).complete_json(_PROMPT, _SCHEMA)
        assert "seed" not in state.requests[0]
        assert "max_tokens" not in state.requests[0]


class TestRetries:
    """Повторы на 429/5xx и на ошибке валидации."""

    def test_retry_on_429(self, server: tuple[str, _ServerState]) -> None:
        endpoint, state = server
        state.responses = [(429, {"error": "rate limit"}), (200, _ok_envelope())]
        result = _provider(endpoint).complete_json_detailed(_PROMPT, _SCHEMA)
        assert result.data == {"ok": True}
        assert len(state.requests) == 2

    def test_retry_on_500_exhausted(self, server: tuple[str, _ServerState]) -> None:
        endpoint, state = server
        state.default_status = 500
        with pytest.raises(LlmRequestError):
            _provider(endpoint).complete_json(_PROMPT, _SCHEMA)
        assert len(state.requests) == 3

    def test_invalid_json_retried_twice_then_fails(self, server: tuple[str, _ServerState]) -> None:
        endpoint, state = server
        state.responses = [
            (200, {"model": _RESOLVED, "choices": [{"message": {"content": "не json"}}]}),
            (200, {"model": _RESOLVED, "choices": [{"message": {"content": '{"ok": "нет"}'}}]}),
            (200, {"model": _RESOLVED, "choices": [{"message": {"content": '{"other": 1}'}}]}),
        ]
        with pytest.raises(LlmValidationError):
            _provider(endpoint).complete_json(_PROMPT, _SCHEMA)
        assert len(state.requests) == 3

    def test_schema_mismatch_recovers_on_retry(self, server: tuple[str, _ServerState]) -> None:
        endpoint, state = server
        state.responses = [
            (200, {"model": _RESOLVED, "choices": [{"message": {"content": '{"ok": "да"}'}}]}),
            (200, _ok_envelope()),
        ]
        assert _provider(endpoint).complete_json(_PROMPT, _SCHEMA) == {"ok": True}
        assert len(state.requests) == 2


class TestCache:
    """Кэш не расходует лимит и не ходит в сеть."""

    def test_cache_hit_skips_network(self, server: tuple[str, _ServerState], tmp_path: Path) -> None:
        endpoint, state = server
        provider = _provider(endpoint, cache_dir=tmp_path, max_calls_per_run=1)
        first = provider.complete_json_detailed(_PROMPT, _SCHEMA)
        second = provider.complete_json_detailed(_PROMPT, _SCHEMA)
        assert len(state.requests) == 1
        assert first.cached is False
        assert second.cached is True
        assert second.data == first.data
        assert second.model_used == _RESOLVED

    def test_different_prompt_misses_cache(self, server: tuple[str, _ServerState], tmp_path: Path) -> None:
        endpoint, state = server
        provider = _provider(endpoint, cache_dir=tmp_path, max_calls_per_run=5)
        provider.complete_json_detailed(_PROMPT, _SCHEMA)
        provider.complete_json_detailed(_PROMPT + " иначе", _SCHEMA)
        assert len(state.requests) == 2

    def test_corrupted_cache_treated_as_miss(self, server: tuple[str, _ServerState], tmp_path: Path) -> None:
        endpoint, state = server
        provider = _provider(endpoint, cache_dir=tmp_path, max_calls_per_run=5)
        provider.complete_json_detailed(_PROMPT, _SCHEMA)
        for path in tmp_path.glob("*.json"):
            path.write_text("{битый", encoding="utf-8")
        provider.complete_json_detailed(_PROMPT, _SCHEMA)
        assert len(state.requests) == 2


class TestCallLimit:
    """Лимит вызовов за прогон."""

    def test_limit_stops_run(self, server: tuple[str, _ServerState]) -> None:
        endpoint, state = server
        provider = _provider(endpoint, max_calls_per_run=1)
        provider.complete_json(_PROMPT, _SCHEMA)
        with pytest.raises(LlmCallLimitError, match="лимит вызовов"):
            provider.complete_json(_PROMPT + " второй", _SCHEMA)
        assert len(state.requests) == 1

    def test_cache_hit_does_not_spend_limit(self, server: tuple[str, _ServerState], tmp_path: Path) -> None:
        endpoint, _state = server
        provider = _provider(endpoint, cache_dir=tmp_path, max_calls_per_run=1)
        provider.complete_json(_PROMPT, _SCHEMA)
        provider.complete_json(_PROMPT, _SCHEMA)
        assert provider._limiter.used == 1  # noqa: SLF001 - проверяем расход лимита


class TestValidation:
    """Проверка схемы отдельно от транспорта."""

    def test_required_field_missing(self) -> None:
        with pytest.raises(LlmValidationError):
            validate_response({}, _SCHEMA)

    def test_extra_field_forbidden(self) -> None:
        with pytest.raises(LlmValidationError):
            validate_response({"ok": True, "extra": 1}, _SCHEMA)

    def test_nested_and_array_schema(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"code": {"type": "string"}, "score": {"type": "number"}},
                        "required": ["code"],
                    },
                },
                "total": {"type": "integer"},
            },
            "required": ["items"],
        }
        result = validate_response({"items": [{"code": "A", "score": 0.5}], "total": 1}, schema)
        assert result["items"][0]["code"] == "A"

    def test_error_message_has_no_values(self) -> None:
        with pytest.raises(LlmValidationError) as exc_info:
            validate_response({"ok": "секретное-значение"}, _SCHEMA)
        assert "секретное-значение" not in str(exc_info.value)


class TestProviderConstruction:
    """Фабрика и настройки."""

    def test_factory_builds_http_provider(self) -> None:
        llm = LlmSettings(
            provider="openai_compatible",
            endpoint="http://127.0.0.1:1",
            model=_ALIAS,
            cache_dir="data/cache",
        )
        provider = get_llm_provider("openai_compatible", llm, api_key=_FAKE_KEY)
        assert isinstance(provider, LlmProvider)
        assert isinstance(provider, OpenAICompatibleProvider)

    def test_factory_requires_key(self) -> None:
        llm = LlmSettings(provider="openai_compatible", endpoint="http://127.0.0.1:1", model=_ALIAS)
        with pytest.raises(Exception, match="LLM_API_KEY"):
            get_llm_provider("openai_compatible", llm, api_key=None)

    def test_env_overrides_config(self) -> None:
        env = {
            "LLM_PROVIDER": "openai_compatible",
            "LLM_MODEL": _ALIAS,
            "LLM_MODEL_NORMALIZE": _ALIAS + "-norm",
            "LLM_ENDPOINT": "http://127.0.0.1:1",
            "LLM_CACHE_DIR": "data/other-cache",
            "LLM_MAX_CALLS_PER_RUN": "7",
            "LLM_API_KEY": _FAKE_KEY,
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.load()
            assert settings.llm.provider == "openai_compatible"
            assert settings.llm.model == _ALIAS
            assert settings.llm.model_normalize == _ALIAS + "-norm"
            assert settings.llm.cache_dir == "data/other-cache"
            assert settings.llm.max_calls_per_run == 7
            assert settings.llm_api_key() == _FAKE_KEY

    def test_api_key_not_stored_in_settings(self) -> None:
        with patch.dict(os.environ, {"LLM_API_KEY": _FAKE_KEY}, clear=True):
            settings = Settings.load()
        dumped = settings.model_dump_json()
        assert _FAKE_KEY not in dumped
        assert not hasattr(settings.llm, "api_key")


class TestEmbeddings:
    """Векторы через тот же провайдер."""

    def test_embed_returns_ordered_vectors(self, server: tuple[str, _ServerState]) -> None:
        endpoint, state = server
        state.responses = [
            (
                200,
                {
                    "data": [
                        {"index": 1, "embedding": [0.3, 0.4]},
                        {"index": 0, "embedding": [0.1, 0.2]},
                    ]
                },
            )
        ]
        provider = _provider(endpoint, embedding_model=_ALIAS + "-embed")
        vectors = provider.embed(["a", "b"])
        assert vectors == [[0.1, 0.2], [0.3, 0.4]]
        assert state.requests[0]["_path"] == "/embeddings"
        assert state.requests[0]["input"] == ["a", "b"]


class TestLlmPing:
    """Команда llm-пинга и режим dry-run."""

    def _settings(self, endpoint: str) -> Settings:
        return Settings(
            llm=LlmSettings(
                provider="openai_compatible",
                endpoint=endpoint,
                model=_ALIAS,
                model_normalize=_ALIAS + "-norm",
                cache_dir="",
                max_calls_per_run=5,
            )
        )

    def test_ping_reports_model_used(self, server: tuple[str, _ServerState]) -> None:
        endpoint, state = server
        state.responses = [(200, _ok_envelope(content={"ok": True}))]
        with patch.dict(os.environ, {"LLM_API_KEY": _FAKE_KEY}, clear=True):
            outcome = ping_llm(self._settings(endpoint))
        assert outcome.ok is True
        assert outcome.model_requested == _ALIAS
        assert outcome.model_used == _RESOLVED
        assert outcome.prompt_tokens == 11
        assert outcome.completion_tokens == 7
        assert len(state.requests) == 1
        assert state.requests[0]["model"] == _ALIAS

    def test_ping_normalize_uses_normalize_model(self, server: tuple[str, _ServerState]) -> None:
        endpoint, state = server
        state.responses = [(200, _ok_envelope(content={"ok": True}))]
        with patch.dict(os.environ, {"LLM_API_KEY": _FAKE_KEY}, clear=True):
            outcome = ping_llm(self._settings(endpoint), normalize=True)
        assert outcome.model_requested == _ALIAS + "-norm"
        assert state.requests[0]["model"] == _ALIAS + "-norm"

    def test_dry_run_sends_nothing(self, server: tuple[str, _ServerState]) -> None:
        endpoint, state = server
        with patch.dict(os.environ, {"LLM_API_KEY": _FAKE_KEY}, clear=True):
            outcome = dry_run_llm(self._settings(endpoint))
        assert outcome.prompt_tokens > 0
        assert outcome.completion_tokens == 0
        assert state.requests == []

    def test_dry_run_makes_no_request(self, server: tuple[str, _ServerState]) -> None:
        endpoint, state = server
        with patch.dict(os.environ, {"LLM_API_KEY": _FAKE_KEY}, clear=True):
            outcome = dry_run_llm(self._settings(endpoint))
        assert state.requests == []
        assert outcome.prompt_tokens > 0
        assert outcome.model_used == ""

    def test_cli_dry_run_prints_estimate(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
        monkeypatch.setenv("LLM_ENDPOINT", "http://127.0.0.1:1")
        monkeypatch.setenv("LLM_MODEL", _ALIAS)
        monkeypatch.setenv("LLM_API_KEY", _FAKE_KEY)
        monkeypatch.setattr(sys, "argv", ["qa-pipeline", "llm-ping", "--dry-run"])
        assert main() == 0
        out = capsys.readouterr().out
        assert "dry-run" in out
        assert "отправлено=0" in out

    def test_cli_reports_error_without_key(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr("qa.cli.load_env_file", lambda: None)
        monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
        monkeypatch.setenv("LLM_ENDPOINT", "http://127.0.0.1:1")
        monkeypatch.setenv("LLM_MODEL", _ALIAS)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.setattr(sys, "argv", ["qa-pipeline", "llm-ping"])
        assert main() == 1
        out = capsys.readouterr().out
        assert "LLM_API_KEY" in out

    def test_cli_dry_run_without_key_is_settings_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Валидация настроек требует ключ при не-stub провайдере и для dry-run."""
        monkeypatch.setattr("qa.cli.load_env_file", lambda: None)
        monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
        monkeypatch.setenv("LLM_ENDPOINT", "http://127.0.0.1:1")
        monkeypatch.setenv("LLM_MODEL", _ALIAS)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.setattr(sys, "argv", ["qa-pipeline", "llm-ping", "--dry-run"])
        assert main() == 1
        out = capsys.readouterr().out
        assert "LLM_API_KEY" in out


def test_real_llm_ping() -> None:
    """Проверка реального эндпоинта; запускается только по QA_REAL_LLM=1."""
    if os.environ.get("QA_REAL_LLM") != "1":
        pytest.skip("QA_REAL_LLM не задан")
    load_env_file()
    settings = Settings.load()
    outcome = ping_llm(settings)
    assert outcome.ok is True
    assert outcome.model_used
