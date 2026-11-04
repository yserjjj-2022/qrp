"""Модуль настроек: валидация, загрузка из config.yaml + .env."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class LlmSettings(BaseModel):
    """Настройки LLM-провайдера."""

    provider: str = "stub"
    model: str | None = None
    endpoint: str | None = None
    max_calls_per_run: int = Field(default=100, ge=1)
    cache_dir: str = "data/cache"


class VectorsSettings(BaseModel):
    """Настройки векторного поиска."""

    backend: str = "stub"
    model: str | None = None


class Settings(BaseModel):
    """Глобальные настройки проекта."""

    db_path: str
    llm: LlmSettings = LlmSettings()
    data_dir: str = "data"
    log_level: str = "INFO"
    vectors: VectorsSettings = VectorsSettings()

    @field_validator("db_path")
    @classmethod
    def _check_db_path(cls, v: str) -> str:
        p = Path(v)
        if not p.is_absolute():
            p = Path.cwd() / p
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)

    @field_validator("log_level")
    @classmethod
    def _check_log_level(cls, v: str) -> str:
        level = v.upper()
        if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            msg = f"Недопустимый level: {v!r}. Допустимы: DEBUG, INFO, WARNING, ERROR, CRITICAL"
            raise ValueError(msg)
        return level

    @field_validator("llm")
    @classmethod
    def _check_llm_api_key(cls, v: LlmSettings) -> LlmSettings:
        """LLM_API_KEY обязателен, если provider != 'stub'."""
        if v.provider != "stub":
            api_key = os.environ.get("LLM_API_KEY")
            if not api_key:
                msg = (
                    "LLM_API_KEY обязателен, когда llm.provider != 'stub'. Установите переменную окружения LLM_API_KEY"
                )
                raise ValueError(msg)
        return v

    @classmethod
    def load(cls, config_path: Path | None = None) -> Settings:
        """Загрузить настройки из config.yaml + переменных окружения."""
        data: dict[str, Any] = {}
        if config_path is None:
            config_path = Path("config.yaml")
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            if isinstance(raw, dict):
                data = raw
        return cls(**data)
