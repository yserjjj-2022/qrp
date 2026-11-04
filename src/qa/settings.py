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
    model_normalize: str | None = None
    endpoint: str | None = None
    max_calls_per_run: int = Field(default=100, ge=1)
    cache_dir: str = "data/cache"


class VectorsSettings(BaseModel):
    """Настройки векторного поиска."""

    backend: str = "stub"
    model: str | None = None


class PipelineSettings(BaseModel):
    """Настройки сборки реплик (M1a)."""

    backchannel_max_words: int = Field(default=3, ge=1)
    backchannel_words: list[str] = Field(
        default_factory=lambda: ["угу", "ага", "да", "так", "ну", "ясно", "хорошо", "понятно", "мгм"]
    )


class M1bSettings(BaseModel):
    """Стартовые пороги нормализации L2 (M1b). Уточняются на золотом наборе."""

    filler_density_threshold: float = Field(default=0.05, gt=0, le=1)
    len_ratio_min: float = Field(default=0.6, gt=0)
    len_ratio_max: float = Field(default=1.1, gt=0)
    review_sample_share: float = Field(default=0.10, ge=0, le=1)


class Settings(BaseModel):
    """Глобальные настройки проекта."""

    db_path: str
    llm: LlmSettings = LlmSettings()
    data_dir: str = "data"
    log_level: str = "INFO"
    vectors: VectorsSettings = VectorsSettings()
    pipeline: PipelineSettings = PipelineSettings()
    m1b: M1bSettings = M1bSettings()

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
