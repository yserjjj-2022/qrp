"""Модуль настроек: валидация, загрузка из config.yaml + .env."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

ColumnType = Literal["str", "text", "int", "date", "list", "notes"]
ListKind = Literal["code", "question", "plain"]


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


class PathsSettings(BaseModel):
    """Пути проекта. Все значения относительны корня репозитория."""

    data: str = "data"
    inputs: str = "data/inputs"
    transcripts: str = "data/inputs/transcripts"
    workbook: str = "data/inputs/Rabochaya_kniga_proekta_v4.xlsx"
    db: str = "data/db/interviews.db"
    backups: str = "data/backups"
    cache: str = "data/cache"
    exports: str = "data/exports"
    golden: str = "data/golden"


class ColumnSpec(BaseModel):
    """Описание одной колонки листа книги."""

    header: str
    type: ColumnType = "str"
    required: bool = False
    list_kind: ListKind = "plain"
    strict: bool = False


class SheetSpec(BaseModel):
    """Схема листа: логическое имя поля -> колонка."""

    columns: dict[str, ColumnSpec] = Field(default_factory=dict)


class WorkbookSettings(BaseModel):
    """Классы листов и схемы колонок рабочей книги."""

    required_sheets: list[str] = Field(default_factory=list)
    optional_sheets: list[str] = Field(default_factory=list)
    optional_prefixes: list[str] = Field(default_factory=list)
    ignored_sheets: list[str] = Field(default_factory=list)
    ignored_prefixes: list[str] = Field(default_factory=list)
    codebook_sheets: list[str] = Field(default_factory=list)
    family_nodes: dict[str, str] = Field(default_factory=dict)
    defaults: dict[str, list[str]] = Field(default_factory=dict)
    sheets: dict[str, SheetSpec] = Field(default_factory=dict)
    prefix_sheets: dict[str, SheetSpec] = Field(default_factory=dict)

    def sheet_class(self, name: str) -> Literal["required", "optional", "ignored", "unknown"]:
        """Класс листа по имени: обязательный, необязательный, игнорируемый, неизвестный."""
        if name in self.required_sheets:
            return "required"
        if name in self.optional_sheets or any(name.startswith(p) for p in self.optional_prefixes):
            return "optional"
        if name in self.ignored_sheets or any(name.startswith(p) for p in self.ignored_prefixes):
            return "ignored"
        return "unknown"

    def spec_for(self, name: str) -> SheetSpec | None:
        """Схема колонок для листа (точное имя или префикс)."""
        if name in self.sheets:
            return self.sheets[name]
        for prefix, spec in self.prefix_sheets.items():
            if name.startswith(prefix):
                return spec
        return None


class Settings(BaseModel):
    """Глобальные настройки проекта."""

    paths: PathsSettings = PathsSettings()
    llm: LlmSettings = LlmSettings()
    log_level: str = "INFO"
    vectors: VectorsSettings = VectorsSettings()
    pipeline: PipelineSettings = PipelineSettings()
    m1b: M1bSettings = M1bSettings()
    workbook: WorkbookSettings = WorkbookSettings()

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_paths(cls, data: Any) -> Any:
        """Перенести устаревшие db_path/data_dir в paths."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        paths = dict(data.get("paths") or {})
        if "db_path" in data and "db" not in paths:
            paths["db"] = data.pop("db_path")
        else:
            data.pop("db_path", None)
        if "data_dir" in data and "data" not in paths:
            paths["data"] = data.pop("data_dir")
        else:
            data.pop("data_dir", None)
        data["paths"] = paths
        return data

    @property
    def db_path(self) -> str:
        """Путь к файлу БД (строка для совместимости)."""
        return self.paths.db

    @property
    def data_dir(self) -> str:
        """Корневой каталог данных (строка для совместимости)."""
        return self.paths.data

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

    def ensure_dirs(self) -> None:
        """Создать каталоги данных, если их нет."""
        for value in (
            self.paths.data,
            self.paths.inputs,
            self.paths.transcripts,
            self.paths.backups,
            self.paths.cache,
            self.paths.exports,
            self.paths.golden,
            str(Path(self.db_path).parent),
        ):
            Path(value).mkdir(parents=True, exist_ok=True)

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
