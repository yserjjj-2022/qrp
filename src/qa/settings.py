"""Модуль настроек: валидация, загрузка из config.yaml + .env."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from qa.pipeline.normalize_rules import FuzzyConfig

ColumnType = Literal["str", "text", "int", "date", "list", "notes"]
ListKind = Literal["code", "question", "plain"]

LLM_PROVIDERS: tuple[str, ...] = ("stub", "openai_compatible")

LLM_ENV_FIELDS: dict[str, str] = {
    "LLM_PROVIDER": "provider",
    "LLM_ENDPOINT": "endpoint",
    "LLM_MODEL": "model",
    "LLM_MODEL_NORMALIZE": "model_normalize",
    "LLM_CACHE_DIR": "cache_dir",
    "LLM_MAX_CALLS_PER_RUN": "max_calls_per_run",
}


class LlmSettings(BaseModel):
    """Настройки LLM-провайдера.

    Секрет LLM_API_KEY здесь не хранится: он читается из окружения через
    `Settings.llm_api_key()` и передаётся провайдеру напрямую.
    """

    provider: str = "stub"
    model: str | None = None
    model_normalize: str | None = None
    endpoint: str | None = None
    max_calls_per_run: int = Field(default=100, ge=1)
    cache_dir: str = "data/cache"

    @model_validator(mode="before")
    @classmethod
    def _apply_env(cls, data: Any) -> Any:  # noqa: ANN401
        """Перекрыть значения из config.yaml переменными окружения.

        Переменные окружения приоритетнее файла конфигурации: значения из
        `.env` задаёт оператор для конкретного запуска.
        """
        merged: dict[str, Any] = dict(data) if isinstance(data, dict) else {}
        for env_name, field_name in LLM_ENV_FIELDS.items():
            raw = os.environ.get(env_name)
            if raw is None or not raw.strip():
                continue
            merged[field_name] = raw.strip()
        return merged

    @field_validator("provider")
    @classmethod
    def _check_provider(cls, v: str) -> str:
        """Допустимые значения LLM_PROVIDER."""
        value = v.strip()
        if value not in LLM_PROVIDERS:
            msg = f"Недопустимый LLM_PROVIDER: {value!r}. Допустимы: {', '.join(LLM_PROVIDERS)}"
            raise ValueError(msg)
        return value


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
    """Настройки нормализации L2 (M1b).

    `min_words_for_model` — минимальная длина реплики респондента для вызова
    модели. `fuzzy_lexicon_threshold` и `fuzzy_lexicon_short_threshold` задают
    нечёткое сопоставление с лексиконом в детерминированном проходе: для слов
    короче `fuzzy_short_word_length` действует более строгий порог.
    Остальные ключи оставлены для отложенных частей M1b (золотой набор,
    проверка отношения длин, review-выборка) и сейчас не используются.
    """

    min_words_for_model: int = Field(default=4, ge=1)
    fuzzy_lexicon_threshold: float = Field(default=0.80, gt=0, le=1)
    fuzzy_lexicon_short_threshold: float = Field(default=0.93, gt=0, le=1)
    fuzzy_short_word_length: int = Field(default=5, ge=2)
    filler_density_threshold: float = Field(default=0.05, gt=0, le=1)
    len_ratio_min: float = Field(default=0.6, gt=0)
    len_ratio_max: float = Field(default=1.1, gt=0)
    review_sample_share: float = Field(default=0.10, ge=0, le=1)

    def fuzzy_config(self) -> FuzzyConfig:
        """Пороги нечёткого сопоставления для детерминированного прохода."""
        return FuzzyConfig(
            threshold=self.fuzzy_lexicon_threshold,
            short_threshold=self.fuzzy_lexicon_short_threshold,
            short_word_length=self.fuzzy_short_word_length,
        )


class GroupsSettings(BaseModel):
    """Канонические значения групп и их алиасы.

    Сравнение идёт по casefold: «ТОРГ», «торг», «Торгующие» дают одно значение.
    """

    canonical: list[str] = Field(default_factory=lambda: ["Все", "ТОРГ", "Не торгующие"])
    aliases: dict[str, list[str]] = Field(default_factory=dict)

    def lookup(self) -> dict[str, str]:
        """Карта «нормализованный алиас -> каноническое значение»."""
        table: dict[str, str] = {}
        for canonical in self.canonical:
            key = _normalize_key(canonical)
            self._put(table, key, canonical)
            for alias in self.aliases.get(canonical, []):
                self._put(table, _normalize_key(alias), canonical)
        return table

    @staticmethod
    def _put(table: dict[str, str], key: str, canonical: str) -> None:
        """Записать алиас; конфликт двух канонических значений — ошибка конфига."""
        existing = table.get(key)
        if existing is not None and existing != canonical:
            msg = f"Алиас группы {key!r} указан и для {existing!r}, и для {canonical!r}"
            raise ValueError(msg)
        table[key] = canonical


def _normalize_key(value: str) -> str:
    """Ключ сравнения: strip + casefold + схлопывание пробелов и подчёркиваний."""
    text = " ".join(value.replace("_", " ").split())
    return text.casefold()


def normalize_group(value: str, groups: GroupsSettings | None = None) -> str:
    """Привести группу к каноническому значению.

    Неизвестное значение не ошибка: возвращается как есть (после strip),
    вызывающая сторона выводит предупреждение.
    """
    settings = groups or GroupsSettings()
    canonical, _recognized = resolve_group(value, settings)
    return canonical


def resolve_group(value: str, groups: GroupsSettings) -> tuple[str, bool]:
    """Каноническая группа и признак «значение распознано»."""
    canonical = groups.lookup().get(_normalize_key(value))
    return (canonical, True) if canonical is not None else (value.strip(), False)


def normalize_lexicon_status(value: str | None, aliases: dict[str, str]) -> tuple[str, bool]:
    """Привести статус лексикона к proposed/approved/rejected.

    Returns:
        (статус, распознан ли). Неизвестное значение даёт 'proposed' и False.
    """
    text = (value or "").strip()
    if not text:
        return "proposed", True
    key = _normalize_key(text)
    for source, target in aliases.items():
        if _normalize_key(source) == key:
            return target, True
    if key in ("proposed", "approved", "rejected"):
        return key, True
    return "proposed", False


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
    groups: GroupsSettings = GroupsSettings()
    lexicon_status_map: dict[str, str] = Field(default_factory=dict)
    # Строка предметной области для промпта нормализации. В репозитории пустая:
    # текст содержит названия брокеров и продуктов и не должен публиковаться.
    # Задаётся переменной окружения QA_DOMAIN_HINT или локальным
    # `config.local.yaml` (в .gitignore). Сведений в L2 не добавляет: только
    # помогает отличать термины от искажений ASR.
    domain_hint: str = ""

    @model_validator(mode="before")
    @classmethod
    def _apply_domain_hint_env(cls, data: Any) -> Any:  # noqa: ANN401
        """Перекрыть domain_hint переменной окружения QA_DOMAIN_HINT.

        Переменная приоритетнее локального файла: значение задаёт оператор для
        конкретного запуска. Пустая переменная не затирает файл.
        """
        merged: dict[str, Any] = dict(data) if isinstance(data, dict) else {}
        raw = os.environ.get("QA_DOMAIN_HINT")
        if raw is not None and raw.strip():
            merged["domain_hint"] = raw.strip()
        return merged

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

    def llm_api_key(self) -> str | None:
        """Ключ LLM из окружения. В модели настроек не хранится."""
        value = os.environ.get("LLM_API_KEY")
        return value.strip() if value and value.strip() else None

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
        """Загрузить настройки из config.yaml + локального config.local.yaml + окружения.

        `config.local.yaml` не коммитится: в нём лежат значения, которых не
        должно быть в публичном репозитории (`domain_hint` с названиями
        брокеров и продуктов). Значения из локального файла перекрывают
        `config.yaml` поверхностно, по верхним ключам.

        `.env` читается вызывающей стороной (`load_env_file`) на входе CLI:
        библиотечный код не меняет окружение процесса неявно.
        """
        if config_path is None:
            config_path = Path("config.yaml")
        data = _read_yaml(config_path)
        # Локальный файл рядом с конфигом: значения, которых не должно быть в
        # публичном репозитории. Отсутствие файла — норма, не ошибка.
        data.update(_read_yaml(config_path.with_name("config.local.yaml")))
        return cls(**data)


def _read_yaml(path: Path) -> dict[str, Any]:
    """Прочитать YAML-файл в словарь; отсутствие файла и не-словарь дают пусто."""
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw if isinstance(raw, dict) else {}


def load_env_file(env_path: Path | None = None) -> None:
    """Прочитать .env и дополнить os.environ.

    Значения, уже заданные в окружении, не перезаписываются. Секреты не
    логируются. Формат: KEY=VALUE, строки с # и пустые игнорируются.
    """
    path = env_path if env_path is not None else Path(".env")
    if not path.is_file():
        return
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, raw_value = stripped.partition("=")
        name = name.strip()
        if name.startswith("export "):
            name = name[len("export ") :].strip()
        if not name or name in os.environ:
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ[name] = value
