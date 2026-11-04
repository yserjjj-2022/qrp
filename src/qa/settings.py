"""Minimal settings loader. Full implementation in Task 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class LlmSettings(BaseModel):
    provider: str = "stub"
    model: str | None = None
    endpoint: str | None = None
    max_calls_per_run: int = 100
    cache_dir: str = "data/cache"


class VectorsSettings(BaseModel):
    backend: str = "stub"
    model: str | None = None


class Settings(BaseModel):
    db_path: str = "data/interviews.db"
    data_dir: str = "data"
    log_level: str = "INFO"
    llm: LlmSettings = LlmSettings()
    vectors: VectorsSettings = VectorsSettings()

    @classmethod
    def load(cls, config_path: Path | None = None) -> Settings:
        if config_path is None:
            config_path = Path("config.yaml")
        data: dict[str, Any] = {}
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            if isinstance(raw, dict):
                data = raw
        return cls(**data)
