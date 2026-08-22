from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def configs_dir() -> Path:
    return repo_root() / "configs"


def load_yaml(name: str) -> dict:
    path = configs_dir() / name
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(repo_root() / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_port: int = 8000
    frontend_url: str = "http://localhost:5173"
    database_url: str = "sqlite:///backend/data/app.db"
    log_level: str = "INFO"
    rpc_url: str = "http://127.0.0.1:8545"
    data_provider_api_key: str = ""
    aave_pool: str = ""
    private_key: str = Field(default="", repr=False)


@lru_cache
def get_settings() -> Settings:
    return Settings()
