from __future__ import annotations

from functools import lru_cache
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    openai_api_key: str | None = Field(default=None, env="OPENAI_API_KEY")
    openai_realtime_model: str = Field(default="gpt-realtime", env="OPENAI_REALTIME_MODEL")
    session_timeout_seconds: int = Field(default=600, env="SESSION_TIMEOUT_SECONDS")
    max_sessions: int = Field(default=4, env="MAX_SESSIONS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
