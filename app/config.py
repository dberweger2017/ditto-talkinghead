from __future__ import annotations

from functools import lru_cache
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    openai_api_key: str | None = Field(default=None, env="OPENAI_API_KEY")
    openai_realtime_model: str = Field(
        default="gpt-4o-realtime-preview-2024-12-17",
        env="OPENAI_REALTIME_MODEL",
    )
    session_timeout_seconds: int = Field(default=600, env="SESSION_TIMEOUT_SECONDS")
    max_sessions: int = Field(default=4, env="MAX_SESSIONS")
    ditto_cfg_pkl: str | None = Field(default=None, env="DITTO_CFG_PKL")
    ditto_data_root: str | None = Field(default=None, env="DITTO_DATA_ROOT")
    ditto_source_path: str | None = Field(default=None, env="DITTO_SOURCE_PATH")
    ditto_output_dir: str = Field(default="./tmp/avatar_outputs", env="DITTO_OUTPUT_DIR")
    ditto_frame_queue_size: int = Field(default=16, env="DITTO_FRAME_QUEUE_SIZE")
    ditto_chunksize: tuple[int, int, int] = Field(default=(3, 5, 2), env="DITTO_CHUNKSIZE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
