from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Literal

class Config(BaseSettings):
    app_name: str = "AI Agent API"
    app_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow"
    )

config = Config()