from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Config(BaseSettings):
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8088, alias="PORT")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        case_sensitive=True,
        extra="ignore"
    )

config = Config()