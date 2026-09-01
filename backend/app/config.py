"""Application settings, loaded from the environment."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "VeriDoc"
    env: str = "development"

    database_url: str = "postgresql+psycopg://veridoc:veridoc@db:5432/veridoc"
    redis_url: str = "redis://redis:6379/0"

    cors_origins: str = "http://localhost:5173"

    # Unused until Phases 2-3; declared here so the contract is visible early.
    forensics_model_path: str = "./ml/checkpoints/forensics_cnn.pt"
    face_match_threshold: float = 0.55
    liveness_threshold: float = 0.5

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
