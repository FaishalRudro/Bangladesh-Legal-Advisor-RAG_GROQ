import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    GEMINI_API_KEY: str

    # Models
    EMBEDDING_MODEL: str
    EMBEDDING_DIMENSIONS: int
    GENERATION_MODEL: str

    # Vector DB
    CHROMA_DB_PATH: str
    CHROMA_COLLECTION: str
    LAW_REGISTRY_COLLECTION: str

    # Chunking
    CHUNK_WORDS: int
    OVERLAP_WORDS: int

    # RAG
    TOP_K_RESULTS: int
    MIN_SCORE_THRESHOLD: float

    # App
    APP_HOST: str
    APP_PORT: int
    DATA_DIR: Path
    LOG_DIR: Path

    # Database
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int

    # Security
    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    OTP_EXPIRE_SECONDS: int
    FRONTEND_URL: str
    SUPER_ADMIN_NAME: str
    SUPER_ADMIN_EMAIL: str
    SUPER_ADMIN_PASSWORD: str

    # SMTP (Brevo)
    SMTP_HOST: str
    SMTP_PORT: int
    SMTP_USER: str
    SMTP_PASSWORD: str
    FROM_EMAIL: str

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Ensure directories exist
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.LOG_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_DB_PATH, exist_ok=True)
