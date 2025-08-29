from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """
    Application settings.
    Loads configuration from a .env file.
    """
    # The model_config attribute replaces the old Config class
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # --- Database ---
    DATABASE_URL: str = "postgresql+psycopg2://user:password@localhost:5432/kost1k_trade_db"

    # --- Exchange API ---
    API_KEY: Optional[str] = None
    API_SECRET: Optional[str] = None


settings = Settings()
