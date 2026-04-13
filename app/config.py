from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    llm_provider: str = "openai"

    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: Optional[str] = None

    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "llama3.2"

    embedding_provider: str = "openai"
    openai_embedding_model: str = "text-embedding-3-small"
    ollama_embedding_model: str = "nomic-embed-text"

    data_dir: str = "./data"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
