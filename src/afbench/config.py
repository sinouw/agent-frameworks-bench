from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Every field can be set from the environment or `.env`."""

    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # One OpenAI-compatible endpoint serves all three implementations (Groq by default).
    llm_base_url: str = "https://api.groq.com/openai/v1"
    groq_api_key: SecretStr | None = None
    answer_model: str = "openai/gpt-oss-20b"
    # The judge is a different model family from the answering model.
    judge_model: str = "qwen/qwen3.8-27b"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    top_k: int = 5
    candidates: int = 20  # per-retriever candidate pool before fusion, where the implementation exposes it
    max_steps: int = 5  # LLM turns per question

    corpus_dir: Path = REPO_ROOT / "data" / "corpus"
    index_dir: Path = REPO_ROOT / "data" / "index"
    sqlite_path: Path = REPO_ROOT / "data" / "metadata.db"
    golden_path: Path = REPO_ROOT / "evals" / "golden.jsonl"
    results_dir: Path = REPO_ROOT / "results"

    def api_key(self) -> str:
        if self.groq_api_key is None:
            raise RuntimeError("GROQ_API_KEY is not set (put it in .env).")
        return self.groq_api_key.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    return Settings()
