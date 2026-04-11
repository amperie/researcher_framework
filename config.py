from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- LLM ---
    llm_provider: str = "anthropic"
    """Provider name: 'anthropic' or 'openai'."""

    llm_model: str | None = None
    """Model ID override. Defaults to claude-opus-4-6 (anthropic) or gpt-4o (openai)."""

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # --- ChromaDB (remote HTTP server) ---
    chroma_host: str = "hp.lan"
    """Hostname or IP of the ChromaDB HTTP server."""

    chroma_port: int = 8000
    """Port the ChromaDB HTTP server listens on."""

    chroma_ssl: bool = False
    """Use HTTPS when connecting to the ChromaDB server."""

    chroma_auth_token: str | None = None
    """Bearer token for ChromaDB server auth. Leave unset for open servers."""

    chroma_collection: str = "experiments"
    """ChromaDB collection name for experiment records."""

    # --- MLflow ---
    mlflow_uri: str = "http://localhost:5000"
    """MLflow tracking server URI. Use 'databricks' for Databricks-hosted tracking."""

    mlflow_experiment: str = "neuralsignal_researcher"
    """MLflow experiment name under which all runs are logged."""

    # --- Experiment execution ---
    neuralsignal_python: str = "uv run python"
    """Shell command used to launch generated experiment scripts in the neuralsignal venv."""

    experiment_timeout_seconds: int = 1800
    """Maximum wall-clock seconds allowed for a single experiment subprocess."""

    experiments_dir: str = "./experiments"
    """Directory where generated experiment scripts are written."""

    # --- NeuralSignal source path ---
    neuralsignal_src_path: str = "../neuralsignal/neuralsignal"
    """Path to the neuralsignal source tree. Injected into PYTHONPATH when
    launching experiment subprocesses so generated scripts can import neuralsignal."""

    # --- Arxiv ---
    max_arxiv_papers: int = 10
    """Maximum number of arxiv papers fetched per research query."""


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return the singleton Config instance (loaded from .env on first call)."""
    return Config()
