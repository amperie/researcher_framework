from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file="configs/.env", env_file_encoding="utf-8")

    # --- LLM ---
    llm_provider: str = "anthropic"
    """Provider name: 'anthropic' or 'openai'."""

    llm_model: str | None = None
    """Model ID override. Defaults to claude-opus-4-6 (anthropic) or gpt-4o (openai)."""

    @field_validator("llm_model", mode="before")
    @classmethod
    def _blank_comment_as_none(cls, v):
        """Treat empty strings or comment placeholders as None.

        python-dotenv does not strip inline comments, so a line like
            LLM_MODEL=# optional override
        is read as the literal string '# optional override...' rather than None.
        """
        if not v or str(v).startswith("#"):
            return None
        return v

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # --- MongoDB ---
    mongo_url: str = "mongodb://hp.lan:27017"
    """pymongo-compatible connection URI for the MongoDB instance that holds
    neuralsignal scan snapshots."""

    datasets_db_name: str = "neuralsignal_datasets"
    """MongoDB database name for the dataset registry and GridFS parquet storage."""

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

    experiments_dir: str = "dev/experiments"
    """Directory where generated experiment scripts are written (under dev/ so gitignored)."""

    # --- Code generation ---
    claude_command: str = "claude"
    """Shell command for the Claude Code CLI used by code_generation_node."""

    code_generation_timeout_seconds: int = 180
    """Maximum seconds to wait for the Claude Code CLI to generate a script."""

    # --- NeuralSignal source path ---
    neuralsignal_src_path: str = "../../neuralsignal/neuralsignal"
    """Path to the neuralsignal source tree. Injected into PYTHONPATH when
    launching experiment subprocesses so generated scripts can import neuralsignal."""

    # --- Arxiv ---
    max_arxiv_papers: int = 10
    """Maximum number of arxiv papers fetched per research query."""


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return the singleton Config instance (loaded from .env on first call)."""
    return Config()
