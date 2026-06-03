# src/config.py

"""
Centralized, environment-driven configuration for VectorFlow-RAG.

All defaults can be overridden via:
- Environment variables (prefix ``VFR_``, nested delimiter ``__``)
- A ``.env`` file at the project root (auto-loaded if present)
- Programmatic override: ``Settings(embedder=EmbedderSettings(...))``

Example::

    from src.config import get_settings
    settings = get_settings()
    embedder_model = settings.embedder.model_name

Override examples::

    VFR_EMBEDDER__MODEL_NAME=BAAI/bge-large-en
    VFR_LLM__MODEL=llama3.2:1b
    VFR_VECTOR_STORE__PERSIST_DIRECTORY=/data/vfr/chroma
    VFR_LOGGING__LEVEL=DEBUG
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root resolves to the directory containing src/ and tests/.
# This works regardless of whether the package is run from the project root,
# imported from a notebook, or executed inside a worktree.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_DIR: Path = PROJECT_ROOT / "indices"


def _to_path(value) -> Path:
    """
    Coerce ``value`` (str/Path/None) to a normalized ``Path``.

    Accepts both POSIX and Windows-style separators so that legacy strings
    like ``"indices\\chroma_db"`` still resolve correctly on macOS/Linux.
    """
    if value is None:
        return value  # type: ignore[return-value]
    if isinstance(value, Path):
        return value
    return Path(str(value).replace("\\", "/"))


# --------------------------------------------------------------------------- #
# Section models
# --------------------------------------------------------------------------- #


class AppSettings(BaseModel):
    """Top-level application metadata."""

    name: str = "VectorFlow-RAG"
    version: str = "0.2.0-foundation"
    project_root: Path = PROJECT_ROOT

    @field_validator("project_root", mode="before")
    @classmethod
    def _normalize(cls, v):
        return _to_path(v)


# English-baseline model defaults — used by the profile resolver to detect
# whether a field is still at its English default (and therefore safe to
# override) versus explicitly set by the user (which always wins).
ENGLISH_EMBEDDER_MODEL = "all-MiniLM-L6-v2"
ENGLISH_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class EmbedderSettings(BaseModel):
    """Embedding model configuration."""

    model_name: str = ENGLISH_EMBEDDER_MODEL
    # ``device=None`` triggers auto-detection (cuda → mps → cpu) at runtime.
    device: Optional[str] = None
    batch_size: int = 32
    normalize: bool = True

    # Instruction prefixes for asymmetric models (e.g. multilingual-e5 needs
    # "query: " / "passage: "). ``None`` means *auto-detect* from the model
    # name inside the embedder adapter — so model-specific behavior stays
    # encapsulated there and no caller needs to know which model is active.
    # ``""`` means *explicitly no prefix*.
    query_prefix: Optional[str] = None
    passage_prefix: Optional[str] = None


class ChunkerSettings(BaseModel):
    chunk_size: int = 500
    overlap: int = 50


class VectorStoreSettings(BaseModel):
    """
    Vector-store configuration.

    The ``backend`` field is forward-compatible: today only ``chromadb`` is
    wired up, but the FAISS phase will plug in here without any further
    schema changes.
    """

    backend: Literal["chromadb", "faiss"] = "chromadb"
    persist_directory: Path = DEFAULT_INDEX_DIR / "chroma_db"
    collection_name: str = "vectorflow_docs"

    # FAISS knobs (no-op until FAISS backend lands).
    faiss_index_type: Literal["hnsw", "flat", "ivf"] = "hnsw"
    faiss_hnsw_m: int = 32
    faiss_hnsw_ef_construction: int = 200
    faiss_hnsw_ef_search: int = 64

    @field_validator("persist_directory", mode="before")
    @classmethod
    def _normalize(cls, v):
        return _to_path(v)


class BM25Settings(BaseModel):
    enabled: bool = True
    language: str = "english"  # passed through to the stemmer when use_stemmer
    # When False, BM25 tokenizes without a language-specific stemmer
    # (Unicode-aware whitespace). Multilingual profiles set this False to
    # avoid applying English morphology rules to non-English tokens.
    use_stemmer: bool = True


class LLMSettings(BaseModel):
    model: str = "tinyllama"
    base_url: str = "http://localhost:11434"
    max_tokens: int = 512
    temperature: float = 0.7
    request_timeout_s: int = 120


class RetrievalSettings(BaseModel):
    """
    Retrieval-time tunables.

    ``alpha`` is retained for the existing alpha-weighted fusion. The
    ``rrf_k`` and ``candidates_per_modality`` fields are placeholders for the
    upcoming RRF rollout — they are unused today.
    """

    k_default: int = 5
    alpha: float = 0.5
    candidates_per_modality: int = 10
    rrf_k: int = 60


class CacheSettings(BaseModel):
    """Cache settings (Phase 6)."""

    backend: Literal["none", "memory", "redis"] = "none"
    redis_url: str = "redis://localhost:6379/0"

    # Default TTL applied when a wrapper doesn't pass one explicitly.
    ttl_seconds: int = 3600

    # Per-namespace TTL overrides (seconds). ``None`` falls back to ``ttl_seconds``.
    # Embeddings are content-keyed and never go stale — long TTL is safe.
    ttl_embedding_s: Optional[int] = 24 * 3600
    # Expansion outputs are LLM-derived — long TTL keeps repeat-query UX consistent.
    ttl_expansion_s: Optional[int] = 24 * 3600
    # Full retrieval results are invalidated by corpus fingerprint, so even
    # large TTLs are safe; we cap at the default for predictable memory growth.
    ttl_retrieval_s: Optional[int] = 3600


class RerankerSettings(BaseModel):
    """Cross-encoder reranker settings — disabled by default."""

    enabled: bool = False
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 3
    device: Optional[str] = None


class LoggingSettings(BaseModel):
    level: str = "INFO"
    format: Literal["text", "json"] = "text"
    file: Optional[Path] = None
    use_color: bool = True

    @field_validator("file", mode="before")
    @classmethod
    def _normalize(cls, v):
        if v in (None, ""):
            return None
        return _to_path(v)

    @field_validator("level", mode="before")
    @classmethod
    def _upper(cls, v):
        return str(v).upper() if v is not None else v


class QueryExpansionSettings(BaseModel):
    """Query expansion knobs (Phase 5).

    Disabled by default — enabling adds 1–7 s of latency per query when
    running on tinyllama CPU. Phase 6 caching will compress this on
    repeat queries.
    """

    enabled: bool = False
    # Strategies executed in order. Only ``multi_query`` and ``hyde`` are
    # supported by Phase 5; future strategies plug in here.
    strategies: List[Literal["multi_query", "hyde"]] = Field(
        default_factory=lambda: ["multi_query"]
    )
    multi_query_count: int = 3
    hyde_count: int = 1
    # Per-strategy wall-clock budget. Strategy returns empty + error on timeout
    # so retrieval still proceeds with the original query.
    timeout_s: float = 10.0
    # Cap on the user query length sent to the expansion LLM (prompt-injection
    # mitigation — adversarial queries can't grow expansion prompts unbounded).
    max_query_length: int = 500
    # Cap on individual expanded variant / HyDE doc length.
    max_variant_length: int = 1500
    # Override the LLM model for expansion. ``None`` → use settings.llm.model.
    expansion_model: Optional[str] = None
    # When True, detect the query language and hint the LLM to generate
    # same-language variants. Off by default; advisory only — affects only
    # the generation prompt, never retrieval routing.
    language_hint: bool = False


class IngestionSettings(BaseModel):
    """File ingestion knobs (Phase 4 + Phase 11)."""

    # 100 MB default — large enough for typical PDFs/XLSX, small enough
    # to fail-fast on accidental ingestion of 1 GB binaries.
    max_file_size_bytes: int = 100 * 1024 * 1024
    # If True, ingest_files() raises on the first failing file; if False
    # (default), it collects failures and continues with the rest.
    fail_fast: bool = False

    # Tesseract OCR language(s), e.g. "eng", "fra", "eng+fra". Language data
    # files (tessdata) are operator-installed; we never auto-download them.
    ocr_lang: str = "eng"
    # When True, tag each ingested chunk with an advisory ``language`` field
    # (via langid). Off by default — adds per-chunk cost and is purely
    # advisory metadata; it NEVER drives retrieval routing.
    detect_language: bool = False


class APISettings(BaseModel):
    """FastAPI server settings — used once the API layer lands."""

    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])


class DatabaseSettings(BaseModel):
    """Relational DB for users + per-user stats (Phase 14).

    ``url`` is a SQLAlchemy URL. Defaults to a local SQLite file so the app
    works with zero setup; production sets ``DATABASE_URL`` (Render/Heroku
    style, ``postgres://…``) which the engine normalizes to ``postgresql+psycopg2``.
    Ingested documents are NEVER stored here — only accounts + statistics.
    """

    url: str = "sqlite:///./var/app.db"
    echo: bool = False


class AuthSettings(BaseModel):
    """Authentication + multi-user settings (Phase 14)."""

    # When False (default, local/tests), data endpoints stay open and stats are
    # attributed to a user only when a token is present. When True (production),
    # the data endpoints require a valid token.
    required: bool = False

    # JWT signing. PRODUCTION MUST override jwt_secret with a long random value
    # (e.g. `python -c "import secrets;print(secrets.token_urlsafe(48))"`).
    jwt_secret: str = "dev-insecure-change-me-in-production-this-is-not-a-secret"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60 * 24 * 7  # 7 days

    # Where this backend is reachable publicly (used to build OAuth callback
    # URLs) and where to send the user after a successful OAuth login.
    public_base_url: str = "http://localhost:8000"
    frontend_url: str = "http://localhost:3000"

    # OAuth app credentials — operator-provided (None disables that provider).
    google_client_id: Optional[str] = None
    google_client_secret: Optional[str] = None
    github_client_id: Optional[str] = None
    github_client_secret: Optional[str] = None

    # Password-reset email delivery (optional). Without SMTP configured, the
    # reset link is logged for local/dev use instead of emailed.
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_from: Optional[str] = None
    reset_token_expiry_minutes: int = 60


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #

# Profile presets map a profile name to the model choices it implies.
# The resolver applies these ONLY to fields still at their English default —
# any explicit env/kwarg override always wins. Prefix handling is delegated
# to the embedder adapter (auto-detected from the model name), so presets
# carry model names + BM25 stemmer policy, nothing model-internal.
_PROFILE_PRESETS: dict = {
    "english": {},  # all defaults — the validated baseline
    "multilingual": {
        "embedder_model": "intfloat/multilingual-e5-small",
        "reranker_model": "jinaai/jina-reranker-v2-base-multilingual",
        "bm25_use_stemmer": False,
    },
    "multilingual_quality": {
        "embedder_model": "BAAI/bge-m3",
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "bm25_use_stemmer": False,
    },
}


# --------------------------------------------------------------------------- #
# Top-level Settings
# --------------------------------------------------------------------------- #


class Settings(BaseSettings):
    """Composite settings root — loaded from env, ``.env``, or defaults."""

    model_config = SettingsConfigDict(
        env_prefix="VFR_",
        env_nested_delimiter="__",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Capability profile. ``english`` (default) is the extensively-validated
    # baseline and is never changed by the resolver. Other profiles opt into
    # multilingual model choices — but only for fields the user hasn't
    # explicitly overridden.
    profile: Literal["english", "multilingual", "multilingual_quality"] = "english"

    app: AppSettings = Field(default_factory=AppSettings)
    embedder: EmbedderSettings = Field(default_factory=EmbedderSettings)
    chunker: ChunkerSettings = Field(default_factory=ChunkerSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    bm25: BM25Settings = Field(default_factory=BM25Settings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    api: APISettings = Field(default_factory=APISettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    query_expansion: QueryExpansionSettings = Field(default_factory=QueryExpansionSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)

    @model_validator(mode="after")
    def _apply_profile(self):
        """
        Apply profile presets to fields still at their English default.

        Explicit overrides always win: if the embedder model differs from the
        English default, the user set it deliberately and we leave it. This
        keeps the English baseline byte-for-byte intact under the default
        profile while letting ``profile=multilingual`` swap models in one knob.
        """
        preset = _PROFILE_PRESETS.get(self.profile, {})
        if not preset:
            return self  # english — no changes

        if self.embedder.model_name == ENGLISH_EMBEDDER_MODEL:
            self.embedder.model_name = preset["embedder_model"]
        if self.reranker.model_name == ENGLISH_RERANKER_MODEL:
            self.reranker.model_name = preset["reranker_model"]
        # Only flip the stemmer if it's still at the English default config
        # (use_stemmer=True AND language=english). A user who set a specific
        # language clearly wants stemming for a known monolingual corpus.
        if self.bm25.use_stemmer and self.bm25.language == "english":
            self.bm25.use_stemmer = preset["bm25_use_stemmer"]
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached ``Settings`` singleton."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the cached singleton — primarily for tests that mutate env."""
    get_settings.cache_clear()
