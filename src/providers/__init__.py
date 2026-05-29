# src/providers/__init__.py

"""
Model-provider abstraction for VectorFlow-RAG (Phase 12).

Importing this package registers all built-in providers (offline + online) so
the factory can resolve any of them by name. Public surface:

* :class:`ModelProvider` — provider base class
* :class:`ChatModelConfig` / :class:`EmbeddingModelConfig` / :class:`RerankerModelConfig`
* :class:`ProviderModel` / :class:`ProviderCapabilities` / :class:`ConnectionStatus`
* :func:`make_chat_provider` / :func:`list_provider_capabilities`
* :class:`SecretStore` / :func:`get_secret_store` / :func:`redact_secret`
"""

from src.providers.base import (
    ChatModelConfig,
    ConnectionStatus,
    EmbeddingModelConfig,
    ModelKind,
    ModelNotFoundError,
    ModelProvider,
    ProviderAuthError,
    ProviderCapabilities,
    ProviderError,
    ProviderLocation,
    ProviderModel,
    ProviderUnavailableError,
    RerankerModelConfig,
)
from src.providers.registry import (
    get_provider_capabilities,
    is_registered,
    list_provider_capabilities,
    make_chat_provider,
    register_provider,
)
from src.providers.secrets import (
    SecretStore,
    get_secret_store,
    redact_secret,
    reset_secret_store,
)

# Import provider modules for their import-time self-registration side effects.
from src.providers import ollama  # noqa: E402,F401  (registers "ollama")

__all__ = [
    "ChatModelConfig",
    "ConnectionStatus",
    "EmbeddingModelConfig",
    "ModelKind",
    "ModelNotFoundError",
    "ModelProvider",
    "ProviderAuthError",
    "ProviderCapabilities",
    "ProviderError",
    "ProviderLocation",
    "ProviderModel",
    "ProviderUnavailableError",
    "RerankerModelConfig",
    "SecretStore",
    "get_provider_capabilities",
    "get_secret_store",
    "is_registered",
    "list_provider_capabilities",
    "make_chat_provider",
    "redact_secret",
    "register_provider",
    "reset_secret_store",
]
