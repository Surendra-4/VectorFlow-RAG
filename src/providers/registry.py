# src/providers/registry.py

"""
Provider registry + chat-provider factory.

Providers self-register at import time (see the ``register_provider`` call at
the bottom of each provider module). The registry is the single place the API
and the runtime config consult to:

* enumerate provider capabilities for the frontend (``list_provider_capabilities``)
* construct a live provider from a :class:`ChatModelConfig`, injecting the
  API key from the backend ``SecretStore`` only when the provider needs one
  (``make_chat_provider``).

Keeping construction here means call sites never branch on provider name.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type

from src.providers.base import (
    ChatModelConfig,
    ModelProvider,
    ProviderCapabilities,
    ProviderError,
)

if TYPE_CHECKING:
    from src.providers.secrets import SecretStore

_PROVIDERS: Dict[str, Tuple[Type[ModelProvider], ProviderCapabilities]] = {}
_LOCK = threading.Lock()


def register_provider(capabilities: ProviderCapabilities, cls: Type[ModelProvider]) -> None:
    """Register a provider class under its canonical capability name."""
    with _LOCK:
        _PROVIDERS[capabilities.name] = (cls, capabilities)


def is_registered(name: str) -> bool:
    return name in _PROVIDERS


def get_provider_class(name: str) -> Type[ModelProvider]:
    entry = _PROVIDERS.get(name)
    if entry is None:
        raise ProviderError(f"Unknown provider: {name!r}", provider=name)
    return entry[0]


def get_provider_capabilities(name: str) -> ProviderCapabilities:
    entry = _PROVIDERS.get(name)
    if entry is None:
        raise ProviderError(f"Unknown provider: {name!r}", provider=name)
    return entry[1]


def list_provider_capabilities() -> List[ProviderCapabilities]:
    """All registered providers' capabilities, sorted offline-first then name."""
    with _LOCK:
        caps = [c for _, c in _PROVIDERS.values()]
    return sorted(caps, key=lambda c: (c.location.value != "offline", c.name))


def make_chat_provider(
    config: ChatModelConfig,
    secret_store: Optional["SecretStore"] = None,
) -> ModelProvider:
    """Construct a live provider for ``config``.

    Injects the API key from ``secret_store`` iff the provider requires one.
    The key is passed to the constructor and never stored on ``config``.
    """
    cls = get_provider_class(config.provider)
    caps = get_provider_capabilities(config.provider)

    api_key = None
    if caps.requires_api_key:
        if secret_store is None:
            from src.providers.secrets import get_secret_store

            secret_store = get_secret_store()
        api_key = secret_store.get_secret(config.provider)

    return cls(config, api_key=api_key)
