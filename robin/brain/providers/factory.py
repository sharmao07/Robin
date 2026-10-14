"""Factory and registry for inference providers."""
import os
from typing import Dict, Optional, Type

from robin.brain.providers.base import (
    InferenceProvider,
    ProviderUnavailableError,
)
from robin.brain.providers.ollama import OllamaProvider

_PROVIDERS: Dict[str, Type[InferenceProvider]] = {
    "ollama": OllamaProvider,
}

_DEFAULT_PROVIDER_INSTANCE: Optional[InferenceProvider] = None


def register_provider(name: str, provider_cls: Type[InferenceProvider]) -> None:
    """Register a new inference provider implementation."""
    _PROVIDERS[name.lower()] = provider_cls


def get_provider(
    name: Optional[str] = None,
    **kwargs,
) -> InferenceProvider:
    """Instantiate or retrieve an inference provider.

    Resolution order:
    1. Explicit 'name' argument.
    2. 'ROBIN_INFERENCE_PROVIDER' environment variable.
    3. Default: 'ollama' (during Milestone 1 migration).
    """
    provider_name = (name or os.environ.get("ROBIN_INFERENCE_PROVIDER") or "ollama").strip().lower()

    if provider_name == "embedded":
        raise ProviderUnavailableError(
            "Embedded llama.cpp provider is not yet implemented (scheduled for Milestone 3). "
            "Use the 'ollama' development provider during Milestone 1."
        )

    provider_cls = _PROVIDERS.get(provider_name)
    if provider_cls is None:
        available = ", ".join(sorted(_PROVIDERS.keys()))
        raise ProviderUnavailableError(
            f"Unknown inference provider '{provider_name}'. Available providers: {available}"
        )

    return provider_cls(**kwargs)


def get_default_provider() -> InferenceProvider:
    """Return a shared singleton provider instance."""
    global _DEFAULT_PROVIDER_INSTANCE
    if _DEFAULT_PROVIDER_INSTANCE is None:
        _DEFAULT_PROVIDER_INSTANCE = get_provider()
    return _DEFAULT_PROVIDER_INSTANCE


def set_default_provider(provider: Optional[InferenceProvider]) -> None:
    """Override or set the shared default provider instance (useful in testing)."""
    global _DEFAULT_PROVIDER_INSTANCE
    _DEFAULT_PROVIDER_INSTANCE = provider


def reset_default_provider() -> None:
    """Reset the shared default provider instance, shutting it down if present."""
    global _DEFAULT_PROVIDER_INSTANCE
    if _DEFAULT_PROVIDER_INSTANCE is not None:
        try:
            _DEFAULT_PROVIDER_INSTANCE.shutdown()
        except Exception:
            pass
        _DEFAULT_PROVIDER_INSTANCE = None
