"""Inference provider package for Robin."""
from robin.brain.providers.base import (
    InferenceProvider,
    InferenceProviderError,
    ModelDescriptor,
    ModelNotFoundError,
    ProviderHealth,
    ProviderUnavailableError,
)
from robin.brain.providers.factory import (
    get_default_provider,
    get_provider,
    register_provider,
    reset_default_provider,
    set_default_provider,
)
from robin.brain.providers.ollama import OllamaProvider

__all__ = [
    "InferenceProvider",
    "InferenceProviderError",
    "ModelDescriptor",
    "ModelNotFoundError",
    "ProviderHealth",
    "ProviderUnavailableError",
    "OllamaProvider",
    "get_provider",
    "get_default_provider",
    "set_default_provider",
    "reset_default_provider",
    "register_provider",
]
