"""Inference provider abstraction for Robin.

Defines the core contract, data types, and exceptions for all local embedded
engines and development adapters.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import threading
from typing import Any, Dict, Generator, List, Optional, Tuple


class InferenceProviderError(Exception):
    """Base exception for all inference provider errors."""
    pass


class ProviderUnavailableError(InferenceProviderError):
    """Raised when an inference provider cannot be initialized or its runtime is missing."""
    pass


class ModelNotFoundError(InferenceProviderError):
    """Raised when a requested model is not available in the provider."""
    pass


@dataclass
class ModelDescriptor:
    """Metadata describing a local or available model."""
    id: str
    name: str
    is_installed: bool = True
    context_length: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderHealth:
    """Detailed health status report for an inference provider."""
    ok: bool
    name: str
    status: str
    current_model: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class InferenceProvider(ABC):
    """Abstract interface representing an AI inference provider."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique provider identifier (e.g. 'ollama', 'embedded')."""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """The default model identifier for this provider."""
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Start or verify runtime dependencies, daemons, or engines."""
        pass

    @abstractmethod
    def is_healthy(self) -> bool:
        """Quick boolean probe indicating whether the provider is ready for inference."""
        pass

    @abstractmethod
    def health(self) -> ProviderHealth:
        """Return a structured health report."""
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """Return a list of locally installed model names or identifiers."""
        pass

    @abstractmethod
    def stream_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.4,
        stop_event: Optional[threading.Event] = None,
    ) -> Generator[Tuple[str, str], None, None]:
        """Stream a chat completion, yielding tuples of (kind, text_chunk).

        Args:
            model: Name or identifier of the model to use.
            messages: List of message dictionaries with 'role' and 'content'.
            max_new_tokens: Optional token generation budget.
            temperature: Sampling temperature.
            stop_event: Optional threading.Event to signal early cancellation.

        Yields:
            Tuples of (kind, chunk) where:
            - kind: 'thinking' (reasoning trace) or 'answer' (visible assistant text)
            - chunk: generated text slice
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Cleanly terminate any background processes owned by this provider."""
        pass
