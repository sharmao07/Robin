"""Ollama development adapter implementing the InferenceProvider interface."""
import logging
import threading
from typing import Any, Dict, Generator, List, Optional, Tuple

from robin.brain.local_llm import stream_reply
from robin.brain.model_manager import list_installed_models
from robin.brain.providers.base import (
    InferenceProvider,
    InferenceProviderError,
    ModelNotFoundError,
    ProviderHealth,
    ProviderUnavailableError,
)
from robin.brain.runtime import OllamaRuntime
from robin.brain.settings import DEFAULT_MODEL

LOG = logging.getLogger("robin.brain.providers.ollama")

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    ollama = None
    HAS_OLLAMA = False


class OllamaProvider(InferenceProvider):
    """Adapter for running inference through a local Ollama daemon."""

    def __init__(
        self,
        client: Optional[Any] = None,
        runtime: Optional[OllamaRuntime] = None,
        default_model: str = DEFAULT_MODEL,
        host: str = "http://127.0.0.1:11434",
    ):
        self._default_model = default_model
        self._host = host
        self._client = client
        self.runtime = runtime or OllamaRuntime()
        self._initialized = False

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def client(self) -> Any:
        if self._client is None:
            if not HAS_OLLAMA:
                raise ProviderUnavailableError(
                    "The 'ollama' Python package is not installed. "
                    "Install dependencies with: pip install -r requirements.txt"
                )
            self._client = ollama.Client(host=self._host)
        return self._client

    def initialize(self) -> None:
        """Start Ollama if needed and verify daemon connectivity."""
        if not HAS_OLLAMA and self._client is None:
            raise ProviderUnavailableError(
                "The 'ollama' Python package is not installed. "
                "Install dependencies with: pip install -r requirements.txt"
            )

        try:
            self.runtime.ensure_running()
        except Exception as error:
            raise ProviderUnavailableError(f"Could not start Ollama daemon: {error}") from error

        if not self.is_healthy():
            raise ProviderUnavailableError(
                f"Ollama server is not responding at {self._host}"
            )

        self._initialized = True
        LOG.info(
            "OllamaProvider initialized successfully with default model: %s",
            self._default_model,
        )

    def is_healthy(self) -> bool:
        """Probe the server status."""
        try:
            return self.runtime.server_ready()
        except Exception:
            return False

    def health(self) -> ProviderHealth:
        healthy = self.is_healthy()
        return ProviderHealth(
            ok=healthy,
            name=self.name,
            status="ready" if healthy else "unavailable",
            current_model=self._default_model,
            details={
                "host": self._host,
                "runtime_pid": getattr(self.runtime.process, "pid", None),
            },
        )

    def list_models(self) -> List[str]:
        """List all models currently installed in Ollama."""
        try:
            return list_installed_models(self.client)
        except Exception as error:
            raise InferenceProviderError(f"Could not list models from Ollama: {error}") from error

    def stream_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.4,
        stop_event: Optional[threading.Event] = None,
    ) -> Generator[Tuple[str, str], None, None]:
        """Stream chat tokens from Ollama using the existing tag-parsing pipeline."""
        if not self.is_healthy():
            raise ProviderUnavailableError("Ollama server is not healthy or responsive.")

        try:
            active_client = self.client
        except Exception as error:
            raise ProviderUnavailableError(str(error)) from error

        try:
            yield from stream_reply(
                model=model,
                client=active_client,
                messages=messages,
                stop_event=stop_event,
            )
        except Exception as error:
            error_str = str(error).lower()
            if "not found" in error_str or "does not exist" in error_str:
                raise ModelNotFoundError(f"Model '{model}' is not available: {error}") from error
            raise InferenceProviderError(f"Inference error with model '{model}': {error}") from error

    def shutdown(self) -> None:
        """Stop the Ollama server only if Robin started it."""
        try:
            self.runtime.stop_if_started()
        except Exception as error:
            LOG.warning("Error stopping Ollama runtime: %s", error)
        self._initialized = False
