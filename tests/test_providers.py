"""Unit tests for Robin's InferenceProvider abstraction and Ollama adapter."""
import os
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from robin.api import build_reply
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


class FakeRuntime:
    """Mock runtime for testing provider lifecycle."""

    def __init__(self, ready=True, fail_on_ensure=False):
        self.ready = ready
        self.fail_on_ensure = fail_on_ensure
        self.stopped = False
        self.process = SimpleNamespace(pid=12345)

    def server_ready(self):
        return self.ready

    def ensure_running(self, timeout=40):
        if self.fail_on_ensure:
            raise RuntimeError("Ollama binary not found on system.")
        return True

    def stop_if_started(self):
        self.stopped = True


class FakeOllamaClient:
    """Mock client for testing model listing and chat completions."""

    def __init__(self, models=None, chat_stream=None, fail_list=False, fail_chat=None):
        self._models = models or ["qwen3:4b", "llama3.2:3b"]
        self._chat_stream = chat_stream
        self._fail_list = fail_list
        self._fail_chat = fail_chat

    def list(self):
        if self._fail_list:
            raise RuntimeError("Connection refused by Ollama daemon.")
        return SimpleNamespace(
            models=[
                SimpleNamespace(model=m, name=None)
                for m in self._models
            ]
        )

    def chat(self, model, messages, stream=True, think=True, options=None):
        if self._fail_chat:
            raise self._fail_chat
        if self._chat_stream is not None:
            return iter(self._chat_stream)
        return iter([
            SimpleNamespace(
                message=SimpleNamespace(thinking="", content=f"Reply to: {messages[-1]['content']}")
            )
        ])


class MockCustomProvider(InferenceProvider):
    """Custom provider used to verify interface conformance and registration."""

    def __init__(self):
        self.initialized = False
        self.stopped = False

    @property
    def name(self) -> str:
        return "custom_mock"

    @property
    def default_model(self) -> str:
        return "custom:1b"

    def initialize(self) -> None:
        self.initialized = True

    def is_healthy(self) -> bool:
        return True

    def health(self) -> ProviderHealth:
        return ProviderHealth(ok=True, name=self.name, status="ready", current_model=self.default_model)

    def list_models(self):
        return ["custom:1b"]

    def stream_chat(self, model, messages, max_new_tokens=None, temperature=0.4, stop_event=None):
        yield ("answer", "Custom answer")

    def shutdown(self) -> None:
        self.stopped = True


class TestInferenceProviderBase(unittest.TestCase):
    def test_abstract_class_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            InferenceProvider()

    def test_custom_provider_conforms_to_interface(self):
        provider = MockCustomProvider()
        provider.initialize()
        self.assertTrue(provider.initialized)
        self.assertEqual(provider.name, "custom_mock")
        self.assertEqual(provider.default_model, "custom:1b")
        self.assertTrue(provider.is_healthy())
        health = provider.health()
        self.assertTrue(health.ok)
        self.assertEqual(health.name, "custom_mock")
        models = provider.list_models()
        self.assertEqual(models, ["custom:1b"])
        chunks = list(provider.stream_chat("custom:1b", [{"role": "user", "content": "hi"}]))
        self.assertEqual(chunks, [("answer", "Custom answer")])
        provider.shutdown()
        self.assertTrue(provider.stopped)

    def test_model_descriptor_dataclass(self):
        desc = ModelDescriptor(id="qwen3:4b", name="Qwen 3 4B", is_installed=True, context_length=32768)
        self.assertEqual(desc.id, "qwen3:4b")
        self.assertEqual(desc.context_length, 32768)


class TestOllamaProvider(unittest.TestCase):
    def setUp(self):
        self.runtime = FakeRuntime(ready=True)
        self.client = FakeOllamaClient()
        self.provider = OllamaProvider(client=self.client, runtime=self.runtime)

    def test_provider_metadata(self):
        self.assertEqual(self.provider.name, "ollama")
        self.assertEqual(self.provider.default_model, "qwen3:4b")

    def test_initialize_success(self):
        self.provider.initialize()
        self.assertTrue(self.provider.is_healthy())
        health = self.provider.health()
        self.assertTrue(health.ok)
        self.assertEqual(health.status, "ready")

    def test_initialize_fails_when_runtime_cannot_start(self):
        failing_runtime = FakeRuntime(fail_on_ensure=True)
        provider = OllamaProvider(client=self.client, runtime=failing_runtime)
        with self.assertRaises(ProviderUnavailableError) as ctx:
            provider.initialize()
        self.assertIn("Could not start Ollama daemon", str(ctx.exception))

    def test_initialize_fails_when_server_not_ready_after_ensure(self):
        unready_runtime = FakeRuntime(ready=False)
        provider = OllamaProvider(client=self.client, runtime=unready_runtime)
        with self.assertRaises(ProviderUnavailableError) as ctx:
            provider.initialize()
        self.assertIn("not responding", str(ctx.exception))

    def test_health_when_unhealthy(self):
        unready_runtime = FakeRuntime(ready=False)
        provider = OllamaProvider(client=self.client, runtime=unready_runtime)
        health = provider.health()
        self.assertFalse(health.ok)
        self.assertEqual(health.status, "unavailable")

    def test_list_models_success(self):
        models = self.provider.list_models()
        self.assertEqual(models, ["llama3.2:3b", "qwen3:4b"])

    def test_list_models_error_wrapped(self):
        failing_client = FakeOllamaClient(fail_list=True)
        provider = OllamaProvider(client=failing_client, runtime=self.runtime)
        with self.assertRaises(InferenceProviderError):
            provider.list_models()

    def test_stream_chat_success(self):
        self.client._chat_stream = [
            SimpleNamespace(message=SimpleNamespace(thinking="step 1", content="")),
            SimpleNamespace(message=SimpleNamespace(thinking="", content="Hello world")),
        ]
        results = list(self.provider.stream_chat(
            model="qwen3:4b",
            messages=[{"role": "user", "content": "Hi"}],
        ))
        self.assertEqual(results[0], ("thinking", "step 1"))
        answer_chunks = [chunk for kind, chunk in results if kind == "answer"]
        self.assertEqual("".join(answer_chunks), "Hello world")

    def test_stream_chat_fails_when_unhealthy(self):
        unready_runtime = FakeRuntime(ready=False)
        provider = OllamaProvider(client=self.client, runtime=unready_runtime)
        with self.assertRaises(ProviderUnavailableError):
            list(provider.stream_chat("qwen3:4b", [{"role": "user", "content": "Hi"}]))

    def test_stream_chat_model_not_found_error(self):
        failing_client = FakeOllamaClient(fail_chat=RuntimeError("model 'missing:model' not found, try pulling it first"))
        provider = OllamaProvider(client=failing_client, runtime=self.runtime)
        with self.assertRaises(ModelNotFoundError):
            list(provider.stream_chat("missing:model", [{"role": "user", "content": "Hi"}]))

    def test_stream_chat_general_error_wrapped(self):
        failing_client = FakeOllamaClient(fail_chat=RuntimeError("unexpected socket disconnection"))
        provider = OllamaProvider(client=failing_client, runtime=self.runtime)
        with self.assertRaises(InferenceProviderError):
            list(provider.stream_chat("qwen3:4b", [{"role": "user", "content": "Hi"}]))

    def test_stream_chat_honors_stop_event(self):
        stop_event = threading.Event()
        stop_event.set()  # Cancel immediately
        self.client._chat_stream = [
            SimpleNamespace(message=SimpleNamespace(thinking="", content="Unreachable token")),
        ]
        results = list(self.provider.stream_chat(
            model="qwen3:4b",
            messages=[{"role": "user", "content": "Hi"}],
            stop_event=stop_event,
        ))
        self.assertEqual(results, [])

    def test_shutdown(self):
        self.assertFalse(self.runtime.stopped)
        self.provider.shutdown()
        self.assertTrue(self.runtime.stopped)

    def test_missing_sdk_raises_provider_unavailable(self):
        with patch("robin.brain.providers.ollama.HAS_OLLAMA", False):
            provider = OllamaProvider(client=None, runtime=self.runtime)
            with self.assertRaises(ProviderUnavailableError) as ctx:
                _ = provider.client
            self.assertIn("requirements.txt", str(ctx.exception))


class TestProviderFactory(unittest.TestCase):
    def setUp(self):
        reset_default_provider()

    def tearDown(self):
        reset_default_provider()

    def test_get_provider_defaults_to_ollama(self):
        provider = get_provider()
        self.assertIsInstance(provider, OllamaProvider)
        self.assertEqual(provider.name, "ollama")

    def test_get_provider_explicit_ollama(self):
        provider = get_provider("ollama")
        self.assertIsInstance(provider, OllamaProvider)

    def test_get_provider_honors_environment_variable(self):
        with patch.dict(os.environ, {"ROBIN_INFERENCE_PROVIDER": "ollama"}):
            provider = get_provider()
            self.assertEqual(provider.name, "ollama")

    def test_get_provider_embedded_raises_clear_error_for_m1(self):
        with self.assertRaises(ProviderUnavailableError) as ctx:
            get_provider("embedded")
        self.assertIn("Milestone 3", str(ctx.exception))

    def test_get_provider_unknown_name_raises_error(self):
        with self.assertRaises(ProviderUnavailableError) as ctx:
            get_provider("unknown_engine_xyz")
        self.assertIn("Unknown inference provider", str(ctx.exception))

    def test_register_custom_provider(self):
        register_provider("test_custom", MockCustomProvider)
        provider = get_provider("test_custom")
        self.assertIsInstance(provider, MockCustomProvider)
        self.assertEqual(provider.name, "custom_mock")

    def test_default_provider_singleton_management(self):
        p1 = get_default_provider()
        p2 = get_default_provider()
        self.assertIs(p1, p2)

        custom = MockCustomProvider()
        set_default_provider(custom)
        self.assertIs(get_default_provider(), custom)

        reset_default_provider()
        p3 = get_default_provider()
        self.assertIsNot(p3, custom)
        self.assertIsInstance(p3, OllamaProvider)


class TestBuildReplyWithProvider(unittest.TestCase):
    @patch("robin.api.save_turn")
    @patch("robin.api.load_recent_messages", return_value=[])
    def test_build_reply_uses_provider_stream_chat(self, mock_history, mock_save):
        mock_provider = Mock(spec=InferenceProvider)
        mock_provider.stream_chat.return_value = iter([
            ("thinking", "analyzing intent..."),
            ("answer", "Here is the "),
            ("answer", "evidence-backed reply."),
        ])

        answer = build_reply(
            message="What was the last blocker?",
            model="qwen3:4b",
            provider=mock_provider,
        )

        self.assertEqual(answer, "Here is the evidence-backed reply.")
        mock_save.assert_called_once_with("What was the last blocker?", "Here is the evidence-backed reply.")
        mock_provider.stream_chat.assert_called_once()
        call_args = mock_provider.stream_chat.call_args
        self.assertEqual(call_args[0][0], "qwen3:4b")
        messages_sent = call_args[0][1]
        self.assertEqual(messages_sent[-1]["content"], "What was the last blocker?")


if __name__ == "__main__":
    unittest.main()
