import unittest
from unittest.mock import patch

from robin.brain.runtime import OllamaRuntime


class RuntimeTests(unittest.TestCase):
    def test_reuses_existing_server(self):
        runtime = OllamaRuntime()

        with patch.object(runtime, "server_ready", return_value=True):
            self.assertFalse(runtime.ensure_running())

    def test_missing_runtime_fails_clearly(self):
        runtime = OllamaRuntime()

        with (
            patch.object(runtime, "server_ready", return_value=False),
            patch.object(runtime, "find_ollama_binary", return_value=None),
        ):
            with self.assertRaisesRegex(RuntimeError, "runtime was not found"):
                runtime.ensure_running()


if __name__ == "__main__":
    unittest.main()
