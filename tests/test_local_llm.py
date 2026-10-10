import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from robin.brain.local_llm import stream_reply


class LocalLLMTests(unittest.TestCase):
    def test_uses_selected_model(self):
        fake_client = Mock()
        fake_client.chat.return_value = iter([
            SimpleNamespace(
                message=SimpleNamespace(
                    thinking="",
                    content="Hello",
                )
            )
        ])

        result = list(
            stream_reply(
                "another-model",
                fake_client,
                [{"role": "user", "content": "Hi"}],
            )
        )

        self.assertEqual(
            fake_client.chat.call_args.kwargs["model"],
            "another-model",
        )
        self.assertTrue(any(kind == "answer" for kind, _ in result))


if __name__ == "__main__":
    unittest.main()
