import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from robin.brain.model_manager import (
    list_installed_models,
    download_model,
)


class ModelManagerTests(unittest.TestCase):
    def test_list_models(self):
        client = Mock()
        client.list.return_value = SimpleNamespace(
            models=[
                SimpleNamespace(model="qwen3:4b"),
                SimpleNamespace(model="tiny-model"),
            ]
        )

        self.assertEqual(
            list_installed_models(client),
            ["qwen3:4b", "tiny-model"],
        )

    def test_download_progress(self):
        client = Mock()
        client.pull.return_value = iter([
            SimpleNamespace(
                status="downloading",
                completed=50,
                total=100,
            )
        ])

        progress = []

        download_model(
            client,
            "test-model",
            lambda status, percent: progress.append(
                (status, percent)
            ),
        )

        self.assertEqual(progress, [("downloading", 50)])


if __name__ == "__main__":
    unittest.main()
