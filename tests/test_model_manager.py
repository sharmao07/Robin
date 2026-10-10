import unittest
from types import SimpleNamespace

from robin.brain.model_manager import (
    download_model,
    list_installed_models,
)


class FakeClient:
    def list(self):
        return SimpleNamespace(models=[
            SimpleNamespace(model="qwen3:4b", name=None),
            SimpleNamespace(model=None, name="llama3.2:3b"),
            SimpleNamespace(model="qwen3:4b", name=None),
        ])

    def pull(self, name, stream=True):
        self.pull_call = (name, stream)
        return iter([
            SimpleNamespace(status="downloading", completed=5, total=10),
            SimpleNamespace(status="success", completed=None, total=None),
        ])


class ModelManagerTests(unittest.TestCase):
    def test_list_installed_models_deduplicates_and_sorts(self):
        names = list_installed_models(FakeClient())
        self.assertEqual(names, ["llama3.2:3b", "qwen3:4b"])

    def test_download_model_reports_progress(self):
        client = FakeClient()
        progress = []

        download_model(
            client,
            "qwen3:4b",
            lambda status, percent: progress.append((status, percent)),
        )

        self.assertEqual(client.pull_call, ("qwen3:4b", True))
        self.assertEqual(progress, [
            ("downloading", 50),
            ("success", None),
        ])

    def test_download_model_does_not_require_callback(self):
        download_model(FakeClient(), "qwen3:4b")


if __name__ == "__main__":
    unittest.main()
