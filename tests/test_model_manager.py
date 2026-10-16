"""Unit tests for Robin's LocalModelManager, model registry, and checksum verification."""

import hashlib
import io
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from robin.brain.model_manager import (
    DEFAULT_MODEL_ID,
    MODEL_REGISTRY,
    ChecksumMismatchError,
    DownloadJob,
    InsufficientDiskSpaceError,
    LocalModelManager,
    ModelArtifact,
    ModelArtifactNotFoundError,
    get_models_dir,
    list_installed_models,
)


class TestModelManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.temp_dir.name)
        self.manager = LocalModelManager(models_dir=self.models_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_canonical_registry_contains_default_qwen3(self):
        self.assertIn(DEFAULT_MODEL_ID, MODEL_REGISTRY)
        artifact = MODEL_REGISTRY[DEFAULT_MODEL_ID]
        self.assertEqual(artifact.id, "qwen3-4b-q4_k_m")
        self.assertEqual(artifact.publisher, "ggml-org")
        self.assertEqual(artifact.license, "Apache-2.0")
        self.assertEqual(artifact.filename, "Qwen3-4B-Q4_K_M.gguf")
        self.assertEqual(
            artifact.sha256,
            "3e4cb14174460404e7a233e531675303b2fbf7749c02f91864fe311ab6344e4f",
        )
        self.assertEqual(artifact.size_bytes, 2497280480)
        self.assertTrue(artifact.download_url.startswith("https://huggingface.co/"))

    def test_get_models_dir_respects_environment_override(self):
        with patch.dict(os.environ, {"ROBIN_MODELS_DIR": str(self.models_dir / "custom")}):
            custom_dir = get_models_dir()
            self.assertEqual(custom_dir, (self.models_dir / "custom").resolve())
            self.assertTrue(custom_dir.is_dir())

    def test_unknown_model_raises_error(self):
        with self.assertRaises(ModelArtifactNotFoundError):
            self.manager.get_artifact("non_existent_model_id")

    def test_verify_file_checksum(self):
        test_file = self.models_dir / "sample.bin"
        content = b"Robin Local AI Model Verification Test Bytes"
        test_file.write_bytes(content)
        expected_digest = hashlib.sha256(content).hexdigest()

        self.assertTrue(self.manager.verify_file_checksum(test_file, expected_digest))
        self.assertFalse(self.manager.verify_file_checksum(test_file, "wrong_hash_12345"))
        self.assertFalse(self.manager.verify_file_checksum(self.models_dir / "absent.bin", expected_digest))

    def test_list_installed_and_is_installed(self):
        artifact = self.manager.get_artifact(DEFAULT_MODEL_ID)
        self.assertFalse(self.manager.is_installed(DEFAULT_MODEL_ID))
        self.assertEqual(len(self.manager.list_installed()), 0)

        # Create dummy file with matching GGUF header and size
        target_path = self.models_dir / artifact.filename
        target_path.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\0" * (artifact.size_bytes - 8))

        self.assertTrue(self.manager.is_installed(DEFAULT_MODEL_ID))
        installed = self.manager.list_installed()
        self.assertEqual(len(installed), 1)
        self.assertEqual(installed[0]["id"], DEFAULT_MODEL_ID)
        self.assertEqual(installed[0]["path"], str(target_path))

    def test_corrupted_model_file_rejected_before_activation(self):
        artifact = self.manager.get_artifact(DEFAULT_MODEL_ID)
        target_path = self.models_dir / artifact.filename
        # Write corrupted header (not GGUF)
        target_path.write_bytes(b"CORRUPTED_NOT_A_VALID_GGUF" + b"\0" * (artifact.size_bytes - 27))

        valid, reason = self.manager.validate_model_file(target_path, artifact)
        self.assertFalse(valid)
        self.assertIn("expected magic header b'GGUF'", reason or "")
        self.assertFalse(self.manager.is_installed(DEFAULT_MODEL_ID))
        self.assertIsNone(self.manager.get_model_path(DEFAULT_MODEL_ID))

    def test_incomplete_download_file_not_activated(self):
        artifact = self.manager.get_artifact(DEFAULT_MODEL_ID)
        tmp_path = self.models_dir / f"{artifact.filename}.tmp.abc12345"
        tmp_path.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\0" * (artifact.size_bytes - 8))

        # Incomplete/temporary downloads must not be activated
        valid, reason = self.manager.validate_model_file(tmp_path, artifact)
        self.assertFalse(valid)
        self.assertIn("unverified temporary download", reason or "")
        self.assertFalse(self.manager.is_installed(DEFAULT_MODEL_ID))

    def test_delete_model(self):
        artifact = self.manager.get_artifact(DEFAULT_MODEL_ID)
        target_path = self.models_dir / artifact.filename
        target_path.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\0" * (artifact.size_bytes - 8))

        self.assertTrue(self.manager.is_installed(DEFAULT_MODEL_ID))
        deleted = self.manager.delete_model(DEFAULT_MODEL_ID)
        self.assertTrue(deleted)
        self.assertFalse(target_path.exists())
        self.assertFalse(self.manager.is_installed(DEFAULT_MODEL_ID))

    def test_check_disk_space_insufficient(self):
        with patch("shutil.disk_usage", return_value=Mock(free=1000)):
            has_space, free_bytes = self.manager.check_disk_space(2000)
            self.assertFalse(has_space)
            self.assertEqual(free_bytes, 1000)

            with self.assertRaises(InsufficientDiskSpaceError):
                self.manager.start_download(DEFAULT_MODEL_ID)

    def test_start_download_immediate_if_already_installed(self):
        artifact = self.manager.get_artifact(DEFAULT_MODEL_ID)
        target_path = self.models_dir / artifact.filename
        target_path.write_bytes(b"\0" * artifact.size_bytes)

        job = self.manager.start_download(DEFAULT_MODEL_ID)
        self.assertEqual(job.status, "completed")
        self.assertEqual(job.percent, 100.0)

    def test_download_progress_and_completion(self):
        content = b"X" * 1024
        expected_sha = hashlib.sha256(content).hexdigest()
        custom_artifact = ModelArtifact(
            id="test-model",
            display_name="Test Model",
            publisher="test-pub",
            repo_id="test/model",
            revision="rev1",
            filename="test_model.gguf",
            download_url="https://example.com/test_model.gguf",
            sha256=expected_sha,
            size_bytes=1024,
            license="MIT",
        )

        with patch.dict(MODEL_REGISTRY, {"test-model": custom_artifact}):
            mock_resp = MagicMock()
            mock_resp.headers = {"Content-Length": "1024"}
            mock_resp.read.side_effect = [content[:512], content[512:], b""]
            mock_resp.__enter__.return_value = mock_resp

            with patch("urllib.request.urlopen", return_value=mock_resp):
                job = self.manager.start_download("test-model")
                # Wait for background thread to complete
                timeout = 5.0
                start = time.time()
                while job.status in ("pending", "downloading", "verifying") and time.time() - start < timeout:
                    time.sleep(0.05)

                self.assertEqual(job.status, "completed")
                self.assertEqual(job.percent, 100.0)
                final_file = self.models_dir / "test_model.gguf"
                self.assertTrue(final_file.is_file())
                self.assertEqual(final_file.read_bytes(), content)

    def test_download_checksum_mismatch_fails_and_cleans_up(self):
        corrupt_content = b"Corrupt Content Bytes"
        wrong_sha = "0000000000000000000000000000000000000000000000000000000000000000"
        corrupt_artifact = ModelArtifact(
            id="corrupt-model",
            display_name="Corrupt Model",
            publisher="test",
            repo_id="test/corrupt",
            revision="rev1",
            filename="corrupt.gguf",
            download_url="https://example.com/corrupt.gguf",
            sha256=wrong_sha,
            size_bytes=len(corrupt_content),
            license="MIT",
        )

        with patch.dict(MODEL_REGISTRY, {"corrupt-model": corrupt_artifact}):
            mock_resp = MagicMock()
            mock_resp.headers = {"Content-Length": str(len(corrupt_content))}
            mock_resp.read.side_effect = [corrupt_content, b""]
            mock_resp.__enter__.return_value = mock_resp

            with patch("urllib.request.urlopen", return_value=mock_resp):
                job = self.manager.start_download("corrupt-model")
                timeout = 5.0
                start = time.time()
                while job.status in ("pending", "downloading", "verifying") and time.time() - start < timeout:
                    time.sleep(0.05)

                self.assertEqual(job.status, "failed")
                self.assertIn("Checksum verification failed", job.error or "")
                final_file = self.models_dir / "corrupt.gguf"
                self.assertFalse(final_file.exists())

    def test_cancel_download(self):
        block_artifact = ModelArtifact(
            id="block-model",
            display_name="Block Model",
            publisher="test",
            repo_id="test/block",
            revision="rev1",
            filename="block.gguf",
            download_url="https://example.com/block.gguf",
            sha256="abc",
            size_bytes=10000,
            license="MIT",
        )

        with patch.dict(MODEL_REGISTRY, {"block-model": block_artifact}):
            # Simulate a slow streaming response
            def slow_chunks(n):
                time.sleep(0.1)
                return b"a" * 100

            mock_resp = MagicMock()
            mock_resp.headers = {"Content-Length": "10000"}
            mock_resp.read.side_effect = slow_chunks
            mock_resp.__enter__.return_value = mock_resp

            with patch("urllib.request.urlopen", return_value=mock_resp):
                job = self.manager.start_download("block-model")
                time.sleep(0.15)
                cancelled = self.manager.cancel_download(job.job_id)
                self.assertTrue(cancelled)

                timeout = 2.0
                start = time.time()
                while job.status != "cancelled" and time.time() - start < timeout:
                    time.sleep(0.05)

                self.assertEqual(job.status, "cancelled")


if __name__ == "__main__":
    unittest.main()
