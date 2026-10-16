"""Local GGUF model manager and registry for Robin's embedded inference engine.

Manages model registry, download with resume and byte-level progress,
SHA-256 integrity verification, disk space checks, and offline model persistence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

LOG = logging.getLogger("robin.brain.model_manager")


class ModelManagerError(Exception):
    """Base exception for model manager errors."""
    pass


class InsufficientDiskSpaceError(ModelManagerError):
    """Raised when free disk space is insufficient for model download."""
    pass


class ChecksumMismatchError(ModelManagerError):
    """Raised when downloaded model digest does not match pinned expected SHA-256."""
    pass


class ModelArtifactNotFoundError(ModelManagerError):
    """Raised when an unknown model artifact ID is requested."""
    pass


@dataclass(frozen=True)
class ModelArtifact:
    """Metadata describing a pinned, verified GGUF model artifact."""
    id: str
    display_name: str
    publisher: str
    repo_id: str
    revision: str
    filename: str
    download_url: str
    sha256: str
    size_bytes: int
    license: str
    context_length: int = 40960
    recommended_ctx: int = 4096
    estimated_ram_mb: int = 3500
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Canonical Model Registry
#
# Pinned artifacts with verified licenses, immutable commit revisions,
# and exact SHA-256 digests.
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, ModelArtifact] = {
    "qwen3-4b-q4_k_m": ModelArtifact(
        id="qwen3-4b-q4_k_m",
        display_name="Qwen3 4B (Q4_K_M)",
        publisher="ggml-org",
        repo_id="ggml-org/Qwen3-4B-GGUF",
        revision="2f3b082b1356a6123f7ed71e65aea340da25d53c",
        filename="Qwen3-4B-Q4_K_M.gguf",
        download_url="https://huggingface.co/ggml-org/Qwen3-4B-GGUF/resolve/2f3b082b1356a6123f7ed71e65aea340da25d53c/Qwen3-4B-Q4_K_M.gguf",
        sha256="3e4cb14174460404e7a233e531675303b2fbf7749c02f91864fe311ab6344e4f",
        size_bytes=2497280480,
        license="Apache-2.0",
        context_length=40960,
        recommended_ctx=4096,
        estimated_ram_mb=3500,
        description="Lightweight 4-billion parameter reasoning model with native tool-calling support.",
    ),
}

DEFAULT_MODEL_ID = "qwen3-4b-q4_k_m"


def get_models_dir() -> Path:
    """Return the platform-specific directory where Robin stores local GGUF models."""
    override = os.environ.get("ROBIN_MODELS_DIR")
    if override and override.strip():
        p = Path(override.strip()).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    app_data = os.environ.get("ROBIN_APP_DATA_DIR")
    if app_data and app_data.strip():
        p = Path(app_data.strip()).expanduser().resolve() / "models"
        p.mkdir(parents=True, exist_ok=True)
        return p

    home = Path.home()
    if os.name == "nt":  # Windows
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else (home / "AppData" / "Roaming")
        models_dir = base / "Robin" / "models"
    elif os.uname().sysname == "Darwin":  # macOS
        models_dir = home / "Library" / "Application Support" / "Robin" / "models"
    else:  # Linux / standard Unix
        xdg_data = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg_data) if xdg_data else (home / ".local" / "share")
        models_dir = base / "robin" / "models"

    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir



@dataclass
class DownloadJob:
    """Tracks asynchronous model download progress and integrity state."""
    job_id: str
    model_id: str
    status: str = "pending"  # pending, downloading, verifying, completed, failed, cancelled
    bytes_downloaded: int = 0
    total_bytes: int = 0
    percent: float = 0.0
    speed_mbps: float = 0.0
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    stop_event: threading.Event = field(default_factory=threading.Event)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "model_id": self.model_id,
            "status": self.status,
            "bytes_downloaded": self.bytes_downloaded,
            "total_bytes": self.total_bytes,
            "percent": round(self.percent, 2),
            "speed_mbps": round(self.speed_mbps, 2),
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class LocalModelManager:
    """Manages local model storage, integrity verification, and downloads."""

    def __init__(self, models_dir: Optional[Path] = None):
        self._models_dir = models_dir or get_models_dir()
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, DownloadJob] = {}
        self._lock = threading.Lock()

    @property
    def models_dir(self) -> Path:
        return self._models_dir

    def get_artifact(self, model_id: str) -> ModelArtifact:
        artifact = MODEL_REGISTRY.get(model_id.lower().strip())
        if artifact is None:
            raise ModelArtifactNotFoundError(
                f"Model '{model_id}' is not in the Robin registry. "
                f"Available models: {', '.join(MODEL_REGISTRY.keys())}"
            )
        return artifact

    def validate_model_file(self, file_path: Path, artifact: ModelArtifact) -> Tuple[bool, Optional[str]]:
        """Validate model file integrity before activation:
        1. Must be a regular file that exists.
        2. Must not be a partial download (*.tmp.*, *.download).
        3. Must have authentic binary GGUF magic bytes (b'GGUF').
        4. Must match expected byte size within tolerance.
        """
        if not file_path.is_file():
            return False, f"File does not exist: {file_path}"

        if ".tmp." in file_path.name or file_path.name.endswith(".download"):
            return False, f"File is an unverified temporary download: {file_path.name}"

        try:
            with open(file_path, "rb") as f:
                magic = f.read(4)
                if magic != b"GGUF":
                    return False, f"Invalid model binary: expected magic header b'GGUF', got {magic!r}"
                version_bytes = f.read(4)
                if len(version_bytes) == 4:
                    import struct
                    version = struct.unpack("<I", version_bytes)[0]
                    if version not in (2, 3):
                        return False, f"Unsupported GGUF version {version} (expected 2 or 3)"
        except Exception as exc:
            return False, f"Failed to inspect model header: {exc}"

        actual_size = file_path.stat().st_size
        if abs(actual_size - artifact.size_bytes) > 4096:
            return False, f"Model file size mismatch: expected ~{artifact.size_bytes} bytes, got {actual_size}"

        return True, None

    def validate_model_checksum(self, model_id: str) -> Tuple[bool, Optional[str]]:
        """Perform full cryptographic SHA-256 integrity verification on an installed model."""
        try:
            artifact = self.get_artifact(model_id)
        except ModelArtifactNotFoundError as exc:
            return False, str(exc)

        target = self._models_dir / artifact.filename
        valid, reason = self.validate_model_file(target, artifact)
        if not valid:
            return False, reason

        if not self.verify_file_checksum(target, artifact.sha256):
            return False, f"SHA-256 digest mismatch for {artifact.filename}. Expected {artifact.sha256}"
        return True, None

    def get_model_path(self, model_id: str) -> Optional[Path]:
        """Return the path to the installed model file, or None if not present, corrupted, or incomplete."""
        try:
            artifact = self.get_artifact(model_id)
        except ModelArtifactNotFoundError:
            return None

        target = self._models_dir / artifact.filename
        valid, _ = self.validate_model_file(target, artifact)
        if valid:
            return target
        return None

    def is_installed(self, model_id: str) -> bool:
        return self.get_model_path(model_id) is not None

    def list_installed(self) -> List[Dict[str, Any]]:
        """List all models currently installed and ready for inference."""
        installed = []
        for artifact in MODEL_REGISTRY.values():
            path = self.get_model_path(artifact.id)
            if path is not None:
                installed.append({
                    **artifact.to_dict(),
                    "path": str(path),
                    "is_installed": True,
                    "actual_size_bytes": path.stat().st_size,
                })
        return installed

    def list_available(self) -> List[Dict[str, Any]]:
        """List all registry models along with their installation status."""
        available = []
        for artifact in MODEL_REGISTRY.values():
            path = self.get_model_path(artifact.id)
            is_inst = path is not None
            available.append({
                **artifact.to_dict(),
                "path": str(path) if path else None,
                "is_installed": is_inst,
            })
        return available

    def check_disk_space(self, required_bytes: int) -> Tuple[bool, int]:
        """Check whether sufficient disk space exists for the model download."""
        try:
            stat = shutil.disk_usage(self._models_dir)
            free_bytes = stat.free
            # Require 15% buffer above raw model size
            needed = int(required_bytes * 1.15)
            return (free_bytes >= needed, free_bytes)
        except Exception as exc:
            LOG.warning("Could not probe disk space for %s: %s", self._models_dir, exc)
            return (True, required_bytes * 2)

    def verify_file_checksum(self, file_path: Path, expected_sha256: str) -> bool:
        """Compute the SHA-256 digest of a file in 1MB chunks and compare."""
        if not file_path.is_file():
            return False
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        digest = h.hexdigest().lower()
        return digest == expected_sha256.lower().strip()

    def delete_model(self, model_id: str) -> bool:
        """Delete an installed model file from disk."""
        path = self.get_model_path(model_id)
        if path is not None and path.is_file():
            try:
                path.unlink()
                LOG.info("Deleted model file %s", path)
                return True
            except OSError as exc:
                LOG.error("Failed to delete model file %s: %s", path, exc)
                return False
        return False

    def start_download(self, model_id: str) -> DownloadJob:
        """Start an asynchronous download job for a registered model."""
        artifact = self.get_artifact(model_id)
        final_path = self._models_dir / artifact.filename

        if final_path.is_file() and final_path.stat().st_size == artifact.size_bytes:
            # Already installed and size matches
            job = DownloadJob(
                job_id=str(uuid.uuid4()),
                model_id=artifact.id,
                status="completed",
                bytes_downloaded=artifact.size_bytes,
                total_bytes=artifact.size_bytes,
                percent=100.0,
                finished_at=time.time(),
            )
            with self._lock:
                self._jobs[job.job_id] = job
            return job

        # Check disk space
        has_space, free_bytes = self.check_disk_space(artifact.size_bytes)
        if not has_space:
            raise InsufficientDiskSpaceError(
                f"Insufficient disk space. Needed ~{artifact.size_bytes // (1024*1024)} MB, "
                f"but only {free_bytes // (1024*1024)} MB is available."
            )

        job_id = str(uuid.uuid4())
        job = DownloadJob(
            job_id=job_id,
            model_id=artifact.id,
            status="downloading",
            total_bytes=artifact.size_bytes,
        )
        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(
            target=self._run_download,
            args=(job, artifact, final_path),
            daemon=True,
            name=f"model-dl-{job_id[:8]}",
        )
        thread.start()
        return job

    def get_job(self, job_id: str) -> Optional[DownloadJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel_download(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status in ("pending", "downloading", "verifying"):
                job.stop_event.set()
                job.status = "cancelled"
                job.finished_at = time.time()
                return True
        return False

    def _run_download(self, job: DownloadJob, artifact: ModelArtifact, final_path: Path) -> None:
        temp_path = self._models_dir / f"{artifact.filename}.tmp.{job.job_id[:8]}"
        try:
            req = urllib.request.Request(
                artifact.download_url,
                headers={"User-Agent": "RobinDesktop/0.1 (+https://github.com/ggml-org/llama.cpp)"},
            )

            # Check if partial download exists and resume if possible
            downloaded = 0
            if temp_path.exists():
                downloaded = temp_path.stat().st_size
                if downloaded < artifact.size_bytes:
                    req.add_header("Range", f"bytes={downloaded}-")
                else:
                    # Clean corrupt oversized temp
                    temp_path.unlink()
                    downloaded = 0

            mode = "ab" if downloaded > 0 else "wb"
            start_time = time.time()
            bytes_since_start = 0

            LOG.info("Starting download for %s -> %s", artifact.id, temp_path)
            with urllib.request.urlopen(req, timeout=30) as resp, open(temp_path, mode) as f:
                content_len = resp.headers.get("Content-Length")
                if content_len:
                    job.total_bytes = int(content_len) + downloaded
                else:
                    job.total_bytes = artifact.size_bytes

                job.bytes_downloaded = downloaded

                while True:
                    if job.stop_event.is_set():
                        LOG.info("Download cancelled for %s", artifact.id)
                        job.status = "cancelled"
                        job.finished_at = time.time()
                        return

                    chunk = resp.read(256 * 1024)  # 256KB chunks
                    if not chunk:
                        break

                    f.write(chunk)
                    job.bytes_downloaded += len(chunk)
                    bytes_since_start += len(chunk)

                    elapsed = time.time() - start_time
                    if elapsed > 0.5:
                        job.speed_mbps = (bytes_since_start / (1024 * 1024)) / elapsed
                        if job.total_bytes > 0:
                            job.percent = min(99.0, (job.bytes_downloaded / job.total_bytes) * 100.0)

            # Verification phase
            job.status = "verifying"
            job.percent = 99.0
            LOG.info("Verifying SHA-256 integrity for %s...", artifact.id)

            if not self.verify_file_checksum(temp_path, artifact.sha256):
                temp_path.unlink(missing_ok=True)
                raise ChecksumMismatchError(
                    f"Checksum verification failed for {artifact.filename}. Expected {artifact.sha256}"
                )

            # Atomic move to final destination
            temp_path.replace(final_path)
            job.status = "completed"
            job.percent = 100.0
            job.finished_at = time.time()
            LOG.info("Successfully downloaded and verified %s -> %s", artifact.id, final_path)

        except Exception as exc:
            LOG.exception("Download failed for %s", artifact.id)
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            job.status = "failed"
            job.error = str(exc)
            job.finished_at = time.time()


_DEFAULT_MANAGER: Optional[LocalModelManager] = None


def get_default_model_manager() -> LocalModelManager:
    """Return the shared singleton LocalModelManager instance."""
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        _DEFAULT_MANAGER = LocalModelManager()
    return _DEFAULT_MANAGER


# ─────────────────────────────────────────────────────────────────────────────
# Backward compatibility helpers for Ollama integration
# ─────────────────────────────────────────────────────────────────────────────

def list_installed_models(client=None) -> List[str]:
    """List installed models across both embedded local engine and optional Ollama client."""
    if client is not None:
        response = client.list()
        names = []
        for item in getattr(response, "models", []):
            name = getattr(item, "model", None) or getattr(item, "name", None)
            if name:
                names.append(name)
        return sorted(set(names))

    mgr = get_default_model_manager()
    return sorted(set([m["id"] for m in mgr.list_installed()]))


def download_model(client, name, progress_callback=None):
    """Download a model via Ollama (legacy compatibility)."""
    for status in client.pull(name, stream=True):
        description = getattr(status, "status", "Downloading")
        completed = getattr(status, "completed", None)
        total = getattr(status, "total", None)

        progress = None
        if total:
            progress = int((completed or 0) * 100 / total)

        if progress_callback:
            progress_callback(description, progress)
