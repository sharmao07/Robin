import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from robin.brain.config import APP_DATA_DIR


API_URL = "http://127.0.0.1:11434/api/tags"


class OllamaRuntime:
    """Start Ollama only when its local API is not already running."""

    def __init__(self):
        self.process = None
        self.log_path = APP_DATA_DIR / "logs" / "ollama-server.log"

    def server_ready(self):
        try:
            with urlopen(API_URL, timeout=1) as response:
                return response.status == 200
        except (OSError, URLError, TimeoutError):
            return False

    def find_ollama_binary(self):
        # When packaged, Robin will look for the bundled Ollama app first.
        executable = Path(sys.executable).resolve()
        bundled = (
            executable.parent.parent
            / "Resources"
            / "Ollama.app"
            / "Contents"
            / "Resources"
            / "ollama"
        )

        candidates = [
            Path(os.environ["ROBIN_OLLAMA_BIN"])
            if os.environ.get("ROBIN_OLLAMA_BIN")
            else bundled,
            Path("/Applications/Ollama.app/Contents/Resources/ollama"),
            Path("/usr/local/bin/ollama"),
        ]

        path_from_shell = shutil.which("ollama")
        if path_from_shell:
            candidates.append(Path(path_from_shell))

        for candidate in candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate

        return None

    def ensure_running(self, timeout=40):
        # Reuse an existing local server rather than starting a duplicate.
        if self.server_ready():
            return False

        binary = self.find_ollama_binary()

        if binary is None:
            raise RuntimeError(
                "Ollama's runtime was not found. "
                "Robin cannot start its local AI engine."
            )

        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["OLLAMA_HOST"] = "127.0.0.1:11434"

        with self.log_path.open("ab") as log_file:
            self.process = subprocess.Popen(
                [str(binary), "serve"],
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            if self.server_ready():
                return True

            if self.process.poll() is not None:
                break

            time.sleep(0.5)

        details = ""
        if self.log_path.exists():
            details = self.log_path.read_text(
                encoding="utf-8",
                errors="replace",
            )[-2000:]

        self.stop_if_started()

        raise RuntimeError(
            "Ollama did not become ready in time.\n"
            f"Server log: {self.log_path}\n{details}"
        )

    def stop_if_started(self):
        # Never terminate a server Robin did not start.
        if self.process is None or self.process.poll() is not None:
            return

        self.process.terminate()

        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)

        self.process = None
