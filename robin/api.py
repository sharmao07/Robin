"""Loopback-only HTTP bridge for Robin's native SwiftUI macOS client.

Run from the repository root with: python -m robin.api
This module deliberately binds only to 127.0.0.1 and does not expose a LAN service.
"""
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from robin.brain.local_llm import load_brain, stream_reply
from robin.brain.memory import initialize_memory, load_recent_messages, save_turn
from robin.brain.model_manager import list_installed_models
from robin.brain.personality import load_personality
from robin.brain.providers import get_provider
from robin.brain.runtime import OllamaRuntime

HOST = "127.0.0.1"
PORT = 8765
MAX_BODY_BYTES = 32 * 1024
LOG = logging.getLogger("robin.api")


def build_reply(message, model, client=None, system_prompt=None, provider=None):
    """Generate and persist one response. Split out for easy unit testing."""
    prompt = message.strip()
    if not prompt:
        raise ValueError("Message cannot be empty.")
    if len(prompt) > 12_000:
        raise ValueError("Message is too long (maximum 12,000 characters).")

    messages = [{"role": "system", "content": system_prompt or load_personality()}]
    messages.extend(load_recent_messages(limit=8))
    messages.append({"role": "user", "content": prompt})

    answer_parts = []
    if provider is not None:
        stream = provider.stream_chat(model, messages)
    else:
        stream = stream_reply(model, client, messages)

    for kind, text in stream:
        if kind == "answer":
            answer_parts.append(text)
    answer = "".join(answer_parts).strip()
    if not answer:
        raise RuntimeError("The local model returned an empty response.")
    save_turn(prompt, answer)
    return answer


class RobinHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class RobinHandler(BaseHTTPRequestHandler):
    server_version = "RobinLocalAPI/0.1"

    def log_message(self, fmt, *args):
        LOG.info("%s - %s", self.address_string(), fmt % args)

    def _json(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _is_local_request(self):
        # Native URLSession requests may omit Origin. Reject explicitly remote origins.
        origin = self.headers.get("Origin")
        if not origin:
            return True
        return origin in ("http://127.0.0.1", "http://localhost") or origin.startswith(
            "http://127.0.0.1:"
        ) or origin.startswith("http://localhost:")

    def do_GET(self):
        if not self._is_local_request():
            self._json(403, {"error": "Only local Robin clients are allowed."})
            return
        path = urlparse(self.path).path
        if path == "/api/health":
            provider = getattr(self.server, "provider", None)
            is_ok = provider.is_healthy() if provider is not None else True
            payload = {"ok": is_ok, "app": "Robin", "model": self.server.model}
            if provider is not None:
                payload["provider"] = provider.name
            self._json(200 if is_ok else 503, payload)
        elif path == "/api/models":
            try:
                provider = getattr(self.server, "provider", None)
                if provider is not None:
                    models = provider.list_models()
                else:
                    models = list_installed_models(self.server.client)
                self._json(200, {"models": models, "current": self.server.model})
            except Exception as error:
                self._json(503, {"error": f"Could not list local models: {error}"})
        else:
            self._json(404, {"error": "Not found."})

    def do_POST(self):
        if not self._is_local_request():
            self._json(403, {"error": "Only local Robin clients are allowed."})
            return
        if urlparse(self.path).path != "/api/chat":
            self._json(404, {"error": "Not found."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > MAX_BODY_BYTES:
                self._json(413, {"error": "Request body is empty or too large."})
                return
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object.")
            message = payload.get("message")
            if not isinstance(message, str):
                raise ValueError("The message field must be text.")
            requested_model = payload.get("model")
            model = self.server.model
            provider = getattr(self.server, "provider", None)
            client = getattr(self.server, "client", None)
            if requested_model:
                available = (
                    provider.list_models()
                    if provider is not None
                    else list_installed_models(client)
                )
                if requested_model not in available:
                    raise ValueError("That model is not installed locally.")
                model = requested_model
            answer = build_reply(message, model, client=client, provider=provider)
            self._json(200, {"answer": answer, "model": model})
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            self._json(400, {"error": str(error)})
        except Exception as error:
            LOG.exception("Chat request failed")
            self._json(500, {"error": f"Robin could not respond: {error}"})


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    provider = get_provider()
    try:
        provider.initialize()
        initialize_memory()
        server = RobinHTTPServer((HOST, PORT), RobinHandler)
        server.provider = provider
        server.model = provider.default_model
        # Retain runtime and client attributes for backward compatibility
        server.runtime = getattr(provider, "runtime", None)
        server.client = getattr(provider, "client", None)
        LOG.info(
            "Robin local API ready at http://%s:%s (provider: %s, model: %s)",
            HOST,
            PORT,
            provider.name,
            server.model,
        )
        try:
            server.serve_forever(poll_interval=0.25)
        finally:
            server.server_close()
    except KeyboardInterrupt:
        LOG.info("Robin API stopping")
    except Exception:
        LOG.exception("Robin API could not start")
        raise
    finally:
        provider.shutdown()


if __name__ == "__main__":
    main()
