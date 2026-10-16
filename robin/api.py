"""Loopback-only HTTP bridge for Robin's native SwiftUI macOS client.

Run from the repository root with: python -m robin.api
This module deliberately binds only to 127.0.0.1 and does not expose a LAN service.
"""
import asyncio
import json
import logging
import threading
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from robin.brain.config import get_or_create_api_token
from robin.brain.local_llm import load_brain, stream_reply
from robin.brain.memory import (
    clear_history,
    create_session,
    delete_project_fact,
    delete_session,
    get_latest_handoff,
    get_project_facts,
    initialize_memory,
    list_sessions,
    load_recent_messages,
    save_handoff,
    save_project_fact,
    save_turn,
    search_sessions,
)
from robin.brain.model_manager import (
    DEFAULT_MODEL_ID,
    InsufficientDiskSpaceError,
    ModelArtifactNotFoundError,
    get_default_model_manager,
    list_installed_models,
)
from robin.brain.personality import load_personality
from robin.brain.providers import get_provider
from robin.brain.config import APP_DATA_DIR
from robin.brain.providers.llama_cpp import resolve_llama_server_binary
from robin.brain.scanner import FixMySlopScanner, get_result, store_result
from robin.brain.capabilities import ExecutionStatus, get_default_registry, run_tool_loop
from robin.brain.knowledge import get_default_knowledge_service
from robin.brain.media import (
    MediaGenerationRequest,
    MediaModality,
    get_default_media_manager,
)

HOST = "127.0.0.1"
PORT = 8765
MAX_BODY_BYTES = 32 * 1024
LOG = logging.getLogger("robin.api")


def build_reply(
    message,
    model,
    client=None,
    system_prompt=None,
    provider=None,
    project_path=None,
    session_id=None,
    return_citations=False,
    return_details=False,
    knowledge_service=None,
    registry=None,
):
    """Generate and persist one response. Split out for easy unit testing."""
    prompt = message.strip()
    if not prompt:
        raise ValueError("Message cannot be empty.")
    if len(prompt) > 12_000:
        raise ValueError("Message is too long (maximum 12,000 characters).")

    citations = []
    evidence_text = ""
    if project_path:
        try:
            svc = knowledge_service or get_default_knowledge_service()
            results = svc.search(project_path, prompt, limit=3)
            if results:
                blocks = []
                for res in results:
                    blocks.append(
                        f"File: {res.rel_path} (lines {res.start_line}-{res.end_line}):\n{res.content}"
                    )
                    citations.append({
                        "file": res.rel_path,
                        "start_line": res.start_line,
                        "end_line": res.end_line,
                        "snippet": res.content[:250],
                        "score": round(res.score, 4),
                    })
                evidence_text = "\n\n".join(blocks)
        except Exception as err:
            LOG.warning("Could not retrieve project knowledge for chat: %s", err)

    if project_path:
        try:
            facts = get_project_facts(project_path)
            if facts:
                fact_lines = [f"- {f['key']}: {f['value']}" for f in facts]
                if evidence_text:
                    evidence_text += "\n\nVerified Project Facts:\n" + "\n".join(fact_lines)
                else:
                    evidence_text = "Verified Project Facts:\n" + "\n".join(fact_lines)
        except Exception:
            pass

    system_content = system_prompt or load_personality()
    if evidence_text:
        system_content += (
            "\n\n[Project Knowledge Context]\n"
            "The following verified code excerpts from the active project may be relevant:\n\n"
            f"{evidence_text}\n\n"
            "Use this project knowledge where relevant to provide truthful, evidence-backed answers. "
            "Cite files and line numbers accurately."
        )

    recent_msgs = load_recent_messages(limit=8, session_id=session_id)

    # Run model-driven capability execution loop
    loop = asyncio.new_event_loop()
    try:
        tool_res = loop.run_until_complete(
            run_tool_loop(
                prompt=prompt,
                model=model,
                provider=provider,
                client=client,
                registry=registry,
                workspace_root=project_path,
                system_prompt=system_content,
                recent_messages=recent_msgs,
                stream_fn=stream_reply,
            )
        )
    finally:
        loop.close()

    answer = tool_res.answer.strip()
    if not answer:
        raise RuntimeError("The local model returned an empty response.")

    if session_id is not None or project_path is not None:
        save_turn(prompt, answer, session_id=session_id, project_path=project_path)
    else:
        save_turn(prompt, answer)

    # Merge R2 knowledge retrieval citations with tool-discovered citations
    merged_citations = list(citations)
    seen_files = {(c["file"], c.get("start_line", 1)) for c in merged_citations}
    for c in tool_res.citations:
        key = (c["file"], c.get("start_line", 1))
        if key not in seen_files:
            seen_files.add(key)
            merged_citations.append(c)

    if return_details:
        return (
            answer,
            merged_citations,
            tool_res.tool_activities,
            tool_res.needs_confirmation,
            tool_res.ticket_id,
            tool_res.confirmation_action,
            tool_res.confirmation_target,
        )
    if return_citations:
        return answer, merged_citations
    return answer



class RobinHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass, auth_token=None):
        super().__init__(server_address, RequestHandlerClass)
        self.auth_token = auth_token or get_or_create_api_token()


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

    def _requires_auth(self, path: str) -> bool:
        # Public health check, models discovery, and engine status endpoints
        if path in ("/api/health", "/api/models", "/api/engine/status", "/api/media/status"):
            return False
        if path.startswith("/api/models/download/") or path.startswith("/api/media/outputs/"):
            return False
        return True

    def _is_authorized(self) -> bool:
        expected_token = getattr(self.server, "auth_token", None)
        if not isinstance(expected_token, str) or not expected_token:
            return False

        token = self.headers.get("X-Robin-Token")
        if not token:
            auth = self.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                token = auth[7:].strip()

        if not token or not isinstance(token, str):
            return False

        return secrets.compare_digest(token.strip(), expected_token.strip())

    def do_GET(self):
        if not self._is_local_request():
            self._json(403, {"error": "Only local Robin clients are allowed."})
            return
        path = urlparse(self.path).path
        if self._requires_auth(path) and not self._is_authorized():
            self._json(401, {"error": "Unauthorized: Missing or invalid Robin API token."})
            return
        if path == "/api/health":
            provider = getattr(self.server, "provider", None)
            is_ok = provider.is_healthy() if provider is not None else True
            payload = {
                "ok": is_ok,
                "app": "Robin",
                "model": self.server.model,
                "status": "ready" if is_ok else "setup_required",
            }
            if provider is not None:
                payload["provider"] = provider.name
            self._json(200, payload)
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
        elif path == "/api/engine/status":
            provider = getattr(self.server, "provider", None)
            is_ok = provider.is_healthy() if provider is not None else False
            engine_name = getattr(provider, "name", "llama.cpp")
            active_model = getattr(self.server, "model", DEFAULT_MODEL_ID)

            try:
                runtime_path = str(resolve_llama_server_binary())
                runtime_installed = True
            except Exception:
                runtime_path = ""
                runtime_installed = False

            mgr = get_default_model_manager()
            model_installed = mgr.is_installed(active_model)
            available_models = mgr.list_available()
            installed_models = mgr.list_installed()

            payload = {
                "engine": engine_name,
                "runtime_installed": runtime_installed,
                "runtime_path": runtime_path,
                "model_installed": model_installed,
                "active_model": active_model,
                "is_healthy": is_ok,
                "available_models": available_models,
                "installed_models": installed_models,
            }
            self._json(200, payload)
        elif path.startswith("/api/models/download/"):
            job_id = path[len("/api/models/download/"):].strip()
            if not job_id:
                self._json(400, {"error": "Missing job_id in URL."})
                return
            mgr = get_default_model_manager()
            job = mgr.get_job(job_id)
            if job is None:
                self._json(404, {"error": "Download job not found."})
                return
            self._json(200, job.to_dict())
        elif path == "/api/audit/scanner":
            # Report whether fixmyslop is available for use
            scanner = FixMySlopScanner()
            self._json(200, {
                "available": scanner.is_available(),
                "scanner": "fixmyslop",
                "install_hint": "pip install fixmyslop",
            })
        elif path.startswith("/api/audit/scan/"):
            # GET /api/audit/scan/<scan_id> — poll job status
            scan_id = path[len("/api/audit/scan/"):]
            result = get_result(scan_id)
            if result is None:
                self._json(404, {"error": "Scan job not found."})
            else:
                status = "running" if result.finished_at is None else "complete"
                payload = {"status": status, **result.to_dict()}
                self._json(200, payload)
        elif path == "/api/capabilities":
            registry = get_default_registry()
            self._json(200, {"capabilities": registry.list_capabilities()})
        elif path == "/api/knowledge/status":
            qs = parse_qs(urlparse(self.path).query)
            project_path = qs.get("project_path", [""])[0].strip()
            if not project_path:
                self._json(400, {"error": "Missing 'project_path' query parameter."})
                return
            service = get_default_knowledge_service()
            try:
                status = service.get_status(project_path)
                if status is None:
                    self._json(200, {"status": "not_indexed", "project_path": project_path})
                else:
                    self._json(200, {"status": "indexed", **status})
            except Exception as exc:
                self._json(400, {"error": str(exc)})
        elif path == "/api/knowledge/search":
            qs = parse_qs(urlparse(self.path).query)
            project_path = qs.get("project_path", [""])[0].strip()
            query = qs.get("query", [""])[0].strip()
            limit_str = qs.get("limit", ["5"])[0]
            try:
                limit = int(limit_str)
            except ValueError:
                limit = 5
            if not project_path or not query:
                self._json(400, {"error": "Both 'project_path' and 'query' query parameters are required."})
                return
            service = get_default_knowledge_service()
            try:
                results = service.search(project_path, query, limit=limit)
                self._json(200, {
                    "project_path": project_path,
                    "query": query,
                    "results": [r.to_dict() for r in results],
                })
            except Exception as exc:
                self._json(400, {"error": str(exc)})
        elif path == "/api/sessions":
            qs = parse_qs(urlparse(self.path).query)
            project_path = qs.get("project_path", [None])[0]
            sessions = list_sessions(project_path=project_path)
            self._json(200, {"sessions": sessions})
        elif path == "/api/sessions/search":
            qs = parse_qs(urlparse(self.path).query)
            query = qs.get("query", [""])[0].strip()
            project_path = qs.get("project_path", [None])[0]
            if not query:
                self._json(400, {"error": "Missing 'query' query parameter."})
                return
            matches = search_sessions(query, project_path=project_path)
            self._json(200, {"query": query, "matches": matches})
        elif path == "/api/memory/facts":
            qs = parse_qs(urlparse(self.path).query)
            project_path = qs.get("project_path", [""])[0].strip()
            if not project_path:
                self._json(400, {"error": "Missing 'project_path' query parameter."})
                return
            facts = get_project_facts(project_path)
            self._json(200, {"project_path": project_path, "facts": facts})
        elif path == "/api/memory/handoff":
            qs = parse_qs(urlparse(self.path).query)
            project_path = qs.get("project_path", [""])[0].strip()
            if not project_path:
                self._json(400, {"error": "Missing 'project_path' query parameter."})
                return
            handoff = get_latest_handoff(project_path)
            self._json(200, {"project_path": project_path, "handoff": handoff})
        elif path == "/api/media/status":
            mgr = get_default_media_manager()
            self._json(200, mgr.get_engine_status())
        elif path == "/api/media/jobs":
            mgr = get_default_media_manager()
            qs = parse_qs(urlparse(self.path).query)
            try:
                limit = int(qs.get("limit", ["50"])[0])
            except ValueError:
                limit = 50
            jobs = [j.to_dict() for j in mgr.list_jobs(limit=limit)]
            self._json(200, {"jobs": jobs})
        elif path.startswith("/api/media/jobs/"):
            job_id = path[len("/api/media/jobs/"):].strip()
            mgr = get_default_media_manager()
            job = mgr.get_job(job_id)
            if not job:
                self._json(404, {"error": f"Media job '{job_id}' not found."})
            else:
                self._json(200, job.to_dict())
        elif path.startswith("/api/media/outputs/"):
            filename = path[len("/api/media/outputs/"):].strip()
            if "/" in filename or "\\" in filename or ".." in filename or not filename:
                self._json(400, {"error": "Invalid filename."})
                return
            outputs_dir = (APP_DATA_DIR / "media" / "outputs").resolve()
            out_file = (outputs_dir / filename).resolve()
            try:
                out_file.relative_to(outputs_dir)
            except ValueError:
                self._json(403, {"error": "Access forbidden."})
                return
            if not out_file.exists() or not out_file.is_file():
                self._json(404, {"error": "Media file not found."})
                return
            ext = out_file.suffix.lower()
            mime = "image/png" if ext == ".png" else ("image/jpeg" if ext in (".jpg", ".jpeg") else ("video/mp4" if ext == ".mp4" else "application/octet-stream"))
            data = out_file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)
        else:
            self._json(404, {"error": "Not found."})


    def do_POST(self):
        if not self._is_local_request():
            self._json(403, {"error": "Only local Robin clients are allowed."})
            return
        path = urlparse(self.path).path
        if self._requires_auth(path) and not self._is_authorized():
            self._json(401, {"error": "Unauthorized: Missing or invalid Robin API token."})
            return

        if path.startswith("/api/models/download/") and path.endswith("/cancel"):
            parts = path.strip("/").split("/")
            if len(parts) == 5:
                job_id = parts[3].strip()
                mgr = get_default_model_manager()
                job = mgr.get_job(job_id)
                if job is None:
                    self._json(404, {"error": "Download job not found."})
                    return
                cancelled = mgr.cancel_download(job_id)
                self._json(200, {"job_id": job_id, "cancelled": cancelled, "status": job.status})
                return
            else:
                self._json(400, {"error": "Invalid cancel URL format."})
                return

        if path.startswith("/api/media/jobs/") and path.endswith("/cancel"):
            parts = path.strip("/").split("/")
            if len(parts) == 5:
                job_id = parts[3].strip()
                mgr = get_default_media_manager()
                cancelled = mgr.cancel_job(job_id)
                self._json(200, {"job_id": job_id, "cancelled": cancelled})
                return
            else:
                self._json(400, {"error": "Invalid cancel URL format."})
                return

        if path == "/api/media/generate":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > MAX_BODY_BYTES:
                    self._json(413, {"error": "Request body is empty or too large."})
                    return
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object.")
                prompt = payload.get("prompt")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("prompt must be a non-empty string.")
                modality_str = str(payload.get("modality", "image")).lower()
                modality = MediaModality.VIDEO if modality_str == "video" else MediaModality.IMAGE
                steps_val = int(payload["steps"]) if "steps" in payload and payload["steps"] is not None else None
                seed_val = int(payload["seed"]) if "seed" in payload and payload["seed"] is not None else None
                req = MediaGenerationRequest(
                    prompt=prompt.strip(),
                    modality=modality,
                    negative_prompt=payload.get("negative_prompt"),
                    model=payload.get("model"),
                    width=int(payload.get("width", 1024)),
                    height=int(payload.get("height", 1024)),
                    aspect_ratio=str(payload.get("aspect_ratio", "1:1")),
                    reference_image_path=payload.get("reference_image_path"),
                    steps=steps_val,
                    seed=seed_val,
                    duration_seconds=float(payload.get("duration_seconds", 4.0)),
                    fps=int(payload.get("fps", 24)),
                    parameters=payload.get("parameters", {}),
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                self._json(400, {"error": str(error)})
                return

            mgr = get_default_media_manager()
            job = mgr.submit_job(req)
            self._json(202, {
                "job_id": job.job_id,
                "status": job.status.value,
                "poll_url": f"/api/media/jobs/{job.job_id}",
                **job.to_dict(),
            })
            return

        if path == "/api/models/download":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = {}
                if length > 0:
                    raw = self.rfile.read(min(length, MAX_BODY_BYTES))
                    payload = json.loads(raw.decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("Expected a JSON object.")
                model_id = payload.get("model_id") or DEFAULT_MODEL_ID
                if not isinstance(model_id, str) or not model_id.strip():
                    raise ValueError("model_id must be a non-empty string.")
                model_id = model_id.strip()
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                self._json(400, {"error": str(error)})
                return

            mgr = get_default_model_manager()
            try:
                job = mgr.start_download(model_id)
                self._json(202, {
                    "job_id": job.job_id,
                    "status": job.status,
                    "model_id": job.model_id,
                    "poll_url": f"/api/models/download/{job.job_id}",
                    **job.to_dict(),
                })
            except (ModelArtifactNotFoundError, InsufficientDiskSpaceError) as err:
                self._json(400, {"error": str(err)})
            except Exception as exc:
                LOG.exception("Failed to start model download: %s", exc)
                self._json(500, {"error": f"Failed to start download: {exc}"})
            return

        if path == "/api/engine/initialize":
            provider = getattr(self.server, "provider", None)
            if provider is None:
                self._json(500, {"error": "No inference provider registered on server."})
                return
            try:
                provider.initialize()
                self._json(200, {
                    "ok": provider.is_healthy(),
                    "engine": provider.name,
                    "active_model": getattr(self.server, "model", DEFAULT_MODEL_ID),
                })
            except Exception as exc:
                LOG.warning("Failed to initialize provider on demand: %s", exc)
                self._json(500, {"error": f"Failed to initialize engine: {exc}"})
            return

        if path == "/api/audit/scan":
            # POST /api/audit/scan — start an async scan job
            # Body: {"project_path": "/absolute/path"}
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > MAX_BODY_BYTES:
                    self._json(413, {"error": "Request body is empty or too large."})
                    return
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object.")
                project_path = payload.get("project_path")
                if not isinstance(project_path, str) or not project_path.strip():
                    raise ValueError("project_path must be a non-empty string.")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                self._json(400, {"error": str(error)})
                return

            scanner = FixMySlopScanner()

            # Reserve a result slot immediately so the scan_id is known
            import uuid, time as _time
            from robin.brain.scanner import ScanResult
            scan_id = str(uuid.uuid4())
            placeholder = ScanResult(
                scan_id=scan_id,
                project_path=project_path.strip(),
                started_at=_time.time(),
            )
            store_result(placeholder)

            def _run_scan():
                """Run the async scanner in a dedicated event loop on a worker thread."""
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(
                        scanner.scan(project_path.strip())
                    )
                    # Preserve the pre-assigned scan_id
                    result.scan_id = scan_id
                    store_result(result)
                finally:
                    loop.close()

            t = threading.Thread(target=_run_scan, daemon=True, name=f"scan-{scan_id[:8]}")
            t.start()

            self._json(202, {
                "scan_id": scan_id,
                "status": "running",
                "poll_url": f"/api/audit/scan/{scan_id}",
            })
            return

        if path.startswith("/api/capabilities/") and path.endswith("/execute"):
            parts = path.strip("/").split("/")
            if len(parts) == 4:
                cap_name = parts[2]
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length < 1 or length > MAX_BODY_BYTES:
                        self._json(413, {"error": "Request body is empty or too large."})
                        return
                    raw = self.rfile.read(length)
                    payload = json.loads(raw.decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("Expected a JSON object.")
                    params = payload.get("params", {})
                    if not isinstance(params, dict):
                        raise ValueError("'params' field must be a JSON object.")
                    confirm = bool(payload.get("confirm", False))
                    approval_token = payload.get("approval_token")
                    strict = bool(payload.get("strict", False))
                    timeout = payload.get("timeout")
                    if timeout is not None:
                        timeout = float(timeout)
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                    self._json(400, {"error": str(error)})
                    return

                registry = get_default_registry()
                loop = asyncio.new_event_loop()
                try:
                    # Enforce strict_tickets=True and permission_granted=False at HTTP API boundary
                    # External API callers cannot bypass confirmation with bare confirm=True
                    result = loop.run_until_complete(
                        registry.execute(
                            name=cap_name,
                            params=params,
                            permission_granted=False,
                            approval_token=approval_token,
                            caller="api",
                            strict_tickets=True,
                            timeout=timeout,
                        )
                    )
                finally:
                    loop.close()

                status_code = 200 if result.success else (403 if result.status == ExecutionStatus.DENIED else 400)
                self._json(status_code, result.to_dict())
                return

        if path == "/api/capabilities/approve":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > MAX_BODY_BYTES:
                    self._json(413, {"error": "Request body is empty or too large."})
                    return
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object.")
                ticket_id = payload.get("ticket_id")
                if not isinstance(ticket_id, str) or not ticket_id.strip():
                    raise ValueError("ticket_id must be a non-empty string.")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                self._json(400, {"error": str(error)})
                return

            from robin.tools.permissions import get_default_approval_manager
            mgr = get_default_approval_manager()
            approved = mgr.approve_ticket(ticket_id.strip())
            if approved:
                self._json(200, {"approved": True, "ticket_id": ticket_id.strip()})
            else:
                self._json(400, {"approved": False, "error": "Invalid or expired ticket ID."})
            return

        if path == "/api/knowledge/index":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > MAX_BODY_BYTES:
                    self._json(413, {"error": "Request body is empty or too large."})
                    return
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object.")
                project_path = payload.get("project_path")
                if not isinstance(project_path, str) or not project_path.strip():
                    raise ValueError("project_path must be a non-empty string.")
                max_files = int(payload.get("max_files", 4000))
                max_file_size = int(payload.get("max_file_size", 256 * 1024))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                self._json(400, {"error": str(error)})
                return

            service = get_default_knowledge_service()
            try:
                stats = service.index_project(
                    project_path=project_path.strip(),
                    max_files=max_files,
                    max_file_size=max_file_size,
                )
                self._json(200, {
                    "status": "indexed",
                    **stats.to_dict(),
                })
            except Exception as exc:
                self._json(400, {"error": str(exc)})
            return

        if path == "/api/knowledge/search":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > MAX_BODY_BYTES:
                    self._json(413, {"error": "Request body is empty or too large."})
                    return
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object.")
                project_path = payload.get("project_path")
                query = payload.get("query")
                if not isinstance(project_path, str) or not project_path.strip():
                    raise ValueError("project_path must be a non-empty string.")
                if not isinstance(query, str) or not query.strip():
                    raise ValueError("query must be a non-empty string.")
                limit = int(payload.get("limit", 5))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                self._json(400, {"error": str(error)})
                return

            service = get_default_knowledge_service()
            try:
                results = service.search(project_path.strip(), query.strip(), limit=limit)
                self._json(200, {
                    "project_path": project_path.strip(),
                    "query": query.strip(),
                    "results": [r.to_dict() for r in results],
                })
            except Exception as exc:
                self._json(400, {"error": str(exc)})
            return

        if path == "/api/knowledge/delete":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > MAX_BODY_BYTES:
                    self._json(413, {"error": "Request body is empty or too large."})
                    return
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object.")
                project_path = payload.get("project_path")
                if not isinstance(project_path, str) or not project_path.strip():
                    raise ValueError("project_path must be a non-empty string.")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                self._json(400, {"error": str(error)})
                return

            service = get_default_knowledge_service()
            try:
                deleted = service.delete_index(project_path.strip())
                self._json(200, {"deleted": deleted, "project_path": project_path.strip()})
            except Exception as exc:
                self._json(400, {"error": str(exc)})
            return

        if path == "/api/sessions":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > MAX_BODY_BYTES:
                    self._json(413, {"error": "Request body is too large."})
                    return
                payload = {}
                if length > 0:
                    raw = self.rfile.read(length)
                    payload = json.loads(raw.decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("Expected a JSON object.")
                title = payload.get("title", "New Chat")
                project_path = payload.get("project_path")
                sid = create_session(title=title, project_path=project_path)
                self._json(201, {"session_id": sid, "title": title, "project_path": project_path})
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                self._json(400, {"error": str(error)})
            return

        if path == "/api/memory/facts":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > MAX_BODY_BYTES:
                    self._json(413, {"error": "Request body is empty or too large."})
                    return
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object.")
                project_path = payload.get("project_path")
                fact_key = payload.get("fact_key") or payload.get("key")
                fact_value = payload.get("fact_value") or payload.get("value")
                source = payload.get("source", "user")
                if not isinstance(project_path, str) or not project_path.strip():
                    raise ValueError("project_path must be a non-empty string.")
                if not isinstance(fact_key, str) or not fact_key.strip():
                    raise ValueError("fact_key must be a non-empty string.")
                if not isinstance(fact_value, str) or not fact_value.strip():
                    raise ValueError("fact_value must be a non-empty string.")
                fid = save_project_fact(
                    project_path=project_path.strip(),
                    fact_key=fact_key.strip(),
                    fact_value=fact_value.strip(),
                    source=str(source),
                )
                self._json(200, {
                    "id": fid,
                    "project_path": project_path.strip(),
                    "fact_key": fact_key.strip(),
                    "fact_value": fact_value.strip(),
                })
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                self._json(400, {"error": str(error)})
            return

        if path == "/api/memory/handoff":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > MAX_BODY_BYTES:
                    self._json(413, {"error": "Request body is empty or too large."})
                    return
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object.")
                project_path = payload.get("project_path")
                progress = payload.get("progress")
                session_id = payload.get("session_id")
                next_step = payload.get("next_step", "")
                blockers = payload.get("blockers", "")
                file_references = payload.get("file_references", [])
                if not isinstance(project_path, str) or not project_path.strip():
                    raise ValueError("project_path must be a non-empty string.")
                if not isinstance(progress, str) or not progress.strip():
                    raise ValueError("progress must be a non-empty string.")
                if not isinstance(file_references, list):
                    raise ValueError("file_references must be a list of strings.")
                hid = save_handoff(
                    project_path=project_path.strip(),
                    progress=progress.strip(),
                    session_id=session_id,
                    next_step=str(next_step),
                    blockers=str(blockers),
                    file_references=file_references,
                )
                self._json(200, {"id": hid, "project_path": project_path.strip(), "saved": True})
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                self._json(400, {"error": str(error)})
            return

        if path != "/api/chat":
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
            project_path = payload.get("project_path")
            session_id = payload.get("session_id")
            if project_path is not None and not isinstance(project_path, str):
                raise ValueError("project_path must be a string if provided.")
            if session_id is not None and not isinstance(session_id, str):
                raise ValueError("session_id must be a string if provided.")
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
            details = build_reply(
                message,
                model,
                client=client,
                provider=provider,
                project_path=project_path,
                session_id=session_id,
                return_details=True,
            )
            (
                answer,
                citations,
                tool_activities,
                needs_confirmation,
                ticket_id,
                confirmation_action,
                confirmation_target,
            ) = details
            response_payload = {
                "answer": answer,
                "model": model,
                "session_id": session_id or "default",
                "citations": citations,
                "tool_activities": tool_activities,
                "needs_confirmation": needs_confirmation,
            }
            if ticket_id:
                response_payload["ticket_id"] = ticket_id
            if confirmation_action:
                response_payload["confirmation_action"] = confirmation_action
            if confirmation_target:
                response_payload["confirmation_target"] = confirmation_target
            self._json(200, response_payload)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            self._json(400, {"error": str(error)})
        except Exception as error:
            LOG.exception("Chat request failed")
            self._json(500, {"error": f"Robin could not respond: {error}"})

    def do_DELETE(self):
        if not self._is_local_request():
            self._json(403, {"error": "Only local Robin clients are allowed."})
            return
        path = urlparse(self.path).path
        if self._requires_auth(path) and not self._is_authorized():
            self._json(401, {"error": "Unauthorized: Missing or invalid Robin API token."})
            return
        if path == "/api/knowledge":
            qs = parse_qs(urlparse(self.path).query)
            project_path = qs.get("project_path", [""])[0].strip()
            if not project_path:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 0:
                    try:
                        raw = self.rfile.read(min(length, MAX_BODY_BYTES))
                        payload = json.loads(raw.decode("utf-8"))
                        if isinstance(payload, dict):
                            project_path = str(payload.get("project_path", "")).strip()
                    except Exception:
                        pass
            if not project_path:
                self._json(400, {"error": "Missing 'project_path' parameter."})
                return
            service = get_default_knowledge_service()
            try:
                deleted = service.delete_index(project_path)
                self._json(200, {"deleted": deleted, "project_path": project_path})
            except Exception as exc:
                self._json(400, {"error": str(exc)})
            return
        if path == "/api/sessions":
            qs = parse_qs(urlparse(self.path).query)
            session_id = qs.get("session_id", [""])[0].strip()
            if not session_id:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 0:
                    try:
                        raw = self.rfile.read(min(length, MAX_BODY_BYTES))
                        payload = json.loads(raw.decode("utf-8"))
                        if isinstance(payload, dict):
                            session_id = str(payload.get("session_id", "")).strip()
                    except Exception:
                        pass
            if not session_id:
                self._json(400, {"error": "Missing 'session_id' parameter."})
                return
            deleted = delete_session(session_id)
            self._json(200, {"deleted": deleted, "session_id": session_id})
            return
        if path == "/api/memory/facts":
            qs = parse_qs(urlparse(self.path).query)
            fact_id = qs.get("id", qs.get("fact_id", [""]))[0].strip()
            if not fact_id:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 0:
                    try:
                        raw = self.rfile.read(min(length, MAX_BODY_BYTES))
                        payload = json.loads(raw.decode("utf-8"))
                        if isinstance(payload, dict):
                            fact_id = str(payload.get("id") or payload.get("fact_id") or "").strip()
                    except Exception:
                        pass
            if not fact_id:
                self._json(400, {"error": "Missing 'id' or 'fact_id' parameter."})
                return
            try:
                deleted = delete_project_fact(int(fact_id))
                self._json(200, {"deleted": deleted, "id": int(fact_id)})
            except (ValueError, TypeError) as exc:
                self._json(400, {"error": f"Invalid fact id: {exc}"})
            return
        if path.startswith("/api/models/") or path == "/api/models":
            model_id = ""
            if path.startswith("/api/models/"):
                model_id = path[len("/api/models/"):].strip()
            if not model_id:
                qs = parse_qs(urlparse(self.path).query)
                model_id = qs.get("model_id", [""])[0].strip()
            if not model_id:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 0:
                    try:
                        raw = self.rfile.read(min(length, MAX_BODY_BYTES))
                        payload = json.loads(raw.decode("utf-8"))
                        if isinstance(payload, dict):
                            model_id = str(payload.get("model_id", "")).strip()
                    except Exception:
                        pass
            if not model_id:
                self._json(400, {"error": "Missing 'model_id' parameter."})
                return
            mgr = get_default_model_manager()
            deleted = mgr.delete_model(model_id)
            self._json(200, {"deleted": deleted, "model_id": model_id})
            return
        self._json(404, {"error": "Not found."})



def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    provider = get_provider()
    try:
        try:
            provider.initialize()
        except Exception as init_err:
            LOG.warning(
                "Inference provider could not be initialized at startup (%s). "
                "Robin API will run in setup/onboarding mode waiting for model installation.",
                init_err,
            )
        initialize_memory()
        server = RobinHTTPServer(
            (HOST, PORT), RobinHandler, auth_token=get_or_create_api_token()
        )
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
