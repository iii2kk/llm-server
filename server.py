from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import socket
import struct
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse


BASE_DIR = Path(__file__).resolve().parent
GRAMMAR_DIR = (BASE_DIR / "grammars").resolve()

load_dotenv(BASE_DIR / ".env")


def required_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Required environment variable {name} is not set or empty")
    return value


LLAMA_BIN_DIR = Path(required_env("LLAMA_BIN_DIR")).expanduser()
MODEL_DIR = Path(required_env("MODEL_DIR")).expanduser()
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8080"))
PROXY_HOST = os.getenv("PROXY_HOST", "0.0.0.0")
PROXY_PORT = int(os.getenv("PROXY_PORT", "8000"))
PROXY_API_KEY = os.getenv("PROXY_API_KEY", "")
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "512"))
GRAMMAR_DEFAULT_MAX_TOKENS = int(os.getenv("GRAMMAR_DEFAULT_MAX_TOKENS", "256"))
LOG_BUFFER_MAX_BYTES = int(os.getenv("LOG_BUFFER_MAX_BYTES", "1048576"))
MODEL_LOAD_TIMEOUT_SECONDS = float(os.getenv("MODEL_LOAD_TIMEOUT_SECONDS", "60"))
SETTINGS_DIR = BASE_DIR / ".llm-server"
MODEL_SETTINGS_FILE = SETTINGS_DIR / "model-settings.json"
RECENT_MODELS_MAX = 5
SAVED_BACKEND_SETTING_KEYS = (
    "model",
    "mmproj_enabled",
    "ctx_size",
    "gpu_layers",
    "threads",
    "batch_size",
    "ubatch_size",
    "parallel",
    "flash_attn",
    "reasoning",
    "reasoning_format",
    "mode",
    "pooling",
)
MODEL_MODES = ("auto", "chat", "embeddings")
POOLING_TYPES = ("auto", "mean", "cls", "last")
GGUF_POOLING_NAMES = {
    0: "none",
    1: "mean",
    2: "cls",
    3: "last",
    4: "rank",
}
GGUF_SCALAR_FORMATS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}
GGUF_METADATA_CACHE: dict[tuple[str, int, int], dict[str, Any]] = {}


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        yield
    finally:
        await registry.stop_all()


app = FastAPI(title="llama.cpp OpenAI-compatible proxy", lifespan=lifespan)


class SuppressStatusAccessLog(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/api/status" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(SuppressStatusAccessLog())
logger = logging.getLogger(__name__)


SHARDED_GGUF_RE = re.compile(r"^(?P<base>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$", re.IGNORECASE)


class BackendLogStore:
    def __init__(self, max_bytes: int, *, echo_stdout: bool = True) -> None:
        self.max_bytes = max(16 * 1024, max_bytes)
        self.echo_stdout = echo_stdout
        self.entries: deque[dict[str, Any]] = deque()
        self.current_bytes = 0
        self.seq = 0
        self.run_id = 0
        self.truncated = False
        self.load_state = "stopped"
        self.load_progress: int | None = None
        self.progress_dots = 0
        self.listeners: set[asyncio.Queue[dict[str, Any]]] = set()
        self._lock = asyncio.Lock()

    async def start_run(self) -> int:
        async with self._lock:
            self.run_id += 1
            self.entries.clear()
            self.current_bytes = 0
            self.seq = 0
            self.truncated = False
            self.load_state = "loading"
            self.load_progress = 0
            self.progress_dots = 0
            run_id = self.run_id
        await self._broadcast_state("state")
        return run_id

    async def append(self, text: str, run_id: int, *, model_id: str | None = None) -> None:
        if not text:
            return

        if self.echo_stdout:
            sys.stdout.write(text)
            sys.stdout.flush()

        entry: dict[str, Any] | None = None
        should_broadcast_state = False
        async with self._lock:
            if run_id != self.run_id:
                return

            entry = {
                "seq": self.seq,
                "ts": time.time(),
                "text": text,
                "run_id": self.run_id,
            }
            if model_id is not None:
                entry["model_id"] = model_id
            self.seq += 1
            self.entries.append(entry)
            self.current_bytes += len(text.encode("utf-8", errors="replace"))
            self._trim_locked()
            should_broadcast_state = self._update_progress_from_log_locked(text)

        await self._broadcast("log", entry)
        if should_broadcast_state:
            await self._broadcast_state("state")

    async def mark_ready(self, run_id: int) -> None:
        async with self._lock:
            if run_id != self.run_id or self.load_state == "ready":
                return
            self.load_state = "ready"
            self.load_progress = 100
        await self._broadcast_state("state")
        await self._broadcast_state("done")

    async def mark_error(self, run_id: int) -> None:
        async with self._lock:
            if run_id != self.run_id or self.load_state == "stopped":
                return
            self.load_state = "error"
        await self._broadcast_state("state")
        await self._broadcast_state("error")

    async def mark_stopped(self, run_id: int | None = None) -> None:
        async with self._lock:
            if run_id is not None and run_id != self.run_id:
                return
            self.load_state = "stopped"
            self.load_progress = None
            self.progress_dots = 0
        await self._broadcast_state("state")

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "run_id": self.run_id,
                "next_seq": self.seq,
                "entries": list(self.entries),
                "truncated": self.truncated,
                "load": self.load_summary_locked(),
            }

    async def load_summary(self) -> dict[str, Any]:
        async with self._lock:
            return self.load_summary_locked()

    def load_summary_locked(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "state": self.load_state,
            "progress": self.load_progress,
        }

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        async with self._lock:
            self.listeners.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        async with self._lock:
            self.listeners.discard(queue)

    def _trim_locked(self) -> None:
        while self.entries and self.current_bytes > self.max_bytes:
            old = self.entries.popleft()
            self.current_bytes -= len(str(old["text"]).encode("utf-8", errors="replace"))
            self.truncated = True

    def _update_progress_from_log_locked(self, text: str) -> bool:
        if self.load_state != "loading":
            return False

        dots = 0
        for part in text.replace("\r", "").split("\n"):
            stripped = part.strip()
            if stripped and set(stripped) <= {"."}:
                dots += stripped.count(".")

        if dots == 0:
            return False

        old_progress = self.load_progress
        self.progress_dots += dots
        self.load_progress = min(99, max(self.load_progress or 0, self.progress_dots))
        return self.load_progress != old_progress

    async def _broadcast_state(self, event: str) -> None:
        await self._broadcast(event, await self.load_summary())

    async def _broadcast(self, event: str, data: dict[str, Any]) -> None:
        async with self._lock:
            listeners = list(self.listeners)
        payload = {"event": event, "data": data}
        for queue in listeners:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass


class BackendInstance:
    def __init__(
        self,
        *,
        model_id: str,
        model_path: Path,
        port: int,
        aggregate_log_store: BackendLogStore,
    ) -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.port = port
        self.backend_url = f"http://{BACKEND_HOST}:{port}"
        self.process: asyncio.subprocess.Process | None = None
        self.command: list[str] = []
        self.settings: dict[str, Any] = {}
        self.started_at: float | None = None
        self.last_used_at: float | None = None
        self.last_exit_code: int | None = None
        self.run_id: int | None = None
        self.effective_mode = "chat"
        self.effective_pooling: str | None = None
        self.log_store = BackendLogStore(LOG_BUFFER_MAX_BYTES)
        self.aggregate_log_store = aggregate_log_store
        self._reader_task: asyncio.Task[None] | None = None
        self._health_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    def is_running(self) -> bool:
        return self.process is not None and self.process.returncode is None

    def is_active(self) -> bool:
        return self.is_running() or self.run_id is not None

    async def status(self) -> dict[str, Any]:
        process = self.process
        running = self.is_running()
        backend_health = await check_backend_health(self.backend_url)
        if running and backend_health and self.run_id is not None:
            await self.log_store.mark_ready(self.run_id)
        load = await self.log_store.load_summary()
        return {
            "model_id": self.model_id,
            "model_path": str(self.model_path),
            "running": running,
            "pid": process.pid if running and process else None,
            "started_at": self.started_at,
            "last_used_at": self.last_used_at,
            "uptime_seconds": int(time.time() - self.started_at) if running and self.started_at else None,
            "last_exit_code": self.last_exit_code,
            "port": self.port,
            "backend_url": self.backend_url,
            "backend_reachable": backend_health,
            "load_run_id": load["run_id"],
            "load_state": load["state"],
            "load_progress": load["progress"],
            "effective_mode": self.effective_mode,
            "effective_pooling": self.effective_pooling,
            "command": self.command,
        }

    async def start(self, settings: dict[str, Any], *, conflict_if_running: bool = True) -> dict[str, Any] | JSONResponse:
        async with self._lock:
            if self.is_running():
                if conflict_if_running:
                    return openai_error_response(
                        f"llama-server is already running for {self.model_id}",
                        status_code=409,
                        error_type="conflict_error",
                        code="backend_already_running",
                    )
                reachable = await wait_for_backend(self.backend_url)
                if reachable and self.run_id is not None:
                    await self.log_store.mark_ready(self.run_id)
                return {
                    "ok": True,
                    "backend_reachable": reachable,
                    "status": await self.status(),
                }

            command = build_llama_command(settings, model=self.model_path, port=self.port)
            run_id = await self.log_store.start_run()
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = prepend_env_path(
                env.get("LD_LIBRARY_PATH", ""),
                str(LLAMA_BIN_DIR.resolve()),
            )
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    env=env,
                    cwd=str(BASE_DIR),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
            except OSError:
                await self.log_store.mark_error(run_id)
                raise
            self.process = process
            self.command = command
            self.settings = dict(settings)
            self.effective_mode = str(settings["effective_mode"])
            self.effective_pooling = settings.get("effective_pooling")
            self.started_at = time.time()
            self.last_used_at = self.started_at
            self.last_exit_code = None
            self.run_id = run_id
            self._reader_task = asyncio.create_task(self._read_process_output(process, run_id))
            self._health_task = asyncio.create_task(self._monitor_backend_health(process, run_id))

        reachable = await wait_for_backend(self.backend_url)
        if reachable:
            await self.log_store.mark_ready(run_id)
        return {
            "ok": True,
            "backend_reachable": reachable,
            "status": await self.status(),
        }

    async def stop(self) -> dict[str, Any]:
        async with self._lock:
            process = self.process
            run_id = self.run_id
            if process is None or process.returncode is not None:
                already_stopped = True
            else:
                already_stopped = False

            if not already_stopped:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=15)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()

            self.last_exit_code = process.returncode if process is not None else self.last_exit_code
            self.process = None
            self.started_at = None
            self.command = []
            self.run_id = None

        if self._health_task is not None:
            self._health_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_task
            self._health_task = None
        if self._reader_task is not None:
            try:
                await asyncio.wait_for(self._reader_task, timeout=2)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._reader_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._reader_task
            self._reader_task = None

        await self.log_store.mark_stopped(run_id)

        if already_stopped:
            return {
                "ok": True,
                "message": f"llama-server is not running for {self.model_id}",
                "status": await self.status(),
            }

        return {"ok": True, "status": await self.status()}

    async def restart(self, settings: dict[str, Any]) -> dict[str, Any]:
        await self.stop()
        result = await self.start(settings)
        return result

    async def _read_process_output(self, process: asyncio.subprocess.Process, run_id: int) -> None:
        try:
            if process.stdout is None:
                return
            while True:
                chunk = await process.stdout.read(1024)
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace")
                await self.log_store.append(text, run_id)
                await self.aggregate_log_store.append(
                    prefix_log_text(self.model_id, text),
                    0,
                    model_id=self.model_id,
                )
        finally:
            try:
                await process.wait()
            except ProcessLookupError:
                pass
            await self._handle_process_exit(process, run_id)

    async def _monitor_backend_health(self, process: asyncio.subprocess.Process, run_id: int) -> None:
        try:
            while process.returncode is None:
                if await check_backend_health(self.backend_url):
                    await self.log_store.mark_ready(run_id)
                    return
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            raise

    async def _handle_process_exit(self, process: asyncio.subprocess.Process, run_id: int) -> None:
        async with self._lock:
            if self.process is not process:
                return
            self.last_exit_code = process.returncode
            self.process = None
            self.started_at = None
            self.command = []
            self.run_id = None

        if process.returncode == 0:
            await self.log_store.mark_stopped(run_id)
        else:
            await self.log_store.mark_error(run_id)


def saved_settings_payload(model_id: str, settings: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": model_id}
    for key in SAVED_BACKEND_SETTING_KEYS:
        if key == "model" or key not in settings:
            continue
        value = settings[key]
        if value is None or value == "":
            continue
        payload[key] = value
    return payload


def load_model_settings_document() -> dict[str, Any]:
    if not MODEL_SETTINGS_FILE.exists():
        return {}
    try:
        raw = json.loads(MODEL_SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load model settings from %s: %s", MODEL_SETTINGS_FILE, exc)
        return {}

    if not isinstance(raw, dict):
        logger.warning("Ignoring model settings from %s: root must be an object", MODEL_SETTINGS_FILE)
        return {}
    return raw


def load_saved_model_settings() -> dict[str, dict[str, Any]]:
    raw = load_model_settings_document()

    models = raw.get("models")
    if not isinstance(models, dict):
        return {}

    saved: dict[str, dict[str, Any]] = {}
    for model_id, settings in models.items():
        if not isinstance(model_id, str) or not isinstance(settings, dict):
            continue
        saved[model_id] = saved_settings_payload(model_id, settings)
    return saved


def load_recent_model_ids() -> list[str]:
    raw = load_model_settings_document()
    recent_models = raw.get("recent_models")
    if not isinstance(recent_models, list):
        return []

    seen: set[str] = set()
    model_ids: list[str] = []
    for item in recent_models:
        if not isinstance(item, str) or not item or item in seen:
            continue
        seen.add(item)
        model_ids.append(item)
        if len(model_ids) >= RECENT_MODELS_MAX:
            break
    return model_ids


def write_saved_model_settings(
    saved_settings: dict[str, dict[str, Any]],
    recent_model_ids: list[str] | None = None,
) -> None:
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "recent_models": list(recent_model_ids or [])[:RECENT_MODELS_MAX],
        "models": {
            model_id: saved_settings_payload(model_id, settings)
            for model_id, settings in sorted(saved_settings.items())
        },
    }
    tmp_path = MODEL_SETTINGS_FILE.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(MODEL_SETTINGS_FILE)


class BackendRegistry:
    def __init__(self) -> None:
        self.instances: dict[str, BackendInstance] = {}
        self.saved_settings: dict[str, dict[str, Any]] = load_saved_model_settings()
        self.recent_model_ids: list[str] = load_recent_model_ids()
        self.aggregate_log_store = BackendLogStore(LOG_BUFFER_MAX_BYTES, echo_stdout=False)
        self._lock = asyncio.Lock()

    async def status(self) -> dict[str, Any]:
        async with self._lock:
            instances = list(self.instances.values())

        statuses = await asyncio.gather(*(instance.status() for instance in instances))
        active = [status for status in statuses if status["running"] or status["load_state"] == "loading"]
        latest = max(active, key=lambda item: item["started_at"] or 0, default=None)
        return {
            "running": any(status["running"] for status in statuses),
            "backend_start_port": BACKEND_PORT,
            "backend_host": BACKEND_HOST,
            "count": len(active),
            "latest_model_id": latest["model_id"] if latest else None,
            "backend_url": latest["backend_url"] if latest else None,
            "backend_reachable": latest["backend_reachable"] if latest else False,
            "backends": statuses,
        }

    async def start(self, settings: dict[str, Any], *, conflict_if_running: bool = True) -> dict[str, Any] | JSONResponse:
        model_id, model_path = resolve_model_reference_required(settings.get("model"))
        normalized_settings = normalize_backend_settings(model_id, model_path, settings)
        if normalized_settings["effective_mode"] == "rerank":
            raise ValueError("rerank models are detected but /v1/rerank is not supported")
        instance = await self._instance_for(model_id, model_path)
        result = await instance.start(normalized_settings, conflict_if_running=conflict_if_running)
        if not isinstance(result, JSONResponse):
            self._save_settings(model_id, normalized_settings)
        return result

    async def restart(self, settings: dict[str, Any]) -> dict[str, Any] | JSONResponse:
        model_id, model_path = resolve_model_reference_required(settings.get("model"))
        normalized_settings = normalize_backend_settings(model_id, model_path, settings)
        if normalized_settings["effective_mode"] == "rerank":
            raise ValueError("rerank models are detected but /v1/rerank is not supported")
        instance = await self._instance_for(model_id, model_path)
        result = await instance.restart(normalized_settings)
        if not isinstance(result, JSONResponse):
            self._save_settings(model_id, normalized_settings)
        return result

    async def stop(self, settings: dict[str, Any]) -> dict[str, Any]:
        stop_all = not settings or optional_bool(settings, "all", False)
        if stop_all:
            return await self.stop_all()

        model_id, model_path = resolve_model_reference_required(settings.get("model"))
        instance = await self._instance_for(model_id, model_path)
        return await instance.stop()

    async def stop_all(self) -> dict[str, Any]:
        async with self._lock:
            instances = list(self.instances.values())
        results = await asyncio.gather(*(instance.stop() for instance in instances))
        return {"ok": True, "stopped": len(results), "status": await self.status()}

    async def logs_for(self, model_ref: str | None) -> BackendLogStore | None:
        if not model_ref:
            return self.aggregate_log_store
        resolved = resolve_model_reference(model_ref)
        model_id = resolved[0] if resolved else model_ref
        async with self._lock:
            instance = self.instances.get(model_id)
        if instance is None:
            return None
        return instance.log_store

    async def backend_for_request(
        self,
        model_ref: Any,
        *,
        purpose: str,
    ) -> tuple[BackendInstance | None, JSONResponse | None]:
        resolved = resolve_model_reference(model_ref)
        if resolved is not None:
            model_id, model_path = resolved
            instance = await self._instance_for(model_id, model_path)
            if not instance.is_running():
                try:
                    settings = normalize_backend_settings(
                        model_id,
                        model_path,
                        dict(self.saved_settings.get(model_id, {"model": model_id})),
                    )
                except ValueError as exc:
                    return None, openai_error_response(
                        str(exc),
                        param="model",
                        code="invalid_backend_settings",
                    )
                capability_error = model_capability_error(model_id, settings["effective_mode"], purpose)
                if capability_error:
                    return None, capability_error
                result = await instance.start(settings, conflict_if_running=False)
                if isinstance(result, JSONResponse):
                    return None, result
                if not result.get("backend_reachable"):
                    return None, openai_error_response(
                        f"Failed to load model: {model_id}",
                        status_code=503,
                        error_type="backend_load_error",
                        param="model",
                        code="model_load_failed",
                    )
                self._mark_recent(model_id)
            else:
                capability_error = model_capability_error(model_id, instance.effective_mode, purpose)
                if capability_error:
                    return None, capability_error
            if instance.is_running() and not await wait_for_backend(instance.backend_url):
                return None, openai_error_response(
                    f"Model is not ready: {model_id}",
                    status_code=503,
                    error_type="backend_load_error",
                    param="model",
                    code="model_load_failed",
                )
            if instance.is_running():
                self._mark_recent(model_id)
            instance.last_used_at = time.time()
            return instance, None

        instance = await self.latest_active_instance(purpose=purpose)
        if instance is None:
            code = "no_embedding_model_loaded" if purpose == "embeddings" else "no_chat_model_loaded"
            return None, openai_error_response(
                f"No {purpose} model is loaded",
                status_code=503,
                error_type="backend_connection_error",
                param="model",
                code=code,
            )
        if not await wait_for_backend(instance.backend_url):
            return None, openai_error_response(
                f"Latest model is not ready: {instance.model_id}",
                status_code=503,
                error_type="backend_load_error",
                param="model",
                code="model_load_failed",
            )
        self._mark_recent(instance.model_id)
        instance.last_used_at = time.time()
        return instance, None

    async def latest_active_instance(self, *, purpose: str | None = None) -> BackendInstance | None:
        async with self._lock:
            active = [
                instance
                for instance in self.instances.values()
                if instance.is_active() and (purpose is None or instance.effective_mode == purpose)
            ]
        return max(active, key=lambda instance: instance.started_at or 0, default=None)

    async def saved_settings_snapshot(self) -> dict[str, dict[str, Any]]:
        async with self._lock:
            return {model_id: dict(settings) for model_id, settings in self.saved_settings.items()}

    async def recent_model_ids_snapshot(self) -> list[str]:
        async with self._lock:
            return list(self.recent_model_ids)

    async def _instance_for(self, model_id: str, model_path: Path) -> BackendInstance:
        async with self._lock:
            instance = self.instances.get(model_id)
            if instance is not None:
                return instance
            port = self._allocate_port_locked()
            instance = BackendInstance(
                model_id=model_id,
                model_path=model_path,
                port=port,
                aggregate_log_store=self.aggregate_log_store,
            )
            self.instances[model_id] = instance
            return instance

    def _allocate_port_locked(self) -> int:
        used = {instance.port for instance in self.instances.values() if instance.is_active()}
        port = BACKEND_PORT
        while port in used or not is_tcp_port_available(BACKEND_HOST, port):
            port += 1
        return port

    def _save_settings(self, model_id: str, settings: dict[str, Any]) -> None:
        self.saved_settings[model_id] = saved_settings_payload(model_id, settings)
        self._mark_recent(model_id, persist=False)
        self._persist_settings()

    def _mark_recent(self, model_id: str, *, persist: bool = True) -> None:
        old = list(self.recent_model_ids)
        self.recent_model_ids = [item for item in self.recent_model_ids if item != model_id]
        self.recent_model_ids.insert(0, model_id)
        self.recent_model_ids = self.recent_model_ids[:RECENT_MODELS_MAX]
        if persist and self.recent_model_ids != old:
            self._persist_settings()

    def _persist_settings(self) -> None:
        try:
            write_saved_model_settings(self.saved_settings, self.recent_model_ids)
        except OSError as exc:
            logger.warning("Failed to save model settings to %s: %s", MODEL_SETTINGS_FILE, exc)


registry = BackendRegistry()


def prepend_env_path(existing: str, first_path: str) -> str:
    if not existing:
        return first_path
    return f"{first_path}:{existing}"


def openai_error_payload(message: str, error_type: str, param: str | None, code: str | None) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }


def openai_error_response(
    message: str,
    *,
    status_code: int = 400,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=openai_error_payload(message, error_type, param, code),
    )


def model_capability_error(model_id: str, effective_mode: str, purpose: str) -> JSONResponse | None:
    if effective_mode == purpose:
        return None
    if purpose == "embeddings":
        message = f"Model is not embedding-capable: {model_id}"
        code = "model_not_embedding_capable"
    else:
        message = f"Model is not chat-capable: {model_id}"
        code = "model_not_chat_capable"
    return openai_error_response(message, param="model", code=code)


def require_auth(authorization: str | None) -> JSONResponse | None:
    if not PROXY_API_KEY:
        return None
    expected = f"Bearer {PROXY_API_KEY}"
    if authorization != expected:
        return openai_error_response(
            "Invalid or missing API key",
            status_code=401,
            error_type="authentication_error",
            code="invalid_api_key",
        )
    return None


def require_auth_or_query(request: Request, authorization: str | None) -> JSONResponse | None:
    if not PROXY_API_KEY:
        return None
    if request.query_params.get("api_key") == PROXY_API_KEY:
        return None
    return require_auth(authorization)


def sse_event(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def prefix_log_text(model_id: str, text: str) -> str:
    lines = text.splitlines(keepends=True)
    if not lines:
        return f"[{model_id}] {text}"
    return "".join(f"[{model_id}] {line}" for line in lines)


def display_model_path(relative_path: Path) -> tuple[str, bool, bool]:
    match = SHARDED_GGUF_RE.match(relative_path.name)
    if not match:
        return str(relative_path), False, False

    display_name = f"{match.group('base')}.gguf"
    display_path = relative_path.with_name(display_name)
    is_first_shard = match.group("index") == "00001"
    return str(display_path), True, is_first_shard


def is_mmproj_file(path: Path) -> bool:
    return path.name.lower().startswith("mmproj") and path.suffix.lower() == ".gguf"


def find_mmproj_for_model(model: Path) -> Path | None:
    candidates = sorted(
        path.resolve()
        for path in model.parent.glob("mmproj*.gguf")
        if path.is_file()
    )
    if not candidates:
        return None
    return candidates[0]


def _gguf_read_exact(handle: Any, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("unexpected end of GGUF metadata")
    return data


def _gguf_read_u32(handle: Any) -> int:
    return int(struct.unpack("<I", _gguf_read_exact(handle, 4))[0])


def _gguf_read_u64(handle: Any) -> int:
    return int(struct.unpack("<Q", _gguf_read_exact(handle, 8))[0])


def _gguf_read_string(handle: Any) -> str:
    length = _gguf_read_u64(handle)
    if length > 16 * 1024 * 1024:
        raise ValueError("GGUF string metadata is too large")
    return _gguf_read_exact(handle, length).decode("utf-8")


def _gguf_skip_value(handle: Any, value_type: int) -> None:
    scalar_format = GGUF_SCALAR_FORMATS.get(value_type)
    if scalar_format is not None:
        handle.seek(struct.calcsize(scalar_format), os.SEEK_CUR)
        return
    if value_type == 8:
        handle.seek(_gguf_read_u64(handle), os.SEEK_CUR)
        return
    if value_type != 9:
        raise ValueError(f"unsupported GGUF metadata type: {value_type}")

    item_type = _gguf_read_u32(handle)
    item_count = _gguf_read_u64(handle)
    item_format = GGUF_SCALAR_FORMATS.get(item_type)
    if item_format is not None:
        handle.seek(struct.calcsize(item_format) * item_count, os.SEEK_CUR)
        return
    if item_type == 8:
        for _ in range(item_count):
            handle.seek(_gguf_read_u64(handle), os.SEEK_CUR)
        return
    raise ValueError(f"unsupported GGUF array metadata type: {item_type}")


def _gguf_read_value(handle: Any, value_type: int) -> Any:
    if value_type == 8:
        return _gguf_read_string(handle)
    scalar_format = GGUF_SCALAR_FORMATS.get(value_type)
    if scalar_format is None:
        raise ValueError(f"unsupported GGUF metadata value type: {value_type}")
    size = struct.calcsize(scalar_format)
    return struct.unpack(f"<{scalar_format}", _gguf_read_exact(handle, size))[0]


def read_gguf_metadata(model: Path) -> dict[str, Any]:
    stat = model.stat()
    cache_key = (str(model), stat.st_size, stat.st_mtime_ns)
    cached = GGUF_METADATA_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    for old_key in [key for key in GGUF_METADATA_CACHE if key[0] == str(model) and key != cache_key]:
        GGUF_METADATA_CACHE.pop(old_key, None)

    result: dict[str, Any] = {
        "architecture": None,
        "pooling": None,
        "embedding_dimensions": None,
        "detected_mode": "chat",
        "metadata_error": None,
    }
    try:
        with model.open("rb") as handle:
            if _gguf_read_exact(handle, 4) != b"GGUF":
                raise ValueError("invalid GGUF magic")
            version = _gguf_read_u32(handle)
            if version not in (2, 3):
                raise ValueError(f"unsupported GGUF version: {version}")
            _gguf_read_u64(handle)
            metadata_count = _gguf_read_u64(handle)

            architecture: str | None = None
            for _ in range(metadata_count):
                key = _gguf_read_string(handle)
                value_type = _gguf_read_u32(handle)
                if key == "tokenizer.ggml.tokens":
                    break

                wanted = key in ("general.architecture", "general.type")
                if architecture:
                    wanted = wanted or key in (
                        f"{architecture}.pooling_type",
                        f"{architecture}.embedding_length",
                    )
                wanted = wanted or key.endswith(".pooling_type") or key.endswith(".embedding_length")
                if wanted:
                    value = _gguf_read_value(handle, value_type)
                    if key == "general.architecture":
                        architecture = str(value)
                        result["architecture"] = architecture
                    elif key.endswith(".pooling_type"):
                        result["pooling"] = GGUF_POOLING_NAMES.get(int(value), f"unknown:{value}")
                    elif key.endswith(".embedding_length"):
                        result["embedding_dimensions"] = int(value)
                else:
                    _gguf_skip_value(handle, value_type)

        pooling = result["pooling"]
        architecture_name = str(result["architecture"] or "").lower()
        if pooling == "rank":
            result["detected_mode"] = "rerank"
        elif pooling in ("mean", "cls", "last") or "embed" in architecture_name:
            result["detected_mode"] = "embeddings"
    except (OSError, UnicodeError, ValueError, struct.error) as exc:
        result["metadata_error"] = str(exc)

    GGUF_METADATA_CACHE[cache_key] = dict(result)
    return result


def validate_model_mode(value: Any) -> str:
    mode = "auto" if value in (None, "") else str(value)
    if mode not in MODEL_MODES:
        raise ValueError("mode must be auto, chat, or embeddings")
    return mode


def validate_pooling(value: Any) -> str:
    pooling = "auto" if value in (None, "") else str(value)
    if pooling not in POOLING_TYPES:
        raise ValueError("pooling must be auto, mean, cls, or last")
    return pooling


def effective_model_config(model: Path, settings: dict[str, Any]) -> dict[str, Any]:
    metadata = read_gguf_metadata(model)
    configured_mode = validate_model_mode(settings.get("mode"))
    configured_pooling = validate_pooling(settings.get("pooling"))
    effective_mode = metadata["detected_mode"] if configured_mode == "auto" else configured_mode
    detected_pooling = metadata["pooling"]
    effective_pooling = detected_pooling if configured_pooling == "auto" else configured_pooling

    if effective_mode == "embeddings" and effective_pooling not in ("mean", "cls", "last"):
        raise ValueError(
            "pooling must be set to mean, cls, or last when an embedding model has no usable GGUF pooling metadata"
        )
    if effective_mode != "embeddings":
        effective_pooling = None

    return {
        **metadata,
        "configured_mode": configured_mode,
        "configured_pooling": configured_pooling,
        "effective_mode": effective_mode,
        "effective_pooling": effective_pooling,
        "capabilities": [effective_mode],
    }


def normalize_backend_settings(model_id: str, model_path: Path, settings: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(settings)
    normalized["model"] = model_id
    normalized["mode"] = validate_model_mode(normalized.get("mode"))
    normalized["pooling"] = validate_pooling(normalized.get("pooling"))
    config = effective_model_config(model_path, normalized)
    normalized["effective_mode"] = config["effective_mode"]
    normalized["effective_pooling"] = config["effective_pooling"]
    return normalized


def model_options() -> list[dict[str, Any]]:
    root = MODEL_DIR.resolve()
    if not root.exists():
        return []

    models: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.gguf"), key=lambda item: str(item).lower()):
        if is_mmproj_file(path):
            continue
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            relative = Path(path.name)
        display_path, is_sharded, is_first_shard = display_model_path(Path(relative))
        if is_sharded and not is_first_shard:
            continue
        stat = resolved.stat()
        saved_settings = registry.saved_settings.get(str(relative), {}) if "registry" in globals() else {}
        try:
            config = effective_model_config(resolved, saved_settings)
        except ValueError as exc:
            metadata = read_gguf_metadata(resolved)
            config = {
                **metadata,
                "configured_mode": str(saved_settings.get("mode") or "auto"),
                "configured_pooling": str(saved_settings.get("pooling") or "auto"),
                "effective_mode": metadata["detected_mode"],
                "effective_pooling": metadata["pooling"],
                "capabilities": [metadata["detected_mode"]],
                "configuration_error": str(exc),
            }
        models.append(
            {
                "name": path.name,
                "path": str(resolved),
                "relative_path": str(relative),
                "display_name": display_path,
                "size_bytes": stat.st_size,
                "mmproj_path": str(find_mmproj_for_model(resolved) or ""),
                "is_sharded": is_sharded,
                **config,
            }
        )
    return models


def model_lookup() -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for item in model_options():
        for key in (
            item["relative_path"],
            item["path"],
            item["display_name"],
            item["name"],
        ):
            if key and key not in lookup:
                lookup[str(key)] = item
    return lookup


def resolve_model_reference(model: Any) -> tuple[str, Path] | None:
    if model in (None, "", "local"):
        return None
    if not isinstance(model, str):
        return None

    lookup = model_lookup()
    item = lookup.get(model)
    if item is not None:
        return str(item["relative_path"]), Path(str(item["path"]))

    root = MODEL_DIR.resolve()
    candidate = Path(model).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()

    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return None

    if not candidate.exists() or not candidate.is_file() or candidate.suffix.lower() != ".gguf":
        return None
    if is_mmproj_file(candidate):
        return None

    return str(relative), candidate


def resolve_model_reference_required(model: Any) -> tuple[str, Path]:
    resolved = resolve_model_reference(model)
    if resolved is None:
        raise ValueError("model must be a GGUF file under MODEL_DIR")
    return resolved


def grammar_options() -> list[dict[str, Any]]:
    root = GRAMMAR_DIR.resolve()
    if not root.exists():
        return []

    grammars: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.gbnf"), key=lambda item: str(item).lower()):
        if not path.is_file():
            continue
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        stat = resolved.stat()
        grammars.append(
            {
                "name": path.name,
                "path": str(resolved),
                "relative_path": str(relative),
                "size_bytes": stat.st_size,
            }
        )
    return grammars


def validate_model_path(model: str) -> Path:
    if not model:
        raise ValueError("model is required")

    root = MODEL_DIR.resolve()
    candidate = Path(model).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()

    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("model must be under MODEL_DIR") from exc

    if not candidate.exists() or not candidate.is_file():
        raise ValueError("model file does not exist")
    if candidate.suffix.lower() != ".gguf":
        raise ValueError("model must be a .gguf file")
    return candidate


def optional_int(settings: dict[str, Any], key: str) -> int | None:
    value = settings.get(key)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc


def optional_gpu_layers(settings: dict[str, Any]) -> str | None:
    value = settings.get("gpu_layers")
    if value in (None, ""):
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "auto":
            return None
        if normalized == "all":
            return "all"
        if not re.fullmatch(r"\d+", normalized):
            raise ValueError("gpu_layers must be auto, all, or a non-negative integer")
        return normalized
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("gpu_layers must be auto, all, or a non-negative integer")
    return str(value)


def optional_bool(settings: dict[str, Any], key: str, default: bool) -> bool:
    value = settings.get(key)
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("1", "true", "on", "yes")
    return bool(value)


def build_llama_command(settings: dict[str, Any], *, model: Path, port: int) -> list[str]:
    llama_server = (LLAMA_BIN_DIR / "llama-server").resolve()
    if not llama_server.exists():
        raise ValueError(f"llama-server not found: {llama_server}")
    if not os.access(llama_server, os.X_OK):
        raise ValueError(f"llama-server is not executable: {llama_server}")

    command = [
        str(llama_server),
        "--host",
        BACKEND_HOST,
        "--port",
        str(port),
        "--model",
        str(model),
        "--log-colors",
        "off",
    ]
    mmproj = find_mmproj_for_model(model)
    if optional_bool(settings, "mmproj_enabled", True) and mmproj is not None:
        command.extend(["--mmproj", str(mmproj)])

    if settings.get("effective_mode") == "embeddings":
        command.append("--embeddings")
        pooling = settings.get("effective_pooling")
        if pooling:
            command.extend(["--pooling", str(pooling)])

    option_map = {
        "ctx_size": "--ctx-size",
        "threads": "--threads",
        "batch_size": "--batch-size",
        "ubatch_size": "--ubatch-size",
        "parallel": "--parallel",
    }
    for key, flag in option_map.items():
        value = optional_int(settings, key)
        if value is not None:
            command.extend([flag, str(value)])

    gpu_layers = optional_gpu_layers(settings)
    if gpu_layers is not None:
        command.extend(["--n-gpu-layers", gpu_layers])

    flash_attn = settings.get("flash_attn")
    if flash_attn in ("on", "off", "auto"):
        command.extend(["--flash-attn", str(flash_attn)])

    reasoning = settings.get("reasoning")
    if reasoning in ("auto", "on", "off"):
        command.extend(["--reasoning", str(reasoning)])

    reasoning_format = settings.get("reasoning_format")
    if reasoning_format in ("auto", "none", "deepseek", "deepseek-legacy"):
        command.extend(["--reasoning-format", str(reasoning_format)])

    return command


def is_tcp_port_available(host: str, port: int) -> bool:
    bind_host = host if host not in ("0.0.0.0", "::") else "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((bind_host, port))
        except OSError:
            return False
    return True


async def check_backend_health(backend_url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            response = await client.get(f"{backend_url}/health")
            if response.status_code != 200:
                return False
            try:
                payload = response.json()
            except json.JSONDecodeError:
                return False
            return isinstance(payload, dict) and payload.get("status") == "ok"
    except httpx.HTTPError:
        return False


async def wait_for_backend(backend_url: str, timeout_seconds: float = MODEL_LOAD_TIMEOUT_SECONDS) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if await check_backend_health(backend_url):
            return True
        await asyncio.sleep(0.5)
    return False


def expand_grammar_file(payload: dict[str, Any]) -> JSONResponse | None:
    grammar_file = payload.pop("grammar_file", None)
    if grammar_file in (None, ""):
        return None
    if not isinstance(grammar_file, str):
        return openai_error_response(
            "grammar_file must be a string",
            param="grammar_file",
            code="invalid_grammar_file",
        )

    raw_path = Path(grammar_file)
    if raw_path.is_absolute() or ".." in raw_path.parts:
        return openai_error_response(
            "grammar_file must be inside ./grammars/",
            param="grammar_file",
            code="invalid_grammar_file",
        )

    candidate = (GRAMMAR_DIR / raw_path).resolve()
    try:
        candidate.relative_to(GRAMMAR_DIR)
    except ValueError:
        return openai_error_response(
            "grammar_file must be inside ./grammars/",
            param="grammar_file",
            code="invalid_grammar_file",
        )

    if not candidate.exists() or not candidate.is_file():
        return openai_error_response(
            f"grammar_file not found: {grammar_file}",
            param="grammar_file",
            code="grammar_file_not_found",
        )

    payload["grammar"] = candidate.read_text(encoding="utf-8")
    return None


def apply_default_max_tokens(payload: dict[str, Any], uses_grammar: bool) -> None:
    if payload.get("max_tokens") is not None:
        return
    payload["max_tokens"] = GRAMMAR_DEFAULT_MAX_TOKENS if uses_grammar else DEFAULT_MAX_TOKENS


def forward_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}
    for name in ("authorization", "x-request-id"):
        value = request.headers.get(name)
        if value:
            headers[name] = value
    return headers


def backend_error_response(status_code: int, body: bytes) -> JSONResponse:
    message = body.decode("utf-8", errors="replace").strip()
    if not message:
        message = f"Backend returned HTTP {status_code}"
    try:
        parsed = json.loads(message)
        if isinstance(parsed, dict) and "error" in parsed:
            return JSONResponse(status_code=status_code, content=parsed)
    except json.JSONDecodeError:
        pass
    return openai_error_response(
        message,
        status_code=status_code,
        error_type="backend_error",
        code="backend_error",
    )


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(WEB_UI)


@app.get("/api/models")
async def api_models(authorization: str | None = Header(default=None)) -> JSONResponse:
    auth_error = require_auth(authorization)
    if auth_error:
        return auth_error
    return JSONResponse(
        {
            "models": model_options(),
            "model_dir": str(MODEL_DIR.resolve()),
            "saved_settings": await registry.saved_settings_snapshot(),
            "recent_models": await registry.recent_model_ids_snapshot(),
        }
    )


@app.get("/api/grammars")
async def api_grammars(authorization: str | None = Header(default=None)) -> JSONResponse:
    auth_error = require_auth(authorization)
    if auth_error:
        return auth_error
    return JSONResponse({"grammars": grammar_options(), "grammar_dir": str(GRAMMAR_DIR.resolve())})


@app.get("/api/status")
async def api_status(authorization: str | None = Header(default=None)) -> JSONResponse:
    auth_error = require_auth(authorization)
    if auth_error:
        return auth_error
    return JSONResponse(await registry.status())


@app.get("/api/logs")
async def api_logs(request: Request, authorization: str | None = Header(default=None)) -> JSONResponse:
    auth_error = require_auth_or_query(request, authorization)
    if auth_error:
        return auth_error
    store = await registry.logs_for(request.query_params.get("model"))
    if store is None:
        return openai_error_response(
            "No logs for model",
            status_code=404,
            param="model",
            code="model_logs_not_found",
        )
    return JSONResponse(await store.snapshot())


@app.get("/api/logs/stream")
async def api_logs_stream(request: Request, authorization: str | None = Header(default=None)):
    auth_error = require_auth_or_query(request, authorization)
    if auth_error:
        return auth_error
    store = await registry.logs_for(request.query_params.get("model"))
    if store is None:
        return openai_error_response(
            "No logs for model",
            status_code=404,
            param="model",
            code="model_logs_not_found",
        )

    async def stream_logs() -> Any:
        queue = await store.subscribe()
        try:
            snapshot = await store.snapshot()
            yield sse_event("snapshot", snapshot)
            yield sse_event("state", snapshot["load"])
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield sse_event(str(item["event"]), item["data"])
        finally:
            await store.unsubscribe(queue)

    return StreamingResponse(
        stream_logs(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/start")
async def api_start(request: Request, authorization: str | None = Header(default=None)) -> JSONResponse:
    auth_error = require_auth(authorization)
    if auth_error:
        return auth_error
    try:
        settings = await request.json()
        if not isinstance(settings, dict):
            return openai_error_response("Request body must be a JSON object", code="invalid_json")
        result = await registry.start(settings)
        if isinstance(result, JSONResponse):
            return result
        return JSONResponse(result)
    except ValueError as exc:
        return openai_error_response(str(exc), param="model", code="invalid_backend_settings")
    except OSError as exc:
        return openai_error_response(
            f"Failed to start llama-server: {exc}",
            status_code=500,
            error_type="backend_start_error",
            code="backend_start_failed",
        )


@app.post("/api/stop")
async def api_stop(request: Request, authorization: str | None = Header(default=None)) -> JSONResponse:
    auth_error = require_auth(authorization)
    if auth_error:
        return auth_error
    try:
        body = await request.body()
        settings = json.loads(body) if body else {}
        if not isinstance(settings, dict):
            return openai_error_response("Request body must be a JSON object", code="invalid_json")
        return JSONResponse(await registry.stop(settings))
    except json.JSONDecodeError:
        return openai_error_response("Request body must be valid JSON", code="invalid_json")
    except ValueError as exc:
        return openai_error_response(str(exc), param="model", code="invalid_backend_settings")


@app.post("/api/restart")
async def api_restart(request: Request, authorization: str | None = Header(default=None)) -> JSONResponse:
    auth_error = require_auth(authorization)
    if auth_error:
        return auth_error
    try:
        settings = await request.json()
        if not isinstance(settings, dict):
            return openai_error_response("Request body must be a JSON object", code="invalid_json")
        result = await registry.restart(settings)
        if isinstance(result, JSONResponse):
            return result
        return JSONResponse(result)
    except ValueError as exc:
        return openai_error_response(str(exc), param="model", code="invalid_backend_settings")
    except OSError as exc:
        return openai_error_response(
            f"Failed to start llama-server: {exc}",
            status_code=500,
            error_type="backend_start_error",
            code="backend_start_failed",
        )


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request, authorization: str | None = Header(default=None)):
    auth_error = require_auth(authorization)
    if auth_error:
        return auth_error

    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return openai_error_response("Request body must be valid JSON", code="invalid_json")

    if not isinstance(payload, dict):
        return openai_error_response("Request body must be a JSON object", code="invalid_json")

    uses_grammar = bool(payload.get("grammar") or payload.get("grammar_file"))
    grammar_error = expand_grammar_file(payload)
    if grammar_error:
        return grammar_error

    apply_default_max_tokens(payload, uses_grammar)

    instance, route_error = await registry.backend_for_request(payload.get("model"), purpose="chat")
    if route_error:
        return route_error
    if instance is None:
        return openai_error_response(
            "No backend selected",
            status_code=503,
            error_type="backend_connection_error",
            code="backend_unavailable",
        )
    payload["model"] = instance.model_id

    stream = bool(payload.get("stream", False))
    backend_endpoint = f"{instance.backend_url}/v1/chat/completions"

    if not stream:
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(backend_endpoint, json=payload, headers=forward_headers(request))
        except httpx.ConnectError:
            return openai_error_response(
                f"Could not connect to llama-server at {instance.backend_url}",
                status_code=503,
                error_type="backend_connection_error",
                code="backend_unavailable",
            )
        except httpx.HTTPError as exc:
            return openai_error_response(
                f"Backend request failed: {exc}",
                status_code=502,
                error_type="backend_error",
                code="backend_error",
            )

        if response.status_code >= 400:
            return backend_error_response(response.status_code, response.content)
        try:
            response_json = response.json()
        except json.JSONDecodeError:
            return openai_error_response(
                "Backend returned a non-JSON response",
                status_code=502,
                error_type="backend_error",
                code="backend_non_json_response",
            )
        return JSONResponse(status_code=response.status_code, content=response_json)

    client = httpx.AsyncClient(timeout=None)
    try:
        backend_request = client.build_request(
            "POST",
            backend_endpoint,
            json=payload,
            headers=forward_headers(request),
        )
        backend_response = await client.send(backend_request, stream=True)
    except httpx.ConnectError:
        await client.aclose()
        return openai_error_response(
            f"Could not connect to llama-server at {instance.backend_url}",
            status_code=503,
            error_type="backend_connection_error",
            code="backend_unavailable",
        )
    except httpx.HTTPError as exc:
        await client.aclose()
        return openai_error_response(
            f"Backend request failed: {exc}",
            status_code=502,
            error_type="backend_error",
            code="backend_error",
        )

    if backend_response.status_code >= 400:
        body = await backend_response.aread()
        await backend_response.aclose()
        await client.aclose()
        return backend_error_response(backend_response.status_code, body)

    async def stream_backend() -> Any:
        try:
            async for chunk in backend_response.aiter_raw():
                yield chunk
        finally:
            await backend_response.aclose()
            await client.aclose()

    media_type = backend_response.headers.get("content-type", "text/event-stream")
    return StreamingResponse(
        stream_backend(),
        status_code=backend_response.status_code,
        media_type=media_type,
    )


@app.post("/v1/embeddings")
async def embeddings(request: Request, authorization: str | None = Header(default=None)) -> JSONResponse:
    auth_error = require_auth(authorization)
    if auth_error:
        return auth_error

    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return openai_error_response("Request body must be valid JSON", code="invalid_json")
    if not isinstance(payload, dict):
        return openai_error_response("Request body must be a JSON object", code="invalid_json")
    if "dimensions" in payload:
        return openai_error_response(
            "dimensions is not supported by this llama-server",
            param="dimensions",
            code="unsupported_parameter",
        )
    if payload.get("stream"):
        return openai_error_response(
            "stream is not supported by /v1/embeddings",
            param="stream",
            code="unsupported_parameter",
        )

    instance, route_error = await registry.backend_for_request(payload.get("model"), purpose="embeddings")
    if route_error:
        return route_error
    if instance is None:
        return openai_error_response(
            "No embedding backend selected",
            status_code=503,
            error_type="backend_connection_error",
            code="backend_unavailable",
        )
    payload["model"] = instance.model_id

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"{instance.backend_url}/v1/embeddings",
                json=payload,
                headers=forward_headers(request),
            )
    except httpx.ConnectError:
        return openai_error_response(
            f"Could not connect to llama-server at {instance.backend_url}",
            status_code=503,
            error_type="backend_connection_error",
            code="backend_unavailable",
        )
    except httpx.HTTPError as exc:
        return openai_error_response(
            f"Backend request failed: {exc}",
            status_code=502,
            error_type="backend_error",
            code="backend_error",
        )

    if response.status_code >= 400:
        return backend_error_response(response.status_code, response.content)
    try:
        response_json = response.json()
    except json.JSONDecodeError:
        return openai_error_response(
            "Backend returned a non-JSON response",
            status_code=502,
            error_type="backend_error",
            code="backend_non_json_response",
        )
    return JSONResponse(status_code=response.status_code, content=response_json)


@app.get("/v1/models")
async def v1_models(authorization: str | None = Header(default=None)) -> JSONResponse:
    auth_error = require_auth(authorization)
    if auth_error:
        return auth_error
    data = [
        {
            "id": model["relative_path"],
            "object": "model",
            "created": 0,
            "owned_by": "local",
            "architecture": model["architecture"],
            "pooling": model["effective_pooling"],
            "embedding_dimensions": model["embedding_dimensions"],
            "detected_mode": model["detected_mode"],
            "effective_mode": model["effective_mode"],
            "capabilities": model["capabilities"],
        }
        for model in model_options()
    ]
    return JSONResponse({"object": "list", "data": data})


WEB_UI = r"""
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Local LLM Server</title>
  <style>
    :root {
      --bg: #f6f7f4;
      --panel: #ffffff;
      --ink: #18201b;
      --muted: #66736a;
      --line: #d9ded8;
      --accent: #1f7a65;
      --warn: #b76b25;
      --danger: #a33a2d;
      --code: #101816;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background: var(--bg);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      width: min(1240px, calc(100% - 28px));
      margin: 0 auto;
      padding: 24px 0 360px;
    }
    body.logs-collapsed main {
      padding-bottom: 92px;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }
    .header-controls {
      display: flex;
      align-items: end;
      justify-content: flex-end;
      gap: 12px;
      min-width: 0;
    }
    .status-menu {
      position: relative;
      flex: 0 0 auto;
    }
    .status-toggle {
      width: 38px;
      height: 38px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0;
      background: #fff;
    }
    .status-toggle .dot {
      width: 13px;
      height: 13px;
    }
    .status-toggle[aria-expanded="true"] {
      outline: 2px solid rgba(31, 122, 101, 0.22);
      outline-offset: 2px;
    }
    .status-json-popover {
      position: absolute;
      right: 0;
      top: calc(100% + 8px);
      z-index: 60;
      width: min(520px, calc(100vw - 28px));
      max-height: 420px;
      box-shadow: 0 12px 36px rgba(16, 24, 22, 0.18);
    }
    .status-json-popover[hidden] {
      display: none;
    }
    h1, h2 {
      margin: 0;
      letter-spacing: 0;
    }
    h1 { font-size: 1.55rem; }
    h2 { font-size: 1rem; }
    .layout {
      display: grid;
      gap: 14px;
      min-width: 0;
    }
    .panel {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 16px;
      overflow: hidden;
    }
    .stack { display: grid; gap: 14px; min-width: 0; }
    .settings {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      min-width: 0;
    }
    label {
      display: block;
      margin: 0 0 5px;
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 700;
    }
    input, select {
      width: 100%;
      min-width: 0;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      background: #fff;
      color: var(--ink);
      font: 0.9rem ui-sans-serif, system-ui, sans-serif;
    }
    .field { min-width: 0; }
    .check-row {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 0.88rem;
      font-weight: 700;
    }
    .check-row input {
      width: 16px;
      min-height: 16px;
      height: 16px;
      padding: 0;
      accent-color: var(--accent);
    }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 14px;
    }
    button {
      border: 0;
      border-radius: 6px;
      padding: 9px 12px;
      color: #fff;
      background: var(--accent);
      font: 700 0.86rem ui-sans-serif, system-ui, sans-serif;
      cursor: pointer;
    }
    button:hover { filter: brightness(1.04); }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.58;
      filter: none;
    }
    button.secondary { background: var(--warn); }
    button.danger { background: var(--danger); }
    button.neutral {
      color: var(--ink);
      background: #eef1ed;
      border: 1px solid var(--line);
    }
    button.compact {
      padding: 6px 8px;
      font-size: 0.78rem;
    }
    .meta {
      margin: 9px 0 0;
      color: var(--muted);
      font-size: 0.82rem;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }
    .summary {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 12px;
      min-width: 0;
    }
    .metric {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfcfa;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 0.72rem;
      font-weight: 700;
    }
    .metric strong {
      display: block;
      margin-top: 3px;
      font-size: 0.95rem;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    table {
      width: 100%;
      min-width: 0;
      table-layout: fixed;
      border-collapse: collapse;
      font-size: 0.84rem;
    }
    th:nth-child(1), td:nth-child(1) { width: 42%; }
    th:nth-child(2), td:nth-child(2) { width: 16%; }
    th:nth-child(3), td:nth-child(3) { width: 8%; }
    th:nth-child(4), td:nth-child(4) { width: 8%; }
    th:nth-child(5), td:nth-child(5) { width: 26%; }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 9px 6px;
      text-align: left;
      vertical-align: top;
      min-width: 0;
    }
    th {
      color: var(--muted);
      font-size: 0.72rem;
      text-transform: uppercase;
      white-space: nowrap;
    }
    .model-cell {
      overflow-wrap: anywhere;
      word-break: break-word;
      font-weight: 700;
    }
    .table-wrap {
      overflow: auto;
      min-width: 0;
    }
    .backend-table th:nth-child(1), .backend-table td:nth-child(1) { width: 43%; }
    .backend-table th:nth-child(2), .backend-table td:nth-child(2) { width: 10%; }
    .backend-table th:nth-child(3), .backend-table td:nth-child(3) { width: 11%; }
    .backend-table th:nth-child(4), .backend-table td:nth-child(4) { width: 7%; }
    .backend-table th:nth-child(5), .backend-table td:nth-child(5) { width: 8%; }
    .backend-table th:nth-child(6), .backend-table td:nth-child(6) { width: 21%; }
    .backend-table td:last-child {
      white-space: nowrap;
    }
    .backend-table button.compact {
      min-width: 0;
    }
    .backend-table td:last-child button {
      margin: 0 3px 0 0;
    }
    .recent-table th:nth-child(1), .recent-table td:nth-child(1) { width: 66%; }
    .recent-table th:nth-child(2), .recent-table td:nth-child(2) { width: 14%; }
    .recent-table th:nth-child(3), .recent-table td:nth-child(3) { width: 20%; }
    .recent-table td:last-child {
      white-space: nowrap;
    }
    .recent-table td:last-child button {
      margin: 0 3px 0 0;
    }
    .model-list-wrap {
      max-height: 520px;
      overflow: auto;
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .model-list-wrap table {
      min-width: 980px;
    }
    .model-table th:nth-child(1), .model-table td:nth-child(1) { width: 42%; }
    .model-table th:nth-child(2), .model-table td:nth-child(2) { width: 11%; }
    .model-table th:nth-child(3), .model-table td:nth-child(3) { width: 11%; }
    .model-table th:nth-child(4), .model-table td:nth-child(4) { width: 11%; }
    .model-table th:nth-child(5), .model-table td:nth-child(5) { width: 10%; }
    .model-table th:nth-child(6), .model-table td:nth-child(6) { width: 15%; }
    .model-list-wrap tbody tr {
      cursor: pointer;
    }
    .model-list-wrap tbody tr:hover {
      background: #f3f6f2;
    }
    .model-list-wrap tbody tr.selected {
      background: #e8f3ee;
      box-shadow: inset 3px 0 0 var(--accent);
    }
    .subtext {
      display: block;
      margin-top: 3px;
      color: var(--muted);
      font-size: 0.74rem;
      font-weight: 600;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      border-radius: 999px;
      padding: 2px 8px;
      color: var(--muted);
      background: #eef1ed;
      border: 1px solid var(--line);
      font-size: 0.72rem;
      font-weight: 800;
      white-space: nowrap;
    }
    .pill.ok {
      color: #16644f;
      background: #e5f4ee;
      border-color: #b9ddce;
    }
    .pill.warn {
      color: #8a4b16;
      background: #fff3e4;
      border-color: #ecd1aa;
    }
    .pill.missing {
      color: #8a2930;
      background: #fdebec;
      border-color: #efc4c8;
    }
    td:last-child {
      white-space: nowrap;
    }
    .state {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      max-width: 100%;
      flex-wrap: nowrap;
      white-space: nowrap;
      font-weight: 700;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: #8c948f;
    }
    .dot.ready { background: #1f9d63; }
    .dot.loading { background: #d96b2b; }
    .dot.error { background: #a33a2d; }
    pre {
      overflow: auto;
      margin: 0;
      border-radius: 8px;
      padding: 12px;
      background: var(--code);
      color: #eff7eb;
      font-size: 0.78rem;
      line-height: 1.45;
    }
    .logs {
      height: 250px;
      min-height: 160px;
      max-height: min(42vh, 420px);
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .log-panel {
      position: fixed;
      left: max(14px, calc((100vw - 1240px) / 2));
      right: max(14px, calc((100vw - 1240px) / 2));
      bottom: 10px;
      z-index: 30;
      box-shadow: 0 12px 36px rgba(16, 24, 22, 0.18);
    }
    .log-panel.collapsed {
      padding: 12px 16px;
    }
    .log-panel.collapsed .logs,
    .log-panel.collapsed .log-select,
    .log-panel.collapsed .log-auto-scroll,
    .log-panel.collapsed #clearLogsBtn {
      display: none;
    }
    .log-panel.collapsed .toolbar {
      margin-bottom: 0;
    }
    .toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 12px;
      min-width: 0;
    }
    .toolbar h2 {
      flex: 0 0 auto;
    }
    .toolbar .field { flex: 1; }
    .status-line {
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 700;
      white-space: nowrap;
    }
    @media (max-width: 900px) {
      .settings, .summary { grid-template-columns: 1fr; }
      header { align-items: flex-start; flex-direction: column; }
      .header-controls {
        width: 100%;
        align-items: end;
        justify-content: space-between;
      }
      .status-json-popover {
        left: 0;
        right: auto;
      }
      main { width: min(100% - 18px, 1240px); padding: 14px 0 340px; }
      body.logs-collapsed main { padding-bottom: 88px; }
      .log-panel {
        left: 9px;
        right: 9px;
        bottom: 8px;
      }
      .toolbar {
        flex-wrap: wrap;
      }
      table { font-size: 0.78rem; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Local LLM Server</h1>
      <div class="header-controls">
        <div class="status-menu">
          <button id="statusToggle" class="status-toggle" type="button" aria-expanded="false" title="Status JSON">
            <span id="proxyStatusDot" class="dot"></span>
          </button>
          <pre id="statusJson" class="status-json-popover" hidden>{}</pre>
        </div>
        <div class="field" style="max-width: 320px">
          <label for="apiKey">Proxy API Key</label>
          <input id="apiKey" type="password" placeholder="PROXY_API_KEY">
        </div>
      </div>
    </header>

    <section class="layout">
      <section class="panel">
        <h2>Running Models</h2>
        <div class="summary" style="margin-top: 12px">
          <div class="metric"><span>Active</span><strong id="activeCount">0</strong></div>
          <div class="metric"><span>Latest</span><strong id="latestModel">none</strong></div>
          <div class="metric"><span>Start Port</span><strong id="startPort">-</strong></div>
        </div>
        <div class="table-wrap">
          <table class="backend-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Mode</th>
                <th>State</th>
                <th>Port</th>
                <th>Uptime</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody id="backendRows"></tbody>
          </table>
        </div>
        <div class="actions">
          <button id="stopAllBtn" class="danger">Stop All</button>
        </div>
      </section>

      <section class="panel">
        <h2>Recent Models</h2>
        <div class="table-wrap" style="margin-top: 12px">
          <table class="recent-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>State</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody id="recentRows"></tbody>
          </table>
        </div>
      </section>

      <section class="panel">
        <h2>Model Catalog</h2>
        <div class="field" style="margin-top: 12px">
          <label for="modelFilter">Filter</label>
          <input id="modelFilter" type="search" placeholder="name or path">
        </div>
        <div class="model-list-wrap">
          <table class="model-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Size</th>
                <th>Mode</th>
                <th>MMProj</th>
                <th>Saved</th>
                <th>State</th>
              </tr>
            </thead>
            <tbody id="modelRows"></tbody>
          </table>
        </div>
        <label class="check-row" for="mmproj_enabled" style="margin-top: 10px">
          <input id="mmproj_enabled" type="checkbox" checked>
          Use MMProj
        </label>
        <p class="meta" id="mmprojMeta"></p>
        <div class="settings" style="margin-top: 12px">
          <div class="field">
            <label for="mode">Mode</label>
            <select id="mode">
              <option value="auto">auto</option>
              <option value="chat">chat</option>
              <option value="embeddings">embeddings</option>
            </select>
          </div>
          <div class="field">
            <label for="pooling">Pooling</label>
            <select id="pooling">
              <option value="auto">auto</option>
              <option value="mean">mean</option>
              <option value="cls">cls</option>
              <option value="last">last</option>
            </select>
          </div>
          <div class="field"><label for="ctx_size">Context</label><input id="ctx_size" type="number" placeholder="4096"></div>
          <div class="field">
            <label for="gpu_layers_mode">GPU Layers</label>
            <select id="gpu_layers_mode">
              <option value="auto">auto</option>
              <option value="all">all</option>
              <option value="custom">custom</option>
            </select>
            <input id="gpu_layers" type="number" min="0" step="1" placeholder="layers" hidden style="margin-top: 8px">
          </div>
          <div class="field"><label for="threads">Threads</label><input id="threads" type="number" placeholder="auto"></div>
          <div class="field"><label for="batch_size">Batch</label><input id="batch_size" type="number" placeholder="2048"></div>
          <div class="field"><label for="ubatch_size">UBatch</label><input id="ubatch_size" type="number" placeholder="512"></div>
          <div class="field"><label for="parallel">Parallel</label><input id="parallel" type="number" placeholder="auto"></div>
          <div class="field">
            <label for="flash_attn">Flash Attention</label>
            <select id="flash_attn">
              <option value="auto">auto</option>
              <option value="on">on</option>
              <option value="off">off</option>
            </select>
          </div>
          <div class="field">
            <label for="reasoning">Reasoning</label>
            <select id="reasoning">
              <option value="off">off</option>
              <option value="auto">auto</option>
              <option value="on">on</option>
            </select>
          </div>
          <div class="field">
            <label for="reasoning_format">Reasoning Format</label>
            <select id="reasoning_format">
              <option value="none">none</option>
              <option value="auto">auto</option>
              <option value="deepseek">deepseek</option>
              <option value="deepseek-legacy">deepseek-legacy</option>
            </select>
          </div>
        </div>
        <div class="actions">
          <button id="startBtn">Start</button>
          <button id="restartBtn" class="secondary">Restart</button>
          <button id="stopBtn" class="danger">Stop</button>
          <button id="refreshBtn" class="neutral">Refresh</button>
        </div>
        <p class="meta" id="modelMeta"></p>
        <p class="meta" id="messageLine"></p>
      </section>

      <section class="panel log-panel" id="logsPanel">
        <div class="toolbar">
          <h2>Logs</h2>
          <div class="field log-select">
            <label for="logModel">Source</label>
            <select id="logModel">
              <option value="">All models</option>
            </select>
          </div>
          <label class="check-row log-auto-scroll" for="autoScroll">
            <input id="autoScroll" type="checkbox" checked>
            Auto-scroll
          </label>
          <button id="clearLogsBtn" class="neutral compact">Clear</button>
          <span class="status-line" id="logStreamState">connecting</span>
          <button id="toggleLogsBtn" class="neutral compact" type="button" aria-expanded="true">Minimize</button>
        </div>
        <pre id="logsPre" class="logs"></pre>
      </section>
    </section>
  </main>

  <script>
    const apiKey = document.getElementById('apiKey');
    const statusToggle = document.getElementById('statusToggle');
    const proxyStatusDot = document.getElementById('proxyStatusDot');
    const statusJson = document.getElementById('statusJson');
    const recentRows = document.getElementById('recentRows');
    const modelRows = document.getElementById('modelRows');
    const modelFilter = document.getElementById('modelFilter');
    const mmprojEnabled = document.getElementById('mmproj_enabled');
    const mmprojMeta = document.getElementById('mmprojMeta');
    const modeInput = document.getElementById('mode');
    const poolingInput = document.getElementById('pooling');
    const gpuLayersMode = document.getElementById('gpu_layers_mode');
    const gpuLayersInput = document.getElementById('gpu_layers');
    const logsPanel = document.getElementById('logsPanel');
    const logsPre = document.getElementById('logsPre');
    const autoScroll = document.getElementById('autoScroll');
    const logModel = document.getElementById('logModel');
    const logStreamState = document.getElementById('logStreamState');
    const toggleLogsBtn = document.getElementById('toggleLogsBtn');
    const messageLine = document.getElementById('messageLine');
    let allModels = [];
    let modelDir = '';
    let savedSettings = {};
    let recentModels = [];
    let selectedModelId = localStorage.getItem('selectedModelId') || '';
    let statusData = {backends: []};
    let logSource = null;
    let logReconnectTimer = null;
    const LOG_VIEW_MAX_CHARS = 200000;

    apiKey.value = localStorage.getItem('proxyApiKey') || '';
    modelFilter.value = localStorage.getItem('modelFilter') || '';

    document.getElementById('startBtn').addEventListener('click', () => startBackend());
    document.getElementById('restartBtn').addEventListener('click', () => restartBackend());
    document.getElementById('stopBtn').addEventListener('click', () => stopBackend(selectedModelId));
    document.getElementById('stopAllBtn').addEventListener('click', () => stopAllBackends());
    document.getElementById('refreshBtn').addEventListener('click', () => refreshAll());
    document.getElementById('clearLogsBtn').addEventListener('click', () => clearLogs());
    toggleLogsBtn.addEventListener('click', () => setLogsCollapsed(!logsPanel.classList.contains('collapsed')));
    statusToggle.addEventListener('click', () => setStatusJsonOpen(statusJson.hidden));
    apiKey.addEventListener('input', () => {
      localStorage.setItem('proxyApiKey', apiKey.value);
      scheduleLogReconnect();
    });
    modelFilter.addEventListener('input', () => {
      localStorage.setItem('modelFilter', modelFilter.value);
      renderModels();
    });
    modeInput.addEventListener('change', () => updatePoolingControl());
    gpuLayersMode.addEventListener('change', () => updateGpuLayersInput(true));
    logModel.addEventListener('change', () => connectLogStream());
    document.addEventListener('click', (event) => {
      if (!statusJson.hidden && !event.target.closest('.status-menu')) {
        setStatusJsonOpen(false);
      }
    });

    setLogsCollapsed(localStorage.getItem('logsCollapsed') === 'true');

    function headers() {
      const h = {'Content-Type': 'application/json'};
      if (apiKey.value) h.Authorization = `Bearer ${apiKey.value}`;
      return h;
    }

    async function api(path, options = {}) {
      const res = await fetch(path, {...options, headers: {...headers(), ...(options.headers || {})}});
      const text = await res.text();
      let data;
      try { data = text ? JSON.parse(text) : {}; } catch { data = {raw: text}; }
      if (!res.ok) throw new Error(data?.error?.message || text || res.statusText);
      return data;
    }

    function settings() {
      if (!selectedModelId) throw new Error('No model selected.');
      const payload = {
        model: selectedModelId,
        mode: modeInput.value,
        pooling: poolingInput.value,
        mmproj_enabled: mmprojEnabled.checked && !mmprojEnabled.disabled,
        flash_attn: document.getElementById('flash_attn').value,
        reasoning: document.getElementById('reasoning').value,
        reasoning_format: document.getElementById('reasoning_format').value,
      };
      if (gpuLayersMode.value === 'all') {
        payload.gpu_layers = 'all';
      } else if (gpuLayersMode.value === 'custom') {
        if (gpuLayersInput.value === '') throw new Error('GPU Layers custom value is required.');
        payload.gpu_layers = Number(gpuLayersInput.value);
      }
      for (const key of ['ctx_size', 'threads', 'batch_size', 'ubatch_size', 'parallel']) {
        const value = document.getElementById(key).value;
        if (value !== '') payload[key] = Number(value);
      }
      return payload;
    }

    function selectedModel() {
      return allModels.find((item) => item.relative_path === selectedModelId);
    }

    function modelName(modelId) {
      const item = allModels.find((entry) => entry.relative_path === modelId);
      return item?.display_name || modelId;
    }

    function updateGpuLayersInput(shouldFocus = false) {
      const custom = gpuLayersMode.value === 'custom';
      gpuLayersInput.hidden = !custom;
      gpuLayersInput.disabled = !custom;
      if (custom && shouldFocus) gpuLayersInput.focus();
    }

    function updatePoolingControl() {
      const item = selectedModel();
      const effectiveMode = modeInput.value === 'auto' ? item?.detected_mode : modeInput.value;
      poolingInput.disabled = effectiveMode !== 'embeddings';
    }

    function hasOwn(object, key) {
      return Object.prototype.hasOwnProperty.call(object || {}, key);
    }

    function setSelectValue(id, value, fallback) {
      const input = document.getElementById(id);
      const next = value == null || value === '' ? fallback : String(value);
      input.value = [...input.options].some((option) => option.value === next) ? next : fallback;
    }

    function setNumberValue(id, settings) {
      const input = document.getElementById(id);
      input.value = hasOwn(settings, id) ? String(settings[id]) : '';
    }

    function applySelectedModelSettings() {
      const item = selectedModel();
      const settings = savedSettings[selectedModelId] || {};
      const mmprojPath = item?.mmproj_path || '';
      const hasMmproj = Boolean(mmprojPath);
      mmprojEnabled.disabled = !hasMmproj;
      mmprojEnabled.checked = hasMmproj && (hasOwn(settings, 'mmproj_enabled') ? Boolean(settings.mmproj_enabled) : true);
      mmprojMeta.textContent = hasMmproj ? `MMProj: ${mmprojPath}` : 'MMProj: none';

      for (const key of ['ctx_size', 'threads', 'batch_size', 'ubatch_size', 'parallel']) {
        setNumberValue(key, settings);
      }

      const gpuLayers = settings.gpu_layers;
      if (gpuLayers === 'all') {
        gpuLayersMode.value = 'all';
        gpuLayersInput.value = '';
      } else if (gpuLayers !== undefined && gpuLayers !== null && gpuLayers !== '') {
        gpuLayersMode.value = 'custom';
        gpuLayersInput.value = String(gpuLayers);
      } else {
        gpuLayersMode.value = 'auto';
        gpuLayersInput.value = '';
      }
      updateGpuLayersInput(false);

      setSelectValue('flash_attn', settings.flash_attn, 'auto');
      setSelectValue('reasoning', settings.reasoning, 'off');
      setSelectValue('reasoning_format', settings.reasoning_format, 'none');
      setSelectValue('mode', settings.mode, 'auto');
      setSelectValue('pooling', settings.pooling, 'auto');
      updatePoolingControl();
    }

    async function loadModels(applySettings = true) {
      const data = await api('/api/models');
      allModels = data.models || [];
      modelDir = data.model_dir || '';
      savedSettings = data.saved_settings || {};
      recentModels = data.recent_models || [];
      renderRecentModels();
      renderModels({applySettings});
    }

    function modelItem(modelId) {
      return allModels.find((item) => item.relative_path === modelId);
    }

    function renderModels(options = {}) {
      const previous = selectedModelId;
      const query = modelFilter.value.trim().toLowerCase();
      const filtered = allModels.filter((item) => {
        const haystack = `${item.display_name} ${item.relative_path} ${item.name} ${item.path}`.toLowerCase();
        return !query || haystack.includes(query);
      });

      if (!allModels.some((item) => item.relative_path === selectedModelId)) {
        selectedModelId = filtered[0]?.relative_path || allModels[0]?.relative_path || '';
      }

      modelRows.innerHTML = '';
      if (!filtered.length) {
        const tr = document.createElement('tr');
        tr.innerHTML = '<td colspan="6" style="color: var(--muted)">No matching GGUF files.</td>';
        modelRows.appendChild(tr);
      }
      for (const item of filtered) {
        const backend = backendForModel(item.relative_path);
        const dotClass = backend?.load_state === 'ready' ? 'ready' : backend?.load_state === 'loading' ? 'loading' : backend?.load_state === 'error' ? 'error' : '';
        const state = backend ? stateLabel(backend) : 'stopped';
        const tr = document.createElement('tr');
        tr.className = item.relative_path === selectedModelId ? 'selected' : '';
        tr.tabIndex = 0;
        tr.setAttribute('role', 'button');
        tr.setAttribute('aria-selected', item.relative_path === selectedModelId ? 'true' : 'false');
        tr.innerHTML = `
          <td class="model-cell">${escapeHtml(item.display_name || item.relative_path)}<span class="subtext">${escapeHtml(item.relative_path)}</span></td>
          <td>${escapeHtml(formatBytes(item.size_bytes))}</td>
          <td><span class="pill ${item.effective_mode === 'embeddings' ? 'ok' : item.effective_mode === 'rerank' ? 'warn' : ''}">${escapeHtml(item.effective_mode || 'chat')}</span><span class="subtext">${escapeHtml(item.architecture || 'unknown')}${item.effective_pooling ? ` / ${escapeHtml(item.effective_pooling)}` : ''}</span></td>
          <td><span class="pill ${item.mmproj_path ? 'ok' : ''}">${item.mmproj_path ? 'yes' : 'none'}</span></td>
          <td><span class="pill ${savedSettings[item.relative_path] ? 'ok' : 'warn'}">${savedSettings[item.relative_path] ? 'saved' : 'default'}</span></td>
          <td><span class="state"><span class="dot ${dotClass}"></span>${escapeHtml(state)}</span></td>
        `;
        tr.addEventListener('click', () => selectModel(item.relative_path));
        tr.addEventListener('keydown', (event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            selectModel(item.relative_path);
          }
        });
        modelRows.appendChild(tr);
      }

      if (selectedModelId) {
        localStorage.setItem('selectedModelId', selectedModelId);
      }
      if (options.applySettings || selectedModelId !== previous) {
        applySelectedModelSettings();
      }
      const selected = selectedModel();
      document.getElementById('modelMeta').textContent =
        `${filtered.length} / ${allModels.length} GGUF files under ${modelDir}; selected: ${selected ? (selected.display_name || selected.relative_path) : 'none'}`;
    }

    function renderRecentModels() {
      recentRows.innerHTML = '';
      if (!recentModels.length) {
        const tr = document.createElement('tr');
        tr.innerHTML = '<td colspan="3" style="color: var(--muted)">No recent models yet.</td>';
        recentRows.appendChild(tr);
        return;
      }

      for (const modelId of recentModels.slice(0, 5)) {
        const item = modelItem(modelId);
        const backend = backendForModel(modelId);
        const missing = !item;
        const running = Boolean(backend && (backend.running || backend.load_state === 'loading' || backend.load_state === 'ready'));
        const dotClass = backend?.load_state === 'ready' ? 'ready' : backend?.load_state === 'loading' ? 'loading' : backend?.load_state === 'error' ? 'error' : '';
        const state = missing ? 'missing' : backend ? stateLabel(backend) : 'stopped';
        const stateHtml = missing
          ? '<span class="pill missing">missing</span>'
          : `<span class="state"><span class="dot ${dotClass}"></span>${escapeHtml(state)}</span>`;
        const startDisabled = missing || running;
        const selectDisabled = missing;
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td class="model-cell">${escapeHtml(item?.display_name || modelId)}<span class="subtext">${escapeHtml(modelId)}</span></td>
          <td>${stateHtml}</td>
          <td>
            <button class="compact ${running ? 'neutral' : ''}" data-action="start-recent" data-model="${escapeAttr(modelId)}" ${startDisabled ? 'disabled' : ''}>${running ? 'Running' : 'Start'}</button>
            <button class="neutral compact" data-action="select-recent" data-model="${escapeAttr(modelId)}" ${selectDisabled ? 'disabled' : ''}>Select</button>
          </td>
        `;
        recentRows.appendChild(tr);
      }

      recentRows.querySelectorAll('button[data-action]').forEach((button) => {
        button.addEventListener('click', () => {
          const id = button.getAttribute('data-model');
          const action = button.getAttribute('data-action');
          if (action === 'start-recent') {
            startRecentModel(id);
          } else if (action === 'select-recent') {
            selectModel(id);
          }
        });
      });
    }

    function selectModel(modelId) {
      selectedModelId = modelId;
      localStorage.setItem('selectedModelId', selectedModelId);
      renderModels();
      applySelectedModelSettings();
    }

    function backendForModel(modelId) {
      return (statusData.backends || []).find((backend) => backend.model_id === modelId);
    }

    function formatBytes(bytes) {
      const value = Number(bytes || 0);
      if (!value) return '-';
      const units = ['B', 'KB', 'MB', 'GB', 'TB'];
      let size = value;
      let unit = 0;
      while (size >= 1024 && unit < units.length - 1) {
        size /= 1024;
        unit += 1;
      }
      return `${size >= 10 || unit === 0 ? size.toFixed(0) : size.toFixed(1)} ${units[unit]}`;
    }

    function stateLabel(backend) {
      const state = backend.load_state || (backend.running ? 'running' : 'stopped');
      const progress = backend.load_progress;
      return state === 'loading' ? `loading ${progress ?? 0}%` : state;
    }

    function setStatusJsonOpen(open) {
      statusJson.hidden = !open;
      statusToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    }

    function renderStatus(data) {
      statusData = data;
      statusJson.textContent = JSON.stringify(data, null, 2);
      proxyStatusDot.className = `dot ${data.running ? 'ready' : ''}`;
      document.getElementById('activeCount').textContent = String(data.count || 0);
      document.getElementById('latestModel').textContent = data.latest_model_id ? modelName(data.latest_model_id) : 'none';
      document.getElementById('startPort').textContent = String(data.backend_start_port ?? '-');
      renderBackends(data.backends || []);
      renderLogOptions(data.backends || []);
      renderRecentModels();
      renderModels({applySettings: false});
    }

    function renderBackends(backends) {
      const rows = document.getElementById('backendRows');
      rows.innerHTML = '';
      if (!backends.length) {
        const tr = document.createElement('tr');
        tr.innerHTML = '<td colspan="6" style="color: var(--muted)">No models have been started.</td>';
        rows.appendChild(tr);
        return;
      }
      for (const backend of backends) {
        const tr = document.createElement('tr');
        const dotClass = backend.load_state === 'ready' ? 'ready' : backend.load_state === 'loading' ? 'loading' : backend.load_state === 'error' ? 'error' : '';
        const uptime = backend.uptime_seconds == null ? '-' : `${backend.uptime_seconds}s`;
        tr.innerHTML = `
          <td class="model-cell">${escapeHtml(modelName(backend.model_id))}</td>
          <td><span class="pill ${backend.effective_mode === 'embeddings' ? 'ok' : ''}">${escapeHtml(backend.effective_mode || 'chat')}</span>${backend.effective_pooling ? `<span class="subtext">${escapeHtml(backend.effective_pooling)}</span>` : ''}</td>
          <td><span class="state"><span class="dot ${dotClass}"></span>${escapeHtml(stateLabel(backend))}</span></td>
          <td>${backend.port ?? '-'}</td>
          <td>${uptime}</td>
          <td>
            <button class="neutral compact" data-action="logs" data-model="${escapeAttr(backend.model_id)}">Logs</button>
            <button class="secondary compact" data-action="restart" data-model="${escapeAttr(backend.model_id)}">Restart</button>
            <button class="danger compact" data-action="stop" data-model="${escapeAttr(backend.model_id)}">Stop</button>
          </td>
        `;
        rows.appendChild(tr);
      }
      rows.querySelectorAll('button[data-action]').forEach((button) => {
        button.addEventListener('click', () => {
          const id = button.getAttribute('data-model');
          const action = button.getAttribute('data-action');
          if (action === 'logs') {
            logModel.value = id;
            connectLogStream();
          } else if (action === 'stop') {
            stopBackend(id);
          } else if (action === 'restart') {
            restartModel(id);
          }
        });
      });
    }

    function renderLogOptions(backends) {
      const previous = logModel.value;
      logModel.innerHTML = '<option value="">All models</option>';
      for (const backend of backends) {
        const opt = document.createElement('option');
        opt.value = backend.model_id;
        opt.textContent = modelName(backend.model_id);
        logModel.appendChild(opt);
      }
      if ([...logModel.options].some((opt) => opt.value === previous)) {
        logModel.value = previous;
      }
    }

    async function loadStatus() {
      const data = await api('/api/status');
      renderStatus(data);
    }

    async function startBackend() {
      await runAction(async () => {
        await api('/api/start', {method: 'POST', body: JSON.stringify(settings())});
      }, 'started');
    }

    async function startRecentModel(modelId) {
      await runAction(async () => {
        const payload = {...(savedSettings[modelId] || {}), model: modelId};
        await api('/api/start', {method: 'POST', body: JSON.stringify(payload)});
      }, 'started');
    }

    async function restartBackend() {
      await runAction(async () => {
        await api('/api/restart', {method: 'POST', body: JSON.stringify(settings())});
      }, 'restarted');
    }

    async function restartModel(modelId) {
      await runAction(async () => {
        const item = allModels.find((entry) => entry.relative_path === modelId);
        const payload = item && selectedModelId === modelId ? settings() : {...(savedSettings[modelId] || {}), model: modelId};
        await api('/api/restart', {method: 'POST', body: JSON.stringify(payload)});
      }, 'restarted');
    }

    async function stopBackend(modelId) {
      await runAction(async () => {
        if (!modelId) throw new Error('No model selected.');
        await api('/api/stop', {method: 'POST', body: JSON.stringify({model: modelId})});
      }, 'stopped');
    }

    async function stopAllBackends() {
      await runAction(async () => {
        await api('/api/stop', {method: 'POST', body: JSON.stringify({all: true})});
      }, 'stopped all');
    }

    async function runAction(action, label) {
      try {
        messageLine.textContent = 'working...';
        await action();
        messageLine.textContent = label;
        await loadModels(false);
        await loadStatus();
        scheduleLogReconnect();
      } catch (err) {
        messageLine.textContent = String(err);
      }
    }

    async function refreshAll() {
      try {
        await loadModels();
        await loadStatus();
      } catch (err) {
        messageLine.textContent = String(err);
      }
    }

    function appendLog(text) {
      logsPre.textContent += text;
      if (logsPre.textContent.length > LOG_VIEW_MAX_CHARS) {
        logsPre.textContent = logsPre.textContent.slice(-LOG_VIEW_MAX_CHARS);
      }
      if (autoScroll.checked) logsPre.scrollTop = logsPre.scrollHeight;
    }

    function setLogsCollapsed(collapsed) {
      document.body.classList.toggle('logs-collapsed', collapsed);
      logsPanel.classList.toggle('collapsed', collapsed);
      toggleLogsBtn.textContent = collapsed ? 'Restore' : 'Minimize';
      toggleLogsBtn.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
      localStorage.setItem('logsCollapsed', collapsed ? 'true' : 'false');
      if (!collapsed && autoScroll.checked) {
        logsPre.scrollTop = logsPre.scrollHeight;
      }
    }

    function clearLogs() {
      logsPre.textContent = '';
    }

    function parseSseData(event) {
      try { return JSON.parse(event.data); } catch { return null; }
    }

    function renderLogSnapshot(data) {
      logsPre.textContent = (data.entries || []).map((entry) => entry.text).join('');
      if (autoScroll.checked) logsPre.scrollTop = logsPre.scrollHeight;
    }

    function connectLogStream() {
      if (logSource) logSource.close();
      const url = new URL('/api/logs/stream', window.location.origin);
      if (apiKey.value) url.searchParams.set('api_key', apiKey.value);
      if (logModel.value) url.searchParams.set('model', logModel.value);
      logStreamState.textContent = 'connecting';
      logSource = new EventSource(url);
      logSource.onopen = () => { logStreamState.textContent = 'connected'; };
      logSource.onerror = () => { logStreamState.textContent = 'disconnected'; };
      logSource.addEventListener('snapshot', (event) => {
        const data = parseSseData(event);
        if (data) renderLogSnapshot(data);
      });
      logSource.addEventListener('log', (event) => {
        const data = parseSseData(event);
        if (data) appendLog(data.text || '');
      });
    }

    function scheduleLogReconnect() {
      clearTimeout(logReconnectTimer);
      logReconnectTimer = setTimeout(connectLogStream, 250);
    }

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, (char) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
      }[char]));
    }

    function escapeAttr(value) {
      return escapeHtml(value).replace(/`/g, '&#96;');
    }

    updateGpuLayersInput();
    updatePoolingControl();
    connectLogStream();
    refreshAll();
    setInterval(loadStatus, 5000);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host=PROXY_HOST, port=PROXY_PORT, reload=False)
