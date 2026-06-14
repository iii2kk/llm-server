from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from pathlib import Path
from typing import Any

from fastapi.responses import JSONResponse

from .command import (
    build_llama_command,
    check_backend_health,
    is_tcp_port_available,
    optional_bool,
    wait_for_backend,
)
from .config import (
    BACKEND_HOST,
    BACKEND_PORT,
    BASE_DIR,
    DEFAULT_LLAMA_BACKEND,
    LOG_BUFFER_MAX_BYTES,
    MODEL_SETTINGS_FILE,
    RECENT_MODELS_MAX,
)
from .logs import BackendLogStore
from .models import (
    normalize_backend_settings,
    resolve_llama_backend,
    resolve_model_reference,
    resolve_model_reference_required,
)
from .responses import (
    model_capability_error,
    openai_error_response,
    prefix_log_text,
    prepend_env_path,
)
from .settings_store import load_recent_model_ids, load_saved_model_settings, saved_settings_payload, write_saved_model_settings

logger = logging.getLogger(__name__)

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
        self.backend_id = DEFAULT_LLAMA_BACKEND
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
            "backend": self.backend_id,
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

            backend_id, llama_bin_dir = resolve_llama_backend(settings.get("backend"))
            command = build_llama_command(
                settings,
                model=self.model_path,
                port=self.port,
                llama_bin_dir=llama_bin_dir,
            )
            run_id = await self.log_store.start_run()
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = prepend_env_path(
                env.get("LD_LIBRARY_PATH", ""),
                str(llama_bin_dir.resolve()),
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
            self.backend_id = backend_id
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
        model_id, model_path = resolve_model_reference_required(
            settings.get("model"),
            self.saved_settings,
        )
        normalized_settings = normalize_backend_settings(model_id, model_path, settings)
        if normalized_settings["effective_mode"] == "rerank":
            raise ValueError("rerank models are detected but /v1/rerank is not supported")
        instance = await self._instance_for(model_id, model_path)
        result = await instance.start(normalized_settings, conflict_if_running=conflict_if_running)
        if not isinstance(result, JSONResponse):
            self._save_settings(model_id, normalized_settings)
        return result

    async def restart(self, settings: dict[str, Any]) -> dict[str, Any] | JSONResponse:
        model_id, model_path = resolve_model_reference_required(
            settings.get("model"),
            self.saved_settings,
        )
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

        model_id, model_path = resolve_model_reference_required(
            settings.get("model"),
            self.saved_settings,
        )
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
        resolved = resolve_model_reference(model_ref, self.saved_settings)
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
        resolved = resolve_model_reference(model_ref, self.saved_settings)
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
