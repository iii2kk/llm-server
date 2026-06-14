from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .backend import registry
from .config import (
    BACKEND_LABELS,
    DEFAULT_LLAMA_BACKEND,
    GRAMMAR_DIR,
    LLAMA_BIN_DIRS,
    MODEL_DIR,
)
from .models import grammar_options, model_options
from .responses import (
    apply_default_max_tokens,
    backend_error_response,
    expand_grammar_file,
    forward_headers,
    openai_error_response,
    require_auth,
    require_auth_or_query,
    sse_event,
)

router = APIRouter()

@router.get("/api/models")
async def api_models(authorization: str | None = Header(default=None)) -> JSONResponse:
    auth_error = require_auth(authorization)
    if auth_error:
        return auth_error
    return JSONResponse(
        {
            "models": model_options(registry.saved_settings),
            "model_dir": str(MODEL_DIR.resolve()),
            "backends": [
                {
                    "id": backend_id,
                    "label": BACKEND_LABELS.get(backend_id, backend_id),
                    "bin_dir": str(bin_dir.resolve()),
                }
                for backend_id, bin_dir in LLAMA_BIN_DIRS.items()
            ],
            "default_backend": DEFAULT_LLAMA_BACKEND,
            "saved_settings": await registry.saved_settings_snapshot(),
            "recent_models": await registry.recent_model_ids_snapshot(),
        }
    )


@router.get("/api/grammars")
async def api_grammars(authorization: str | None = Header(default=None)) -> JSONResponse:
    auth_error = require_auth(authorization)
    if auth_error:
        return auth_error
    return JSONResponse({"grammars": grammar_options(), "grammar_dir": str(GRAMMAR_DIR.resolve())})


@router.get("/api/status")
async def api_status(authorization: str | None = Header(default=None)) -> JSONResponse:
    auth_error = require_auth(authorization)
    if auth_error:
        return auth_error
    return JSONResponse(await registry.status())


@router.get("/api/logs")
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


@router.get("/api/logs/stream")
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


@router.post("/api/start")
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


@router.post("/api/stop")
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


@router.post("/api/restart")
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


