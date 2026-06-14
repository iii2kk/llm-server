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
@router.post("/v1/chat/completions", response_model=None)
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


@router.post("/v1/embeddings")
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


@router.get("/v1/models")
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
        for model in model_options(registry.saved_settings)
    ]
    return JSONResponse({"object": "list", "data": data})
