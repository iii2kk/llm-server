from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from .config import DEFAULT_MAX_TOKENS, GRAMMAR_DEFAULT_MAX_TOKENS, GRAMMAR_DIR, PROXY_API_KEY

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
