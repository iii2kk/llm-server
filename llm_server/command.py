from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import time
from pathlib import Path
from typing import Any

import httpx

from .config import BACKEND_HOST, DEFAULT_LLAMA_BACKEND, MODEL_LOAD_TIMEOUT_SECONDS
from .models import find_mmproj_for_model, resolve_llama_backend

def optional_int(settings: dict[str, Any], key: str) -> int | None:
    value = settings.get(key)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc


def optional_reasoning_budget(settings: dict[str, Any]) -> int | None:
    value = optional_int(settings, "reasoning_budget")
    if value is not None and value < -1:
        raise ValueError("reasoning_budget must be -1 or a non-negative integer")
    return value


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


def build_llama_command(
    settings: dict[str, Any],
    *,
    model: Path,
    port: int,
    llama_bin_dir: Path | None = None,
) -> list[str]:
    if llama_bin_dir is None:
        backend_id, llama_bin_dir = resolve_llama_backend(settings.get("backend"))
    else:
        backend_id = (
            DEFAULT_LLAMA_BACKEND
            if settings.get("backend") in (None, "")
            else str(settings["backend"]).strip().lower()
        )
    llama_server = (llama_bin_dir / "llama-server").resolve()
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
    if backend_id == "rocm":
        command.append("--direct-io")

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

    reasoning_budget = optional_reasoning_budget(settings)
    if reasoning_budget is not None:
        command.extend(["--reasoning-budget", str(reasoning_budget)])

    reasoning_format = settings.get("reasoning_format")
    if reasoning_format in ("auto", "none", "deepseek", "deepseek-legacy"):
        command.extend(["--reasoning-format", str(reasoning_format)])

    if settings.get("effective_mtp"):
        command.extend(
            [
                "--spec-type",
                "draft-mtp",
                "--spec-draft-n-max",
                str(settings.get("mtp_draft_tokens", 3)),
            ]
        )

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
