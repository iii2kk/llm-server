from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
GRAMMAR_DIR = (BASE_DIR / "grammars").resolve()

load_dotenv(BASE_DIR / ".env")


def required_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Required environment variable {name} is not set or empty")
    return value


def find_llama_bin_dir(value: str) -> Path | None:
    """Resolve either a llama.cpp build directory or its bin directory."""
    if not value.strip():
        return None

    configured_dir = Path(value).expanduser()
    for bin_dir in (configured_dir, configured_dir / "bin"):
        llama_server = bin_dir / "llama-server"
        if llama_server.is_file() and os.access(llama_server, os.X_OK):
            return bin_dir
    return None


def available_llama_bin_dirs(configured_dirs: dict[str, str]) -> dict[str, Path]:
    """Keep only configured backends that have a usable llama-server."""
    available: dict[str, Path] = {}
    for backend_id, value in configured_dirs.items():
        bin_dir = find_llama_bin_dir(value)
        if bin_dir is not None:
            available[backend_id] = bin_dir
    return available


legacy_llama_bin_dir = os.getenv("LLAMA_BIN_DIR", "").strip()
CONFIGURED_LLAMA_BIN_DIRS = {
    "cuda": os.getenv("LLAMA_BIN_DIR_CUDA", "").strip(),
    "vulkan": os.getenv("LLAMA_BIN_DIR_VULKAN", legacy_llama_bin_dir).strip(),
    "vulkan-rocmfpx": os.getenv("LLAMA_BIN_DIR_VULKAN_ROCMFPX", "").strip(),
    "rocm": os.getenv("LLAMA_BIN_DIR_ROCM", "").strip(),
    "rocm-fastmtp": os.getenv("LLAMA_BIN_DIR_ROCM_FASTMTP", "").strip(),
}
LLAMA_BIN_DIRS = available_llama_bin_dirs(CONFIGURED_LLAMA_BIN_DIRS)
if not LLAMA_BIN_DIRS:
    configured = ", ".join(
        f"{backend_id}={value}"
        for backend_id, value in CONFIGURED_LLAMA_BIN_DIRS.items()
        if value
    )
    detail = f" Configured values: {configured}." if configured else ""
    raise RuntimeError(
        "No usable llama-server binary was found. Set at least one LLAMA_BIN_DIR_* "
        "variable to a build directory or bin directory containing an executable "
        f"llama-server.{detail}"
    )
DEFAULT_LLAMA_BACKEND = os.getenv("DEFAULT_LLAMA_BACKEND", "vulkan").strip().lower()
if DEFAULT_LLAMA_BACKEND not in LLAMA_BIN_DIRS:
    DEFAULT_LLAMA_BACKEND = next(iter(LLAMA_BIN_DIRS))
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
DEFAULT_STARTUP_PROFILE_FILE = SETTINGS_DIR / "startup-profile.json"
REQUEST_RESPONSE_LOG_DIR = Path(
    os.getenv("REQUEST_RESPONSE_LOG_DIR", str(SETTINGS_DIR / "request-logs"))
).expanduser()
REQUEST_RESPONSE_LOG_RETENTION_DAYS = int(os.getenv("REQUEST_RESPONSE_LOG_RETENTION_DAYS", "7"))
RECENT_MODELS_MAX = 5
DEFAULT_MODEL_PURPOSES = ("chat", "embeddings")
SAVED_BACKEND_SETTING_KEYS = (
    "model",
    "backend",
    "mmproj_enabled",
    "ctx_size",
    "gpu_layers",
    "threads",
    "batch_size",
    "ubatch_size",
    "parallel",
    "cache_type_k",
    "cache_type_v",
    "flash_attn",
    "mtp",
    "mtp_draft_tokens",
    "reasoning",
    "reasoning_effort",
    "reasoning_preserve",
    "reasoning_budget",
    "reasoning_format",
    "mode",
    "pooling",
)
MODEL_MODES = ("auto", "chat", "embeddings")
POOLING_TYPES = ("auto", "mean", "cls", "last")
MTP_MODES = ("auto", "on", "off")
REASONING_EFFORTS = ("default", "minimal", "low", "medium", "high", "xhigh", "max")
ROCM_BACKEND_IDS = frozenset(("rocm", "rocm-fastmtp"))
FASTMTP_BACKEND_IDS = frozenset(("rocm-fastmtp",))
ROCMFPX_BACKEND_IDS = frozenset(("vulkan-rocmfpx",))
BACKEND_LABELS = {
    "cuda": "CUDA",
    "vulkan": "Vulkan",
    "vulkan-rocmfpx": "Vulkan ROCmFPx (Qwen3.8 fork)",
    "rocm": "ROCm (HIP)",
    "rocm-fastmtp": "ROCm FastMTP (patched)",
}
CACHE_TYPES = (
    "auto",
    "f32",
    "f16",
    "bf16",
    "q8_0",
    "q4_0",
    "q4_1",
    "iq4_nl",
    "q5_0",
    "q5_1",
)
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
