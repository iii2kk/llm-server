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


legacy_llama_bin_dir = os.getenv("LLAMA_BIN_DIR", "").strip()
LLAMA_BIN_DIRS = {
    backend_id: Path(value).expanduser()
    for backend_id, value in {
        "vulkan": os.getenv("LLAMA_BIN_DIR_VULKAN", legacy_llama_bin_dir).strip(),
        "rocm": os.getenv("LLAMA_BIN_DIR_ROCM", "").strip(),
    }.items()
    if value
}
if not LLAMA_BIN_DIRS:
    raise RuntimeError(
        "At least one of LLAMA_BIN_DIR, LLAMA_BIN_DIR_VULKAN, or LLAMA_BIN_DIR_ROCM must be set"
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
RECENT_MODELS_MAX = 5
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
    "flash_attn",
    "reasoning",
    "reasoning_budget",
    "reasoning_format",
    "mode",
    "pooling",
)
MODEL_MODES = ("auto", "chat", "embeddings")
POOLING_TYPES = ("auto", "mean", "cls", "last")
BACKEND_LABELS = {
    "vulkan": "Vulkan",
    "rocm": "ROCm (HIP)",
}
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
