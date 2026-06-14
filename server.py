from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_server import config as _config
from llm_server import command as _command
from llm_server import logic as _logic
from llm_server import models as _models
from llm_server.api import (
    api_grammars,
    api_logs,
    api_logs_stream,
    api_models,
    api_restart,
    api_start,
    api_status,
    api_stop,
    chat_completions,
    embeddings,
    v1_models,
)
from llm_server.app import SuppressStatusAccessLog, app, create_app, lifespan
from llm_server.logic import *


# Keep the original module-level customization points working while callers
# migrate to the package modules.
def _sync_compat_config() -> None:
    _logic.LLAMA_BIN_DIRS = LLAMA_BIN_DIRS
    _logic.DEFAULT_LLAMA_BACKEND = DEFAULT_LLAMA_BACKEND
    _config.LLAMA_BIN_DIRS = LLAMA_BIN_DIRS
    _config.DEFAULT_LLAMA_BACKEND = DEFAULT_LLAMA_BACKEND
    _command.DEFAULT_LLAMA_BACKEND = DEFAULT_LLAMA_BACKEND
    _models.LLAMA_BIN_DIRS = LLAMA_BIN_DIRS
    _models.DEFAULT_LLAMA_BACKEND = DEFAULT_LLAMA_BACKEND


def resolve_llama_backend(value: Any) -> tuple[str, Path]:
    _sync_compat_config()
    return _logic.resolve_llama_backend(value)


def normalize_backend_settings(
    model_id: str,
    model_path: Path,
    settings: dict[str, Any],
) -> dict[str, Any]:
    _sync_compat_config()
    return _logic.normalize_backend_settings(model_id, model_path, settings)


def build_llama_command(
    settings: dict[str, Any],
    *,
    model: Path,
    port: int,
    llama_bin_dir: Path | None = None,
) -> list[str]:
    _sync_compat_config()
    return _logic.build_llama_command(
        settings,
        model=model,
        port=port,
        llama_bin_dir=llama_bin_dir,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host=PROXY_HOST, port=PROXY_PORT, reload=False)
