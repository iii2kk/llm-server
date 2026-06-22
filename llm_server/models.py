from __future__ import annotations

import os
import mmap
import re
import struct
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_LLAMA_BACKEND,
    GGUF_METADATA_CACHE,
    GGUF_POOLING_NAMES,
    GGUF_SCALAR_FORMATS,
    GRAMMAR_DIR,
    LLAMA_BIN_DIRS,
    MODEL_DIR,
    MODEL_MODES,
    MTP_MODES,
    POOLING_TYPES,
)

SHARDED_GGUF_RE = re.compile(r"^(?P<base>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$", re.IGNORECASE)
EMBEDDED_MTP_ARCHITECTURES = {"qwen35", "qwen35moe", "step35", "cohere2moe"}
GGUF_METADATA_SEARCH_BYTES = 32 * 1024 * 1024

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


def _gguf_find_scalar_metadata(model: Path, key: str) -> Any | None:
    key_bytes = key.encode("utf-8")
    pattern = struct.pack("<Q", len(key_bytes)) + key_bytes
    with model.open("rb") as handle:
        search_size = min(model.stat().st_size, GGUF_METADATA_SEARCH_BYTES)
        if search_size <= 0:
            return None
        with mmap.mmap(handle.fileno(), length=search_size, access=mmap.ACCESS_READ) as mapped:
            position = mapped.find(pattern)
            if position < 0:
                return None
            value_offset = position + len(pattern)
            if value_offset + 4 > search_size:
                return None
            value_type = int(struct.unpack_from("<I", mapped, value_offset)[0])
            scalar_format = GGUF_SCALAR_FORMATS.get(value_type)
            if scalar_format is None:
                return None
            scalar_size = struct.calcsize(scalar_format)
            scalar_offset = value_offset + 4
            if scalar_offset + scalar_size > search_size:
                return None
            return struct.unpack_from(f"<{scalar_format}", mapped, scalar_offset)[0]


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
        "mtp_layers": 0,
        "mtp_supported": False,
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
                wanted = wanted or key.endswith(".nextn_predict_layers")
                if wanted:
                    value = _gguf_read_value(handle, value_type)
                    if key == "general.architecture":
                        architecture = str(value)
                        result["architecture"] = architecture
                    elif key.endswith(".pooling_type"):
                        result["pooling"] = GGUF_POOLING_NAMES.get(int(value), f"unknown:{value}")
                    elif key.endswith(".embedding_length"):
                        result["embedding_dimensions"] = int(value)
                    elif key.endswith(".nextn_predict_layers"):
                        result["mtp_layers"] = max(0, int(value))
                else:
                    _gguf_skip_value(handle, value_type)

        architecture_name = str(result["architecture"] or "").lower()
        if result["mtp_layers"] == 0 and architecture_name in EMBEDDED_MTP_ARCHITECTURES:
            mtp_layers = _gguf_find_scalar_metadata(
                model,
                f"{architecture_name}.nextn_predict_layers",
            )
            if mtp_layers is not None:
                result["mtp_layers"] = max(0, int(mtp_layers))
        result["mtp_supported"] = (
            result["mtp_layers"] > 0
            and architecture_name in EMBEDDED_MTP_ARCHITECTURES
        )
        pooling = result["pooling"]
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


def validate_mtp_mode(value: Any) -> str:
    mode = "auto" if value in (None, "") else str(value).strip().lower()
    if mode not in MTP_MODES:
        raise ValueError("mtp must be auto, on, or off")
    return mode


def validate_mtp_draft_tokens(value: Any) -> int:
    if value in (None, ""):
        return 3
    if isinstance(value, bool) or isinstance(value, float):
        raise ValueError("mtp_draft_tokens must be a positive integer")
    if isinstance(value, int):
        tokens = value
    elif isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        tokens = int(value)
    else:
        raise ValueError("mtp_draft_tokens must be a positive integer")
    if tokens < 1:
        raise ValueError("mtp_draft_tokens must be a positive integer")
    return tokens


def resolve_llama_backend(value: Any) -> tuple[str, Path]:
    backend_id = DEFAULT_LLAMA_BACKEND if value in (None, "") else str(value).strip().lower()
    llama_bin_dir = LLAMA_BIN_DIRS.get(backend_id)
    if llama_bin_dir is None:
        available = ", ".join(LLAMA_BIN_DIRS)
        raise ValueError(f"backend must be one of: {available}")
    return backend_id, llama_bin_dir


def effective_model_config(model: Path, settings: dict[str, Any]) -> dict[str, Any]:
    metadata = read_gguf_metadata(model)
    configured_mode = validate_model_mode(settings.get("mode"))
    configured_pooling = validate_pooling(settings.get("pooling"))
    configured_mtp = validate_mtp_mode(settings.get("mtp"))
    mtp_draft_tokens = validate_mtp_draft_tokens(settings.get("mtp_draft_tokens"))
    effective_mode = metadata["detected_mode"] if configured_mode == "auto" else configured_mode
    detected_pooling = metadata["pooling"]
    effective_pooling = detected_pooling if configured_pooling == "auto" else configured_pooling

    if effective_mode == "embeddings" and effective_pooling not in ("mean", "cls", "last"):
        raise ValueError(
            "pooling must be set to mean, cls, or last when an embedding model has no usable GGUF pooling metadata"
        )
    if effective_mode != "embeddings":
        effective_pooling = None

    if configured_mtp == "on" and effective_mode != "chat":
        raise ValueError("MTP can only be enabled for chat models")
    if configured_mtp == "on" and metadata["mtp_layers"] == 0:
        raise ValueError("MTP is enabled but the GGUF has no nextn_predict_layers metadata")
    if configured_mtp == "on" and not metadata["mtp_supported"]:
        raise ValueError(
            f"MTP is not supported for embedded heads on architecture: {metadata['architecture'] or 'unknown'}"
        )
    effective_mtp = effective_mode == "chat" and metadata["mtp_supported"] and configured_mtp != "off"

    return {
        **metadata,
        "configured_mode": configured_mode,
        "configured_pooling": configured_pooling,
        "configured_mtp": configured_mtp,
        "effective_mode": effective_mode,
        "effective_pooling": effective_pooling,
        "effective_mtp": effective_mtp,
        "mtp_draft_tokens": mtp_draft_tokens,
        "capabilities": [effective_mode],
    }


def normalize_backend_settings(model_id: str, model_path: Path, settings: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(settings)
    normalized["model"] = model_id
    normalized["backend"] = resolve_llama_backend(normalized.get("backend"))[0]
    normalized["mode"] = validate_model_mode(normalized.get("mode"))
    normalized["pooling"] = validate_pooling(normalized.get("pooling"))
    normalized["mtp"] = validate_mtp_mode(normalized.get("mtp"))
    normalized["mtp_draft_tokens"] = validate_mtp_draft_tokens(normalized.get("mtp_draft_tokens"))
    config = effective_model_config(model_path, normalized)
    normalized["effective_mode"] = config["effective_mode"]
    normalized["effective_pooling"] = config["effective_pooling"]
    normalized["effective_mtp"] = config["effective_mtp"]
    return normalized


def model_options(
    saved_settings_by_model: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if saved_settings_by_model is None:
        # Lazy import preserves the original module-level behavior without
        # making model discovery and backend management import each other.
        from .backend import registry

        saved_settings_by_model = registry.saved_settings

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
        saved_settings = (saved_settings_by_model or {}).get(str(relative), {})
        try:
            config = effective_model_config(resolved, saved_settings)
        except ValueError as exc:
            metadata = read_gguf_metadata(resolved)
            config = {
                **metadata,
                "configured_mode": str(saved_settings.get("mode") or "auto"),
                "configured_pooling": str(saved_settings.get("pooling") or "auto"),
                "configured_mtp": str(saved_settings.get("mtp") or "auto"),
                "effective_mode": metadata["detected_mode"],
                "effective_pooling": metadata["pooling"],
                "effective_mtp": False,
                "mtp_draft_tokens": saved_settings.get("mtp_draft_tokens", 3),
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


def model_lookup(
    saved_settings_by_model: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for item in model_options(saved_settings_by_model):
        for key in (
            item["relative_path"],
            item["path"],
            item["display_name"],
            item["name"],
        ):
            if key and key not in lookup:
                lookup[str(key)] = item
    return lookup


def resolve_model_reference(
    model: Any,
    saved_settings_by_model: dict[str, dict[str, Any]] | None = None,
) -> tuple[str, Path] | None:
    if model in (None, "", "local"):
        return None
    if not isinstance(model, str):
        return None

    lookup = model_lookup(saved_settings_by_model)
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


def resolve_model_reference_required(
    model: Any,
    saved_settings_by_model: dict[str, dict[str, Any]] | None = None,
) -> tuple[str, Path]:
    resolved = resolve_model_reference(model, saved_settings_by_model)
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
