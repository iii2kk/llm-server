from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .config import (
    BASE_DIR,
    DEFAULT_MODEL_PURPOSES,
    DEFAULT_STARTUP_PROFILE_FILE,
    MODEL_SETTINGS_FILE,
    RECENT_MODELS_MAX,
    SAVED_BACKEND_SETTING_KEYS,
    SETTINGS_DIR,
)

logger = logging.getLogger(__name__)

def resolve_startup_profile_path(path: str | Path | None = None) -> Path:
    if path in (None, ""):
        return DEFAULT_STARTUP_PROFILE_FILE
    profile_path = Path(str(path)).expanduser()
    if not profile_path.is_absolute():
        profile_path = BASE_DIR / profile_path
    return profile_path


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


def load_default_model_ids() -> dict[str, str]:
    raw = load_model_settings_document()
    default_models = raw.get("default_models")
    if not isinstance(default_models, dict):
        return {}

    defaults: dict[str, str] = {}
    for purpose in DEFAULT_MODEL_PURPOSES:
        model_id = default_models.get(purpose)
        if isinstance(model_id, str) and model_id:
            defaults[purpose] = model_id
    return defaults


def write_saved_model_settings(
    saved_settings: dict[str, dict[str, Any]],
    recent_model_ids: list[str] | None = None,
    default_model_ids: dict[str, str] | None = None,
) -> None:
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "recent_models": list(recent_model_ids or [])[:RECENT_MODELS_MAX],
        "default_models": {
            purpose: model_id
            for purpose, model_id in sorted((default_model_ids or {}).items())
            if purpose in DEFAULT_MODEL_PURPOSES and model_id
        },
        "models": {
            model_id: saved_settings_payload(model_id, settings)
            for model_id, settings in sorted(saved_settings.items())
        },
    }
    tmp_path = MODEL_SETTINGS_FILE.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(MODEL_SETTINGS_FILE)


def load_startup_profile(path: str | Path) -> dict[str, Any]:
    profile_path = resolve_startup_profile_path(path)
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to load startup profile from {profile_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"startup profile root must be an object: {profile_path}")
    return raw


def startup_profile_models(raw: dict[str, Any]) -> list[dict[str, Any]]:
    models = raw.get("models")
    if isinstance(models, dict):
        items = []
        for model_id, settings in models.items():
            if not isinstance(model_id, str) or not isinstance(settings, dict):
                continue
            items.append(saved_settings_payload(model_id, {"model": model_id, **settings}))
        return items
    if not isinstance(models, list):
        return []

    items = []
    for settings in models:
        if not isinstance(settings, dict):
            continue
        model_id = settings.get("model")
        if not isinstance(model_id, str) or not model_id:
            continue
        items.append(saved_settings_payload(model_id, settings))
    return items


def startup_profile_default_models(raw: dict[str, Any]) -> dict[str, str] | None:
    default_models = raw.get("default_models")
    if default_models is None:
        return None
    if not isinstance(default_models, dict):
        return {}
    return {
        purpose: model_id
        for purpose in DEFAULT_MODEL_PURPOSES
        if isinstance((model_id := default_models.get(purpose)), str) and model_id
    }


def write_startup_profile(
    *,
    path: str | Path | None,
    models: list[dict[str, Any]],
    default_model_ids: dict[str, str],
) -> Path:
    profile_path = resolve_startup_profile_path(path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "default_models": {
            purpose: model_id
            for purpose, model_id in sorted(default_model_ids.items())
            if purpose in DEFAULT_MODEL_PURPOSES and model_id
        },
        "models": [
            saved_settings_payload(str(settings["model"]), settings)
            for settings in models
            if isinstance(settings.get("model"), str) and settings["model"]
        ],
    }
    tmp_path = profile_path.with_suffix(f"{profile_path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(profile_path)
    return profile_path
