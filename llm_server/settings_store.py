from __future__ import annotations

import json
import logging
from typing import Any

from .config import MODEL_SETTINGS_FILE, RECENT_MODELS_MAX, SAVED_BACKEND_SETTING_KEYS, SETTINGS_DIR

logger = logging.getLogger(__name__)

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


def write_saved_model_settings(
    saved_settings: dict[str, dict[str, Any]],
    recent_model_ids: list[str] | None = None,
) -> None:
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "recent_models": list(recent_model_ids or [])[:RECENT_MODELS_MAX],
        "models": {
            model_id: saved_settings_payload(model_id, settings)
            for model_id, settings in sorted(saved_settings.items())
        },
    }
    tmp_path = MODEL_SETTINGS_FILE.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(MODEL_SETTINGS_FILE)


