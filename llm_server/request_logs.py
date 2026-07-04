from __future__ import annotations

import json
import logging
import re
import time
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from .config import REQUEST_RESPONSE_LOG_DIR, REQUEST_RESPONSE_LOG_RETENTION_DAYS


logger = logging.getLogger(__name__)

LOG_FILE_RE = re.compile(r"^.+-[0-9a-f]{12}\.(?P<date>\d{4}-\d{2}-\d{2})\.jsonl$")


def decode_response_body(content: bytes, content_type: str | None = None) -> Any:
    text = content.decode("utf-8", errors="replace")
    if "json" in (content_type or "").lower():
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


class RequestResponseLogger:
    def __init__(
        self,
        log_dir: Path = REQUEST_RESPONSE_LOG_DIR,
        *,
        retention_days: int = REQUEST_RESPONSE_LOG_RETENTION_DAYS,
    ) -> None:
        self.log_dir = log_dir
        self.retention_days = max(1, retention_days)
        self._last_cleanup_at = 0.0

    def log(
        self,
        *,
        request_id: str,
        endpoint: str,
        model_id: str,
        request_payload: dict[str, Any],
        status_code: int,
        response_body: Any,
        started_at: float,
        completed_at: float | None = None,
        stream: bool = False,
        error: str | None = None,
    ) -> None:
        completed = completed_at or time.time()
        record: dict[str, Any] = {
            "ts": datetime.fromtimestamp(completed, tz=timezone.utc).isoformat(),
            "request_id": request_id,
            "endpoint": endpoint,
            "model": model_id,
            "stream": stream,
            "duration_ms": round((completed - started_at) * 1000, 3),
            "request": request_payload,
            "response": {
                "status_code": status_code,
                "body": response_body,
            },
        }
        if error:
            record["error"] = error

        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._cleanup_old_logs(force=False)
            path = self._path_for(model_id, datetime.fromtimestamp(completed).date())
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str, separators=(",", ":")))
                handle.write("\n")
        except OSError as exc:
            logger.warning("Failed to write request/response log for %s: %s", model_id, exc)

    def cleanup(self) -> None:
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._cleanup_old_logs(force=True)
        except OSError as exc:
            logger.warning("Failed to clean request/response logs in %s: %s", self.log_dir, exc)

    def _path_for(self, model_id: str, log_date: date) -> Path:
        digest = sha256(model_id.encode("utf-8", errors="replace")).hexdigest()[:12]
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("._-")
        if not slug:
            slug = "model"
        slug = slug[:96]
        return self.log_dir / f"{slug}-{digest}.{log_date.isoformat()}.jsonl"

    def _cleanup_old_logs(self, *, force: bool) -> None:
        now = time.time()
        if not force and now - self._last_cleanup_at < 3600:
            return
        self._last_cleanup_at = now

        cutoff_date = date.today() - timedelta(days=self.retention_days)
        cutoff_mtime = now - (self.retention_days * 24 * 60 * 60)
        for path in self.log_dir.glob("*.jsonl"):
            if self._is_expired(path, cutoff_date, cutoff_mtime):
                try:
                    path.unlink()
                except OSError as exc:
                    logger.warning("Failed to delete expired request/response log %s: %s", path, exc)

    def _is_expired(self, path: Path, cutoff_date: date, cutoff_mtime: float) -> bool:
        match = LOG_FILE_RE.match(path.name)
        if match:
            try:
                return date.fromisoformat(match.group("date")) < cutoff_date
            except ValueError:
                return False
        try:
            return path.stat().st_mtime < cutoff_mtime
        except OSError:
            return False


request_logger = RequestResponseLogger()
