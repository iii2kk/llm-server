from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from .config import REQUEST_RESPONSE_LOG_DIR, REQUEST_RESPONSE_LOG_RETENTION_DAYS


logger = logging.getLogger(__name__)

LOG_FILE_RE = re.compile(r"^.+-[0-9a-f]{12}\.(?P<date>\d{4}-\d{2}-\d{2})\.jsonl$")
MAX_LOG_LIMIT = 500
DEFAULT_LOG_LIMIT = 100


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


def request_log_options(log_dir: Path = REQUEST_RESPONSE_LOG_DIR) -> dict[str, Any]:
    models: dict[str, set[str]] = defaultdict(set)
    model_counts: dict[str, int] = defaultdict(int)
    dates: set[str] = set()
    endpoints: set[str] = set()
    status_counts = {"success": 0, "error": 0}
    total = 0

    for _path, _line_no, record in _iter_log_records(log_dir):
        total += 1
        model = _string_value(record.get("model"))
        log_date = _record_date(record)
        if model:
            models[model].add(log_date)
            model_counts[model] += 1
        dates.add(log_date)
        endpoint = _string_value(record.get("endpoint"))
        if endpoint:
            endpoints.add(endpoint)
        status_counts[_record_status(record)] += 1

    model_items = [
        {
            "model": model,
            "dates": sorted(model_dates, reverse=True),
            "count": model_counts[model],
        }
        for model, model_dates in sorted(models.items())
    ]
    return {
        "log_dir": str(log_dir),
        "models": model_items,
        "dates": sorted(dates, reverse=True),
        "endpoints": sorted(endpoints),
        "status_counts": status_counts,
        "total": total,
    }


def read_request_logs(
    *,
    log_dir: Path = REQUEST_RESPONSE_LOG_DIR,
    model: str | None = None,
    log_date: str | None = None,
    endpoint: str | None = None,
    status: str | None = None,
    query: str | None = None,
    offset: int = 0,
    limit: int = DEFAULT_LOG_LIMIT,
) -> dict[str, Any]:
    normalized_status = status if status in {"success", "error"} else None
    normalized_endpoint = endpoint.strip() if endpoint else None
    normalized_model = model.strip() if model else None
    normalized_query = query.strip().lower() if query else None
    safe_offset = max(0, offset)
    safe_limit = min(MAX_LOG_LIMIT, max(1, limit))

    records: list[dict[str, Any]] = []
    for path, line_no, record in _iter_log_records(log_dir, log_date=log_date):
        if normalized_model and record.get("model") != normalized_model:
            continue
        if normalized_endpoint and record.get("endpoint") != normalized_endpoint:
            continue
        record_status = _record_status(record)
        if normalized_status and record_status != normalized_status:
            continue
        if normalized_query and normalized_query not in json.dumps(record, ensure_ascii=False, default=str).lower():
            continue

        records.append(
            {
                "file": path.name,
                "line": line_no,
                "sort_key": _record_sort_key(path, line_no, record),
                "record": _compact_log_record(record),
            }
        )

    records.sort(key=lambda item: item["sort_key"], reverse=True)
    total = len(records)
    page = records[safe_offset : safe_offset + safe_limit]
    for item in page:
        item.pop("sort_key", None)
    next_offset = safe_offset + safe_limit if safe_offset + safe_limit < total else None
    return {
        "records": page,
        "total": total,
        "offset": safe_offset,
        "limit": safe_limit,
        "next_offset": next_offset,
    }


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


def _iter_log_records(
    log_dir: Path,
    *,
    log_date: str | None = None,
) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    if not log_dir.exists():
        return

    for path in sorted(log_dir.glob("*.jsonl")):
        file_date = _date_from_filename(path)
        if log_date and file_date != log_date:
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict):
                        yield path, line_no, record
        except OSError as exc:
            logger.warning("Failed to read request/response log %s: %s", path, exc)


def _date_from_filename(path: Path) -> str | None:
    match = LOG_FILE_RE.match(path.name)
    if not match:
        return None
    return match.group("date")


def _record_date(record: dict[str, Any]) -> str:
    ts = _string_value(record.get("ts"))
    if len(ts) >= 10:
        return ts[:10]
    return date.today().isoformat()


def _record_status(record: dict[str, Any]) -> str:
    status_code = _status_code(record)
    return "error" if record.get("error") or status_code >= 400 else "success"


def _status_code(record: dict[str, Any]) -> int:
    response = record.get("response")
    if isinstance(response, dict):
        try:
            return int(response.get("status_code") or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def _record_sort_key(path: Path, line_no: int, record: dict[str, Any]) -> tuple[str, float, int]:
    ts = _string_value(record.get("ts"))
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return ts, mtime, line_no


def _compact_log_record(record: dict[str, Any]) -> dict[str, Any]:
    compact = _compact_value(record, depth=0)
    if not isinstance(compact, dict):
        return {}
    compact["status"] = _record_status(record)
    return compact


def _compact_value(value: Any, *, depth: int) -> Any:
    if depth > 10:
        return {"omitted": "max_depth"}
    if isinstance(value, dict):
        return {str(key): _compact_value(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        if _is_numeric_list(value) and len(value) > 16:
            return {
                "omitted": "numeric_array",
                "length": len(value),
                "preview": value[:8],
            }
        if len(value) > 50:
            return {
                "omitted": "array",
                "length": len(value),
                "preview": [_compact_value(item, depth=depth + 1) for item in value[:5]],
            }
        return [_compact_value(item, depth=depth + 1) for item in value]
    if isinstance(value, str):
        if _looks_like_large_data(value):
            return {
                "omitted": "large_string",
                "length": len(value),
                "preview": value[:160],
            }
        if len(value) > 8000:
            return {
                "omitted": "long_string",
                "length": len(value),
                "preview": value[:4000],
            }
    return value


def _is_numeric_list(value: list[Any]) -> bool:
    return bool(value) and all(isinstance(item, int | float) and not isinstance(item, bool) for item in value[:64])


def _looks_like_large_data(value: str) -> bool:
    if len(value) < 2048:
        return False
    if value.startswith("data:"):
        return True
    sample = value[:2048]
    base64_chars = sum(1 for char in sample if char.isalnum() or char in "+/=\n\r")
    return base64_chars / len(sample) > 0.96


def _string_value(value: Any) -> str:
    return value if isinstance(value, str) else ""
