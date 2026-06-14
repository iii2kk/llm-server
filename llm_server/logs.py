from __future__ import annotations

import asyncio
import sys
import time
from collections import deque
from typing import Any

class BackendLogStore:
    def __init__(self, max_bytes: int, *, echo_stdout: bool = True) -> None:
        self.max_bytes = max(16 * 1024, max_bytes)
        self.echo_stdout = echo_stdout
        self.entries: deque[dict[str, Any]] = deque()
        self.current_bytes = 0
        self.seq = 0
        self.run_id = 0
        self.truncated = False
        self.load_state = "stopped"
        self.load_progress: int | None = None
        self.progress_dots = 0
        self.listeners: set[asyncio.Queue[dict[str, Any]]] = set()
        self._lock = asyncio.Lock()

    async def start_run(self) -> int:
        async with self._lock:
            self.run_id += 1
            self.entries.clear()
            self.current_bytes = 0
            self.seq = 0
            self.truncated = False
            self.load_state = "loading"
            self.load_progress = 0
            self.progress_dots = 0
            run_id = self.run_id
        await self._broadcast_state("state")
        return run_id

    async def append(self, text: str, run_id: int, *, model_id: str | None = None) -> None:
        if not text:
            return

        if self.echo_stdout:
            sys.stdout.write(text)
            sys.stdout.flush()

        entry: dict[str, Any] | None = None
        should_broadcast_state = False
        async with self._lock:
            if run_id != self.run_id:
                return

            entry = {
                "seq": self.seq,
                "ts": time.time(),
                "text": text,
                "run_id": self.run_id,
            }
            if model_id is not None:
                entry["model_id"] = model_id
            self.seq += 1
            self.entries.append(entry)
            self.current_bytes += len(text.encode("utf-8", errors="replace"))
            self._trim_locked()
            should_broadcast_state = self._update_progress_from_log_locked(text)

        await self._broadcast("log", entry)
        if should_broadcast_state:
            await self._broadcast_state("state")

    async def mark_ready(self, run_id: int) -> None:
        async with self._lock:
            if run_id != self.run_id or self.load_state == "ready":
                return
            self.load_state = "ready"
            self.load_progress = 100
        await self._broadcast_state("state")
        await self._broadcast_state("done")

    async def mark_error(self, run_id: int) -> None:
        async with self._lock:
            if run_id != self.run_id or self.load_state == "stopped":
                return
            self.load_state = "error"
        await self._broadcast_state("state")
        await self._broadcast_state("error")

    async def mark_stopped(self, run_id: int | None = None) -> None:
        async with self._lock:
            if run_id is not None and run_id != self.run_id:
                return
            self.load_state = "stopped"
            self.load_progress = None
            self.progress_dots = 0
        await self._broadcast_state("state")

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "run_id": self.run_id,
                "next_seq": self.seq,
                "entries": list(self.entries),
                "truncated": self.truncated,
                "load": self.load_summary_locked(),
            }

    async def load_summary(self) -> dict[str, Any]:
        async with self._lock:
            return self.load_summary_locked()

    def load_summary_locked(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "state": self.load_state,
            "progress": self.load_progress,
        }

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        async with self._lock:
            self.listeners.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        async with self._lock:
            self.listeners.discard(queue)

    def _trim_locked(self) -> None:
        while self.entries and self.current_bytes > self.max_bytes:
            old = self.entries.popleft()
            self.current_bytes -= len(str(old["text"]).encode("utf-8", errors="replace"))
            self.truncated = True

    def _update_progress_from_log_locked(self, text: str) -> bool:
        if self.load_state != "loading":
            return False

        dots = 0
        for part in text.replace("\r", "").split("\n"):
            stripped = part.strip()
            if stripped and set(stripped) <= {"."}:
                dots += stripped.count(".")

        if dots == 0:
            return False

        old_progress = self.load_progress
        self.progress_dots += dots
        self.load_progress = min(99, max(self.load_progress or 0, self.progress_dots))
        return self.load_progress != old_progress

    async def _broadcast_state(self, event: str) -> None:
        await self._broadcast(event, await self.load_summary())

    async def _broadcast(self, event: str, data: dict[str, Any]) -> None:
        async with self._lock:
            listeners = list(self.listeners)
        payload = {"event": event, "data": data}
        for queue in listeners:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass


