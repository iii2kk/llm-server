from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import router
from .backend import registry
from .request_logs import request_logger


WEB_DIR = Path(__file__).resolve().parent / "web"


class SuppressStatusAccessLog(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/api/status" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(SuppressStatusAccessLog())


async def autoload_startup_profile() -> None:
    profile_path = os.getenv("MODEL_STARTUP_FILE", "").strip()
    if not profile_path:
        return
    try:
        result = await registry.load_startup_profile(profile_path)
    except Exception as exc:
        logging.getLogger(__name__).warning("Startup profile was not loaded: %s", exc)
        return
    logging.getLogger(__name__).info(
        "Loaded startup profile %s: %s model(s), %s error(s)",
        result["path"],
        result["loaded"],
        len(result["errors"]),
    )


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    request_logger.cleanup()
    startup_task = asyncio.create_task(autoload_startup_profile())
    try:
        yield
    finally:
        if not startup_task.done():
            startup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await startup_task
        await registry.stop_all()


def create_app() -> FastAPI:
    application = FastAPI(
        title="llama.cpp OpenAI-compatible proxy",
        lifespan=lifespan,
    )
    application.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    application.include_router(router)

    @application.get("/", response_class=FileResponse)
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @application.get("/request-logs", response_class=FileResponse)
    async def request_logs() -> FileResponse:
        return FileResponse(WEB_DIR / "request-logs.html")

    return application


app = create_app()
