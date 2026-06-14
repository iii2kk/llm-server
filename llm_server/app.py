from __future__ import annotations

import contextlib
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import router
from .backend import registry


WEB_DIR = Path(__file__).resolve().parent / "web"


class SuppressStatusAccessLog(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/api/status" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(SuppressStatusAccessLog())


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        yield
    finally:
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

    return application


app = create_app()
