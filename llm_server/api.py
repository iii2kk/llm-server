from fastapi import APIRouter

from .admin_api import (
    api_grammars,
    api_logs,
    api_logs_stream,
    api_models,
    api_restart,
    api_start,
    api_status,
    api_stop,
    router as admin_router,
)
from .openai_api import (
    chat_completions,
    embeddings,
    router as openai_router,
    v1_models,
)


router = APIRouter()
router.include_router(admin_router)
router.include_router(openai_router)

