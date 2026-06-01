from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import internal, websocket
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.deps.redis import close_redis, init_redis
from app.api.v1.scenario import router as scenario_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """서버 시작/종료 시 외부 의존성 초기화 및 정리."""
    setup_logging()
    settings = get_settings()

    app.state.settings = settings
    app.state.redis = await init_redis(settings.redis_url)

    yield

    await close_redis(app.state.redis)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="bada-ai-server",
        version="0.1.0",
        docs_url="/docs" if settings.env != "prod" else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.env != "prod" else None,
        lifespan=lifespan,
    )

    if settings.env != "prod":
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(websocket.router, prefix="/ws", tags=["websocket"])
    app.include_router(internal.router, prefix="/internal/v1", tags=["internal"])
    app.include_router(scenario_router)
    return app

app = create_app()