from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import internal, websocket
from app.api.v1.community import router as community_router
from app.api.v1.scenario import router as scenario_router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.metrics import log_metric, now_ms
from app.db.base import init_db
from app.deps.redis import close_redis, init_redis
from app.workers.avti_worker import drain as drain_avti


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """서버 시작/종료 시 외부 의존성 초기화 및 정리."""
    setup_logging()
    settings = get_settings()

    app.state.settings = settings
    app.state.redis = await init_redis(settings.redis_url)

    await init_db()

    try:
        yield
    finally:
        await drain_avti()
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

    @app.middleware("http")
    async def _request_metrics(request: Request, call_next):
        """REST 요청 단위 지연 계측"""
        start = now_ms()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", None) or request.url.path
            log_metric(
                "http_request",
                method=request.method,
                path=path,
                status=status,
                duration_ms=round(now_ms() - start, 1),
            )

    if settings.env != "prod":
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/health", tags=["health"])
    async def health() -> dict:
        """공개 헬스체크"""
        return {"status": "ok"}

    app.include_router(websocket.router, prefix="/ws", tags=["websocket"])
    app.include_router(internal.router, prefix="/internal/v1", tags=["internal"])
    app.include_router(scenario_router)
    app.include_router(community_router)
    return app

app = create_app()
