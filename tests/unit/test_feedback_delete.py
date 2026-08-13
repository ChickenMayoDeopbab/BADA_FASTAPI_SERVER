from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from app.api.v1 import internal
from app.core.security import require_internal_secret
from app.deps.db import get_db
from app.services.feedback_service import delete_feedback


class _FakeDB:
    def __init__(self, session_ids: set[str]) -> None:
        self.session_ids = set(session_ids)
        self.commits = 0

    async def execute(self, stmt: object) -> SimpleNamespace:
        params = stmt.compile().params
        assert len(params) == 1, params
        sid = next(iter(params.values()))
        removed = sid in self.session_ids
        self.session_ids.discard(sid)
        return SimpleNamespace(rowcount=1 if removed else 0)

    async def commit(self) -> None:
        self.commits += 1



async def test_delete_existing_feedback_returns_true() -> None:
    db = _FakeDB({"sess-1"})

    assert await delete_feedback(db, "sess-1") is True
    assert db.session_ids == set()
    assert db.commits == 1


async def test_delete_missing_feedback_returns_false() -> None:
    db = _FakeDB(set())

    assert await delete_feedback(db, "sess-없음") is False


async def test_delete_db_error_propagates() -> None:
    class _BoomDB:
        async def execute(self, _stmt: object) -> None:
            raise RuntimeError("db down")

        async def commit(self) -> None:
            pass

    with pytest.raises(RuntimeError, match="db down"):
        await delete_feedback(_BoomDB(), "sess-1")



def _make_app(session_ids: set[str], *, real_secret: bool = False) -> FastAPI:
    app = FastAPI()
    app.include_router(internal.router, prefix="/internal/v1")
    if not real_secret:
        app.dependency_overrides[require_internal_secret] = lambda: None
    db = _FakeDB(session_ids)

    async def _fake_db():
        yield db

    app.dependency_overrides[get_db] = _fake_db
    app.state.fake_db = db
    return app


async def _delete(app: FastAPI, path: str, headers: dict | None = None) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.delete(path, headers=headers)


async def test_delete_endpoint_204_removes_row() -> None:
    app = _make_app({"sess-1"})

    resp = await _delete(app, "/internal/v1/feedback/sess-1")

    assert resp.status_code == 204
    assert resp.content == b""
    assert app.state.fake_db.session_ids == set()


async def test_delete_endpoint_204_when_missing_idempotent() -> None:
    app = _make_app(set())

    resp = await _delete(app, "/internal/v1/feedback/sess-1")

    assert resp.status_code == 204


async def test_delete_endpoint_requires_secret() -> None:
    app = _make_app({"sess-1"}, real_secret=True)

    no_header = await _delete(app, "/internal/v1/feedback/sess-1")
    wrong = await _delete(
        app, "/internal/v1/feedback/sess-1", headers={"X-Internal-Secret": "wrong"}
    )
    right = await _delete(
        app,
        "/internal/v1/feedback/sess-1",
        headers={"X-Internal-Secret": "test-internal-secret"},
    )

    assert no_header.status_code == 403
    assert wrong.status_code == 403
    assert right.status_code == 204
