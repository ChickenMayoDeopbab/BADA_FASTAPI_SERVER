from types import SimpleNamespace

import httpx
import pytest
from fastapi import BackgroundTasks, FastAPI, Response

from app.api.v1 import community, internal
from app.services import example_service


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(internal.router, prefix="/internal/v1")
    return app


async def _post(path: str, headers: dict | None = None) -> httpx.Response:
    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(path, headers=headers or {})


_SECRET = {"X-Internal-Secret": "test-internal-secret"}


async def test_example_ops_require_the_internal_secret() -> None:
    assert (await _post("/internal/v1/examples/presets/prebake")).status_code == 403


async def test_prebake_route_reports_joined(monkeypatch) -> None:
    monkeypatch.setattr(example_service, "trigger_prebake", lambda: True)
    resp = await _post("/internal/v1/examples/presets/prebake", _SECRET)
    assert resp.status_code == 202 and resp.json() == {"joined": True}


async def test_prebake_route_maps_refusal(monkeypatch) -> None:
    def _off():
        raise example_service.ExampleOpError(409, "꺼짐")

    monkeypatch.setattr(example_service, "trigger_prebake", _off)
    resp = await _post("/internal/v1/examples/presets/prebake", _SECRET)
    assert resp.status_code == 409 and resp.json()["detail"] == "꺼짐"


@pytest.mark.parametrize(
    ("path", "name", "args"),
    [
        ("/internal/v1/examples/presets/3/rebake", "rebake_preset", ()),
        ("/internal/v1/examples/presets/3/restore?prev=examples/prev/a.wav.T", "restore_preset",
         ("examples/prev/a.wav.T",)),
        ("/internal/v1/examples/presets/3/reject?restore_key=examples/3-old.wav", "reject_preset",
         ("examples/3-old.wav",)),
    ],
)
async def test_preset_routes_pass_arguments_and_map_errors(monkeypatch, path, name, args) -> None:
    seen: dict = {}

    async def _ok(sid, *args, **kw):
        seen.update(sid=sid, args=args, kw=kw)
        return {"ok": True}

    monkeypatch.setattr(example_service, name, _ok)
    resp = await _post(path, _SECRET)
    assert resp.status_code in (200, 202) and resp.json() == {"ok": True}
    assert seen["sid"] == 3 and seen["args"] == args

    async def _refuse(*_a, **_k):
        raise example_service.ExampleOpError(503, "백업 실패")

    monkeypatch.setattr(example_service, name, _refuse)
    resp = await _post(path, _SECRET)
    assert resp.status_code == 503


async def _copy(monkeypatch, *, already: bool, switch: bool) -> list:
    scheduled: list = []
    scenario = SimpleNamespace(scenario_id=77, title="t", category="ETC")

    async def _svc_copy(_db, _post_id, _user_id):
        return scenario, already

    monkeypatch.setattr(community, "svc_copy_scenario", _svc_copy)
    monkeypatch.setattr(community, "get_settings", lambda: SimpleNamespace(example_qwen_ns=switch))
    background = BackgroundTasks()
    await community.copy_attached_scenario(5, Response(), background, db=None, user_id=7)
    scheduled.extend((task.func, task.args) for task in background.tasks)
    return scheduled


async def test_copy_schedules_bake_only_when_switch_on_and_new(monkeypatch) -> None:
    assert await _copy(monkeypatch, already=False, switch=True) == [(community.bake_example_audio, (77, "copy"))]
    assert await _copy(monkeypatch, already=True, switch=True) == []
    assert await _copy(monkeypatch, already=False, switch=False) == []
