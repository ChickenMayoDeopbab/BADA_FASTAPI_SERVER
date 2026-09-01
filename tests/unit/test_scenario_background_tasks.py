import asyncio
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException

from app.api.v1 import scenario as scenario_api


def _response(scenario_id: int = 77) -> SimpleNamespace:
    return SimpleNamespace(scenario=SimpleNamespace(scenario_id=scenario_id))


async def _create(monkeypatch, svc) -> BackgroundTasks:
    monkeypatch.setattr(scenario_api, "svc_create_custom_scenario", svc)
    background = BackgroundTasks()
    await scenario_api.create_custom_scenario(
        body=SimpleNamespace(), background=background, db=object(), user_id=7
    )
    return background


async def test_creation_runs_thumbnail_and_prebake(monkeypatch) -> None:
    called: list[tuple[str, int]] = []

    async def _thumb(sid: int) -> None:
        called.append(("thumb", sid))

    async def _bake(sid: int) -> None:
        called.append(("bake", sid))

    monkeypatch.setattr(scenario_api, "generate_scenario_thumbnail", _thumb)
    monkeypatch.setattr(scenario_api, "bake_example_audio", _bake)

    async def _svc(_db, _body, _uid):
        return _response(77)

    background = await _create(monkeypatch, _svc)
    await background()

    assert sorted(called) == [("bake", 77), ("thumb", 77)]


async def test_background_tasks_run_concurrently(monkeypatch) -> None:
    order: list[str] = []

    async def _slow_thumb(_sid: int) -> None:
        order.append("thumb-start")
        await asyncio.sleep(0.02)
        order.append("thumb-end")

    async def _bake(_sid: int) -> None:
        await asyncio.sleep(0.005)
        order.append("bake-done")

    monkeypatch.setattr(scenario_api, "generate_scenario_thumbnail", _slow_thumb)
    monkeypatch.setattr(scenario_api, "bake_example_audio", _bake)

    async def _svc(_db, _body, _uid):
        return _response(77)

    background = await _create(monkeypatch, _svc)
    await background()

    assert order.index("bake-done") < order.index("thumb-end"), (
        f"예시 굽기가 썸네일을 기다리고 있다: {order}"
    )


async def test_one_failing_task_does_not_block_the_other(monkeypatch) -> None:
    done: list[str] = []

    async def _boom(_sid: int) -> None:
        raise RuntimeError("썸네일 폭발")

    async def _bake(_sid: int) -> None:
        done.append("bake")

    monkeypatch.setattr(scenario_api, "generate_scenario_thumbnail", _boom)
    monkeypatch.setattr(scenario_api, "bake_example_audio", _bake)

    async def _svc(_db, _body, _uid):
        return _response(77)

    background = await _create(monkeypatch, _svc)
    await background()

    assert done == ["bake"]


async def test_no_tasks_scheduled_when_creation_fails(monkeypatch) -> None:

    async def _svc(_db, _body, _uid):
        raise ValueError("파싱 실패")

    monkeypatch.setattr(scenario_api, "svc_create_custom_scenario", _svc)
    background = BackgroundTasks()
    with pytest.raises(HTTPException):
        await scenario_api.create_custom_scenario(
            body=SimpleNamespace(), background=background, db=object(), user_id=7
        )
    assert background.tasks == []
