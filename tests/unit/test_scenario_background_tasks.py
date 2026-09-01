from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException

from app.api.v1 import scenario as scenario_api
from app.services.example_service import bake_example_audio
from app.services.scenario_image_service import generate_scenario_thumbnail


def _response(scenario_id: int = 77) -> SimpleNamespace:
    return SimpleNamespace(scenario=SimpleNamespace(scenario_id=scenario_id))


async def _call(monkeypatch, svc) -> BackgroundTasks:
    monkeypatch.setattr(scenario_api, "svc_create_custom_scenario", svc)
    background = BackgroundTasks()
    await scenario_api.create_custom_scenario(
        body=SimpleNamespace(), background=background, db=object(), user_id=7
    )
    return background


async def test_creation_schedules_thumbnail_and_example_audio(monkeypatch) -> None:
    async def _svc(_db, _body, _uid):
        return _response(77)

    background = await _call(monkeypatch, _svc)

    scheduled = {(t.func, t.args) for t in background.tasks}
    assert (generate_scenario_thumbnail, (77,)) in scheduled
    assert (bake_example_audio, (77,)) in scheduled, "예시 오디오 미리 굽기가 예약돼야 한다"


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
