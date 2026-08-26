"""첨부된 시나리오를 내 목록으로 복제 (계획 0037 · F59).

스냅샷이다 — 복제 후 원본이 바뀌거나 지워져도 복제본은 그대로다.
같은 뿌리를 이미 갖고 있으면 새로 만들지 않는다.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from app.db.models import ScenarioORM
from tests.unit.community_env import Env, community_app


def _scenario(user_id: int = 8, **over) -> ScenarioORM:
    now = datetime.now(UTC).replace(tzinfo=None)
    fields = {
        "title": "병원 예약 전화",
        "content": "처음 거는 예약 전화",
        "category": "OTHER",
        "ai_prompt": "You are a hospital desk clerk.",
        "tts_voice_id": "voice-abc",
        "call_target": "○○병원 접수",
        "call_purpose": "진료 예약",
        "script": [{"step": 1, "ai_goal": "인사", "hint": "예약하려고요"}],
        "example_dialogue": [{"speaker": "ai", "text": "네 ○○병원입니다"}],
        "example_audio_url": "examples/1-deadbeef.wav",
        "user_id": user_id,
        "is_custom": True,
        "is_warmup": False,
        "created_at": now,
    }
    fields.update(over)
    return ScenarioORM(**fields)


async def _add(env: Env, row: ScenarioORM) -> int:
    async with env.sessions() as db:
        db.add(row)
        await db.commit()
        await db.refresh(row)
        return row.scenario_id


async def _post_with_scenario(env: Env, scenario_id: int, *, author: int) -> int:
    """author 가 자기 시나리오를 첨부한 글을 만든다."""
    env.login(author)
    resp = await env.client.post(
        "/api/v1/community/posts",
        json={
            "title": "이걸로 연습했어요",
            "content": "내용",
            "attachments": [{"kind": "SCENARIO", "ref_id": scenario_id}],
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["post_id"]


async def _count(env: Env) -> int:
    async with env.sessions() as db:
        return await db.scalar(select(func.count()).select_from(ScenarioORM))


@pytest.mark.asyncio
async def test_copy_puts_a_new_scenario_in_my_list() -> None:
    async with community_app() as env:
        origin = await _add(env, _scenario(user_id=8))
        post_id = await _post_with_scenario(env, origin, author=8)

        env.login(7)
        resp = await env.client.post(f"/api/v1/community/posts/{post_id}/scenario/copy")

        assert resp.status_code == 201, resp.text
        assert resp.json()["already_copied"] is False
        assert await _count(env) == 2

        async with env.sessions() as db:
            copy = await db.get(ScenarioORM, resp.json()["scenario_id"])
        assert copy.user_id == 7
        assert copy.origin_scenario_id == origin
        assert copy.is_custom is True
        assert copy.is_warmup is False
        assert copy.deleted_at is None


@pytest.mark.asyncio
async def test_copy_carries_the_content_but_not_the_audio() -> None:
    """example_dialogue 를 안 가져오면 복제자가 Anthropic 을 다시 부른다.

    example_audio_url 은 키에 scenario_id 가 박혀 있어 복제본에서는 절대 안 맞는다.
    """
    async with community_app() as env:
        origin = await _add(env, _scenario(user_id=8))
        post_id = await _post_with_scenario(env, origin, author=8)

        env.login(7)
        resp = await env.client.post(f"/api/v1/community/posts/{post_id}/scenario/copy")

        async with env.sessions() as db:
            copy = await db.get(ScenarioORM, resp.json()["scenario_id"])
            src = await db.get(ScenarioORM, origin)

        for field in (
            "title", "content", "category", "ai_prompt", "tts_voice_id",
            "call_target", "call_purpose", "script", "example_dialogue",
        ):
            assert getattr(copy, field) == getattr(src, field), field
        assert copy.example_audio_url is None


@pytest.mark.asyncio
async def test_copying_twice_returns_the_same_scenario() -> None:
    async with community_app() as env:
        origin = await _add(env, _scenario(user_id=8))
        post_id = await _post_with_scenario(env, origin, author=8)

        env.login(7)
        first = await env.client.post(f"/api/v1/community/posts/{post_id}/scenario/copy")
        second = await env.client.post(f"/api/v1/community/posts/{post_id}/scenario/copy")

        assert first.status_code == 201
        assert second.status_code == 200
        assert second.json()["already_copied"] is True
        assert second.json()["scenario_id"] == first.json()["scenario_id"]
        assert await _count(env) == 2, "두 번째 호출이 행을 또 만들었다"


@pytest.mark.asyncio
async def test_copy_of_a_copy_still_points_at_the_root() -> None:
    """체인이 길어지면 중복 검사가 뿌리를 놓친다."""
    async with community_app() as env:
        origin = await _add(env, _scenario(user_id=8))
        post_8 = await _post_with_scenario(env, origin, author=8)

        env.login(7)
        copied = (
            await env.client.post(f"/api/v1/community/posts/{post_8}/scenario/copy")
        ).json()["scenario_id"]
        post_7 = await _post_with_scenario(env, copied, author=7)

        env.login(9)
        resp = await env.client.post(f"/api/v1/community/posts/{post_7}/scenario/copy")

        assert resp.status_code == 201, resp.text
        async with env.sessions() as db:
            grandchild = await db.get(ScenarioORM, resp.json()["scenario_id"])
        assert grandchild.origin_scenario_id == origin, "뿌리가 아니라 중간 복제본을 가리킨다"


@pytest.mark.asyncio
async def test_root_owner_reached_through_someone_elses_copy_gets_no_duplicate() -> None:
    """8이 만든 걸 7이 복제해 자기 글에 붙였고, 8이 그 글에서 가져가기를 누른다.

    첨부의 소유자는 7이라 "자기 것" 검사에 안 걸리고, 8의 목록에는
    origin=X 인 행이 없다(8이 가진 건 X 자신이고 X 의 origin 은 null).
    """
    async with community_app() as env:
        origin = await _add(env, _scenario(user_id=8))
        post_8 = await _post_with_scenario(env, origin, author=8)

        env.login(7)
        copied = (
            await env.client.post(f"/api/v1/community/posts/{post_8}/scenario/copy")
        ).json()["scenario_id"]
        post_7 = await _post_with_scenario(env, copied, author=7)

        before = await _count(env)
        env.login(8)
        resp = await env.client.post(f"/api/v1/community/posts/{post_7}/scenario/copy")

        assert resp.status_code == 200, resp.text
        assert resp.json()["already_copied"] is True
        assert resp.json()["scenario_id"] == origin, "원본이 아니라 다른 걸 돌려줬다"
        assert await _count(env) == before, "자기 시나리오의 복제본이 생겼다"


@pytest.mark.asyncio
async def test_cannot_copy_my_own_attachment() -> None:
    async with community_app() as env:
        mine = await _add(env, _scenario(user_id=7))
        post_id = await _post_with_scenario(env, mine, author=7)

        env.login(7)
        resp = await env.client.post(f"/api/v1/community/posts/{post_id}/scenario/copy")

        assert resp.status_code == 400


@pytest.mark.asyncio
async def test_copy_is_refused_once_the_origin_is_deleted() -> None:
    async with community_app() as env:
        origin = await _add(env, _scenario(user_id=8))
        post_id = await _post_with_scenario(env, origin, author=8)

        async with env.sessions() as db:
            row = await db.get(ScenarioORM, origin)
            row.deleted_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=1)
            await db.commit()

        env.login(7)
        resp = await env.client.post(f"/api/v1/community/posts/{post_id}/scenario/copy")

        assert resp.status_code == 410


@pytest.mark.asyncio
async def test_copy_needs_a_scenario_attachment() -> None:
    async with community_app() as env:
        env.login(8)
        created = await env.client.post(
            "/api/v1/community/posts", json={"title": "첨부 없음", "content": "내용"}
        )
        post_id = created.json()["post_id"]

        env.login(7)
        resp = await env.client.post(f"/api/v1/community/posts/{post_id}/scenario/copy")

        assert resp.status_code == 400


@pytest.mark.asyncio
async def test_copy_from_a_missing_post_is_404() -> None:
    async with community_app() as env:
        env.login(7)
        resp = await env.client.post("/api/v1/community/posts/99999/scenario/copy")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_deleting_my_copy_lets_me_take_it_again() -> None:
    """복제본을 지웠으면 다시 가져올 수 있어야 한다. 살아있는 것만 중복으로 본다."""
    async with community_app() as env:
        origin = await _add(env, _scenario(user_id=8))
        post_id = await _post_with_scenario(env, origin, author=8)

        env.login(7)
        first = await env.client.post(f"/api/v1/community/posts/{post_id}/scenario/copy")

        async with env.sessions() as db:
            row = await db.get(ScenarioORM, first.json()["scenario_id"])
            row.deleted_at = datetime.now(UTC).replace(tzinfo=None)
            await db.commit()

        again = await env.client.post(f"/api/v1/community/posts/{post_id}/scenario/copy")

        assert again.status_code == 201, again.text
        assert again.json()["scenario_id"] != first.json()["scenario_id"]
