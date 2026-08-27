from datetime import UTC, datetime, timedelta

from tests.unit.community_env import community_app, create_post

_KST = "+09:00"
_SLACK = timedelta(seconds=10)


def _parsed(raw: str) -> datetime:
    parsed = datetime.fromisoformat(raw)
    assert parsed.tzinfo is not None, f"오프셋이 없다: {raw}"
    return parsed


async def test_post_created_at_carries_kst_offset() -> None:
    async with community_app(user_id=7) as env:
        resp = await env.client.post(
            "/api/v1/community/posts", json={"title": "제목", "content": "내용"}
        )

    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["created_at"].endswith(_KST), f"created_at 오프셋 없음: {body['created_at']}"
    assert body["updated_at"].endswith(_KST), f"updated_at 오프셋 없음: {body['updated_at']}"


async def test_post_created_at_points_at_the_real_instant() -> None:
    before = datetime.now(UTC)
    async with community_app(user_id=7) as env:
        resp = await env.client.post(
            "/api/v1/community/posts", json={"title": "제목", "content": "내용"}
        )
    after = datetime.now(UTC)

    created = _parsed(resp.json()["created_at"])
    assert before - _SLACK <= created <= after + _SLACK, (
        f"실제 생성 순간과 어긋난다: {created.isoformat()} (기대 범위 {before}~{after})"
    )


async def test_post_detail_and_list_carry_kst_offset() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)

        detail = await env.client.get(f"/api/v1/community/posts/{post_id}")
        listing = await env.client.get("/api/v1/community/posts")
        mine = await env.client.get("/api/v1/community/me/posts")

    assert detail.json()["created_at"].endswith(_KST)
    assert listing.json()["posts"][0]["created_at"].endswith(_KST)
    assert mine.json()["posts"][0]["created_at"].endswith(_KST)


async def test_comment_created_at_carries_kst_offset() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)

        created = await env.client.post(
            f"/api/v1/community/posts/{post_id}/comments", json={"content": "댓글"}
        )
        listing = await env.client.get(f"/api/v1/community/posts/{post_id}/comments")

    assert created.status_code == 201, created.text
    assert created.json()["created_at"].endswith(_KST)
    assert listing.json()["comments"][0]["created_at"].endswith(_KST)


async def test_comment_created_at_points_at_the_real_instant() -> None:
    before = datetime.now(UTC)
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        resp = await env.client.post(
            f"/api/v1/community/posts/{post_id}/comments", json={"content": "댓글"}
        )
    after = datetime.now(UTC)

    created = _parsed(resp.json()["created_at"])
    assert before - _SLACK <= created <= after + _SLACK
