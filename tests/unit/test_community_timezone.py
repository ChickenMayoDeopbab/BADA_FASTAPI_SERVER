from datetime import UTC, datetime, timedelta

from app.core.timeutil import KST
from app.db.external import training_records_table
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


async def _attached_started_at(started: datetime) -> str:
    async with community_app(user_id=7) as env:
        async with env.sessions() as db:
            await db.execute(
                training_records_table.insert().values(
                    record_id=901,
                    user_id=7,
                    scenario_name="병원 예약 전화",
                    session_type="PRACTICE",
                    started_at=started,
                    duration_seconds=132,
                    anxiety_score=41,
                )
            )
            await db.commit()

        created = await env.client.post(
            "/api/v1/community/posts",
            json={
                "title": "제목",
                "content": "내용",
                "attachments": [{"kind": "TRAINING_RECORD", "ref_id": 901}],
            },
        )
        assert created.status_code == 201, created.text
        detail = await env.client.get(
            f"/api/v1/community/posts/{created.json()['post_id']}"
        )

    raw = detail.json()["attachments"][0]["training_record"]["started_at"]
    assert raw.endswith(_KST), f"started_at 만 오프셋이 없다: {raw}"
    return raw


async def test_attached_training_record_started_at_reads_spring_kst_wall_clock() -> None:
    started = datetime.now(KST).replace(tzinfo=None)
    raw = await _attached_started_at(started)
    assert _parsed(raw) == started.replace(tzinfo=KST), f"started_at 이 밀렸다: {raw} (기대 {started} KST)"


async def test_attached_training_record_before_switch_reads_utc_wall_clock() -> None:
    raw = await _attached_started_at(datetime(2026, 8, 25, 15, 30))
    assert raw == "2026-08-26T00:30:00+09:00", raw
