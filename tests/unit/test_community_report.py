from datetime import timedelta

from sqlalchemy import func, select

from app.core.timeutil import now_utc
from app.db.models import CommunityReportORM, PostAttachmentORM
from app.schemas.community import CommentCreateRequest
from app.services.community_comment import create_comment
from tests.unit.community_env import community_app, create_post


async def _report_count(env) -> int:
    async with env.sessions() as session:
        return (await session.execute(select(func.count()).select_from(CommunityReportORM))).scalar_one()


async def test_report_post_stores_snapshot_and_due_at() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env, title="신고할 제목", content="신고할 본문")
        async with env.sessions() as session:
            session.add(PostAttachmentORM(post_id=post_id, kind="FILE", ref_id=31, created_at=now_utc()))
            await session.commit()
        env.login(8)

        resp = await env.client.post(f"/api/v1/community/posts/{post_id}/reports", json={"reason": "ABUSE"})
        async with env.sessions() as session:
            row = (await session.execute(select(CommunityReportORM))).scalar_one()

    assert resp.status_code == 201
    assert resp.json()["target_type"] == "POST"
    assert resp.json()["reason"] == "ABUSE"
    assert resp.json()["status"] == "PENDING"
    assert row.reported_user_id == 7
    assert row.content_snapshot == {
        "title": "신고할 제목",
        "content": "신고할 본문",
        "attachments": [{"kind": "FILE", "ref_id": 31}],
    }
    assert row.due_at - row.created_at == timedelta(hours=24)


async def test_report_comment_stores_comment_snapshot() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        async with env.sessions() as session:
            comment, _ = await create_comment(session, post_id, CommentCreateRequest(content="문제 댓글"), 8)
        resp = await env.client.post(
            f"/api/v1/community/comments/{comment.comment_id}/reports", json={"reason": "SPAM"}
        )
        async with env.sessions() as session:
            row = (await session.execute(select(CommunityReportORM))).scalar_one()

    assert resp.status_code == 201
    assert resp.json()["target_type"] == "COMMENT"
    assert row.reported_user_id == 8
    assert row.content_snapshot == {"content": "문제 댓글", "post_id": post_id}


async def test_unknown_report_reason_is_rejected() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        env.login(8)
        resp = await env.client.post(f"/api/v1/community/posts/{post_id}/reports", json={"reason": "UNKNOWN"})

    assert resp.status_code == 422


async def test_reporting_own_content_is_rejected() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        resp = await env.client.post(f"/api/v1/community/posts/{post_id}/reports", json={"reason": "OTHER"})
        report_count = await _report_count(env)

    assert resp.status_code == 400
    assert report_count == 0


async def test_reporting_missing_content_returns_404() -> None:
    async with community_app(user_id=7) as env:
        post_resp = await env.client.post("/api/v1/community/posts/999/reports", json={"reason": "SPAM"})
        comment_resp = await env.client.post("/api/v1/community/comments/999/reports", json={"reason": "SPAM"})

    assert post_resp.status_code == 404
    assert comment_resp.status_code == 404


async def test_reporting_deleted_post_returns_404() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        await env.client.delete(f"/api/v1/community/posts/{post_id}")
        env.login(8)
        resp = await env.client.post(f"/api/v1/community/posts/{post_id}/reports", json={"reason": "SPAM"})

    assert resp.status_code == 404


async def test_reporting_deleted_comment_returns_404() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        async with env.sessions() as session:
            comment, _ = await create_comment(session, post_id, CommentCreateRequest(content="삭제 댓글"), 8)
        env.login(8)
        await env.client.delete(f"/api/v1/community/comments/{comment.comment_id}")
        env.login(7)
        resp = await env.client.post(
            f"/api/v1/community/comments/{comment.comment_id}/reports", json={"reason": "SPAM"}
        )

    assert resp.status_code == 404


async def test_duplicate_report_returns_409() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        env.login(8)
        await env.client.post(f"/api/v1/community/posts/{post_id}/reports", json={"reason": "ABUSE"})
        resp = await env.client.post(f"/api/v1/community/posts/{post_id}/reports", json={"reason": "SPAM"})
        report_count = await _report_count(env)

    assert resp.status_code == 409
    assert report_count == 1


async def test_different_users_can_report_same_target() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        env.login(8)
        first = await env.client.post(f"/api/v1/community/posts/{post_id}/reports", json={"reason": "ABUSE"})
        env.login(9)
        second = await env.client.post(f"/api/v1/community/posts/{post_id}/reports", json={"reason": "HATE"})
        report_count = await _report_count(env)

    assert first.status_code == 201
    assert second.status_code == 201
    assert report_count == 2
