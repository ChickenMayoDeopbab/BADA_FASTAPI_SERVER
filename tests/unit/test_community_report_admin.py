from datetime import timedelta

from sqlalchemy import select, update

from app.core.timeutil import now_utc
from app.db.models import CommunityReportORM
from tests.unit.community_env import community_app, create_post


async def _create_report(env, *, author_id: int = 7, reporter_id: int = 8, reason: str = "ABUSE") -> int:
    env.login(author_id)
    post_id = await create_post(env)
    env.login(reporter_id)
    resp = await env.client.post(f"/api/v1/community/posts/{post_id}/reports", json={"reason": reason})
    assert resp.status_code == 201
    return resp.json()["report_id"]


async def test_admin_lists_pending_reports_with_snapshot_and_user_names() -> None:
    async with community_app() as env:
        report_id = await _create_report(env)
        env.login(9)
        resp = await env.client.get("/api/v1/admin/community/reports")

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["reports"][0]["report_id"] == report_id
    assert body["reports"][0]["reporter_name"] == "사용자2"
    assert body["reports"][0]["reported_user_name"] == "사용자1"
    assert body["reports"][0]["content_snapshot"] == {"title": "제목", "content": "내용", "attachments": []}


async def test_report_list_filters_status_and_paginates_by_due_date() -> None:
    async with community_app() as env:
        first = await _create_report(env, author_id=7, reporter_id=8)
        second = await _create_report(env, author_id=8, reporter_id=7, reason="SPAM")
        async with env.sessions() as session:
            await session.execute(
                update(CommunityReportORM)
                .where(CommunityReportORM.report_id == second)
                .values(due_at=now_utc() - timedelta(minutes=1))
            )
            await session.commit()
        env.login(9)
        first_page = await env.client.get("/api/v1/admin/community/reports?size=1")
        second_page = await env.client.get("/api/v1/admin/community/reports?size=1&page=2")
        resolved = await env.client.get("/api/v1/admin/community/reports?status=RESOLVED")

    assert first_page.json()["reports"][0]["report_id"] == second
    assert first_page.json()["reports"][0]["is_overdue"] is True
    assert first_page.json()["has_next"] is True
    assert second_page.json()["reports"][0]["report_id"] == first
    assert resolved.json()["total"] == 0


async def test_admin_gets_report_detail() -> None:
    async with community_app() as env:
        report_id = await _create_report(env)
        env.login(9)
        resp = await env.client.get(f"/api/v1/admin/community/reports/{report_id}")

    assert resp.status_code == 200
    assert resp.json()["report_id"] == report_id
    assert resp.json()["status"] == "PENDING"


async def test_non_admin_cannot_read_reports() -> None:
    async with community_app(user_id=7) as env:
        resp = await env.client.get("/api/v1/admin/community/reports")

    assert resp.status_code == 403
    assert resp.json()["detail"] == "ADMIN_REQUIRED"


async def test_missing_report_returns_404_for_admin() -> None:
    async with community_app(user_id=9) as env:
        resp = await env.client.get("/api/v1/admin/community/reports/999")

    assert resp.status_code == 404


async def test_admin_dismisses_report_without_deleting_content_or_sanctioning_user() -> None:
    async with community_app() as env:
        report_id = await _create_report(env)
        env.login(9)
        resp = await env.client.post(
            f"/api/v1/admin/community/reports/{report_id}/resolve",
            json={"action": "DISMISS", "note": "위반 아님"},
        )
        env.login(8)
        post = await env.client.get(f"/api/v1/community/posts/{resp.json()['target_id']}")

    assert resp.status_code == 200
    assert resp.json()["status"] == "DISMISSED"
    assert resp.json()["resolved_by_user_id"] == 9
    assert resp.json()["resolution_note"] == "위반 아님"
    assert post.status_code == 200
    assert env.spring.moderation_updates == []


async def test_dismiss_rejects_suspended_until() -> None:
    async with community_app() as env:
        report_id = await _create_report(env)
        env.login(9)
        resp = await env.client.post(
            f"/api/v1/admin/community/reports/{report_id}/resolve",
            json={
                "action": "DISMISS",
                "note": "위반 아님",
                "suspended_until": (now_utc() + timedelta(days=1)).isoformat(),
            },
        )
        detail = await env.client.get(f"/api/v1/admin/community/reports/{report_id}")

    assert resp.status_code == 400
    assert detail.json()["status"] == "PENDING"


async def test_admin_bans_author_and_deletes_reported_post() -> None:
    async with community_app() as env:
        report_id = await _create_report(env)
        env.login(9)
        resp = await env.client.post(
            f"/api/v1/admin/community/reports/{report_id}/resolve",
            json={"action": "BAN", "note": "심각한 괴롭힘"},
        )
        target_id = resp.json()["target_id"]
        env.login(8)
        post = await env.client.get(f"/api/v1/community/posts/{target_id}")

    assert resp.status_code == 200
    assert resp.json()["status"] == "RESOLVED"
    assert post.status_code == 404
    assert env.spring.moderation_updates == [
        {
            "user_id": 7,
            "moderation_status": "BANNED",
            "suspended_until": None,
            "reason": "심각한 괴롭힘",
            "actor_user_id": 9,
        }
    ]


async def test_admin_suspends_comment_author_and_deletes_comment_thread() -> None:
    async with community_app(user_id=8) as env:
        post_id = await create_post(env)
        env.login(7)
        root = await env.client.post(f"/api/v1/community/posts/{post_id}/comments", json={"content": "위반 댓글"})
        root_id = root.json()["comment_id"]
        await env.client.post(
            f"/api/v1/community/posts/{post_id}/comments",
            json={"content": "위반 답글", "parent_comment_id": root_id},
        )
        env.login(8)
        report = await env.client.post(f"/api/v1/community/comments/{root_id}/reports", json={"reason": "ABUSE"})
        suspended_until = now_utc() + timedelta(days=1)
        env.login(9)
        resp = await env.client.post(
            f"/api/v1/admin/community/reports/{report.json()['report_id']}/resolve",
            json={"action": "SUSPEND", "note": "일시 정지", "suspended_until": suspended_until.isoformat()},
        )
        env.login(8)
        comments = await env.client.get(f"/api/v1/community/posts/{post_id}/comments")

    assert resp.status_code == 200
    assert comments.json()["comments"] == []
    assert env.spring.moderation_updates[0]["moderation_status"] == "SUSPENDED"
    assert env.spring.moderation_updates[0]["suspended_until"] == suspended_until


async def test_sanction_failure_rolls_back_content_and_report_changes() -> None:
    async with community_app() as env:
        report_id = await _create_report(env)
        env.spring.moderation_succeeds = False
        env.login(9)
        resp = await env.client.post(
            f"/api/v1/admin/community/reports/{report_id}/resolve",
            json={"action": "BAN", "note": "제재 실패 테스트"},
        )
        async with env.sessions() as session:
            report = await session.scalar(select(CommunityReportORM).where(CommunityReportORM.report_id == report_id))
        env.login(8)
        post = await env.client.get(f"/api/v1/community/posts/{report.target_id}")

    assert resp.status_code == 503
    assert report.status == "PENDING"
    assert report.resolved_at is None
    assert post.status_code == 200


async def test_resolving_processed_report_is_idempotent() -> None:
    async with community_app() as env:
        report_id = await _create_report(env)
        env.login(9)
        first = await env.client.post(
            f"/api/v1/admin/community/reports/{report_id}/resolve",
            json={"action": "BAN", "note": "위반"},
        )
        second = await env.client.post(
            f"/api/v1/admin/community/reports/{report_id}/resolve",
            json={"action": "BAN", "note": "위반"},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["status"] == "RESOLVED"
    assert len(env.spring.moderation_updates) == 1


async def test_suspend_resolution_requires_future_deadline() -> None:
    async with community_app() as env:
        report_id = await _create_report(env)
        env.login(9)
        resp = await env.client.post(
            f"/api/v1/admin/community/reports/{report_id}/resolve",
            json={"action": "SUSPEND", "note": "기간 없음"},
        )

    assert resp.status_code == 400


async def test_admin_can_change_user_moderation_status_directly() -> None:
    async with community_app(user_id=9) as env:
        resp = await env.client.patch(
            "/api/v1/admin/community/users/7/moderation-status",
            json={"status": "BANNED", "reason": "반복 위반"},
        )

    assert resp.status_code == 204
    assert env.spring.moderation_updates == [
        {
            "user_id": 7,
            "moderation_status": "BANNED",
            "suspended_until": None,
            "reason": "반복 위반",
            "actor_user_id": 9,
        }
    ]


async def test_non_admin_cannot_resolve_report() -> None:
    async with community_app() as env:
        report_id = await _create_report(env)
        env.login(7)
        resp = await env.client.post(
            f"/api/v1/admin/community/reports/{report_id}/resolve",
            json={"action": "DISMISS", "note": "권한 없음"},
        )

    assert resp.status_code == 403
