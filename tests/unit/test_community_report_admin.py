from datetime import timedelta

from sqlalchemy import update

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
