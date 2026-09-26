import json
from datetime import UTC, datetime

import httpx

from app.core.enums import CommunityReportReason, CommunityReportStatus, CommunityReportTargetType
from app.schemas.community import CommunityReportResponse
from app.services.community_report_alert import CommunityReportAlertService
from tests.unit.community_env import community_app, create_post


def _report() -> CommunityReportResponse:
    return CommunityReportResponse(
        report_id=12,
        target_type=CommunityReportTargetType.POST,
        target_id=34,
        reason=CommunityReportReason.ABUSE,
        status=CommunityReportStatus.PENDING,
        created_at=datetime(2026, 9, 25, 1, tzinfo=UTC),
        due_at=datetime(2026, 9, 26, 1, tzinfo=UTC),
    )


async def test_discord_webhook_receives_report_summary_without_content() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text="ok")

    service = CommunityReportAlertService(
        "https://discord.com/api/webhooks/123/token", transport=httpx.MockTransport(handler)
    )

    sent = await service.notify_report_created(_report())
    payload = json.loads(requests[0].content)

    assert sent is True
    assert requests[0].method == "POST"
    assert requests[0].url.params["wait"] == "true"
    assert payload == {
        "content": "\n".join(
            (
                "새 커뮤니티 신고가 접수되었습니다.",
                "신고 ID: 12",
                "대상: POST #34",
                "사유: ABUSE",
                "처리 기한: 2026-09-26 10:00:00 KST",
                "관리자 신고 목록에서 24시간 이내 처리해 주세요.",
            )
        )
    }


async def test_missing_webhook_skips_alert() -> None:
    assert await CommunityReportAlertService(None).notify_report_created(_report()) is False


async def test_webhook_failure_does_not_raise() -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(500))
    service = CommunityReportAlertService("https://discord.com/api/webhooks/123/token", transport=transport)

    assert await service.notify_report_created(_report()) is False


async def test_successful_reports_schedule_operator_alerts() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        env.login(8)
        comment_response = await env.client.post(
            f"/api/v1/community/posts/{post_id}/comments", json={"content": "신고할 댓글"}
        )
        post_response = await env.client.post(
            f"/api/v1/community/posts/{post_id}/reports", json={"reason": "ABUSE"}
        )
        env.login(7)
        comment_report_response = await env.client.post(
            f"/api/v1/community/comments/{comment_response.json()['comment_id']}/reports", json={"reason": "SPAM"}
        )

    assert post_response.status_code == 201
    assert comment_report_response.status_code == 201
    assert [report.report_id for report in env.report_alerts.reports] == [
        post_response.json()["report_id"],
        comment_report_response.json()["report_id"],
    ]


async def test_rejected_report_does_not_schedule_operator_alert() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        response = await env.client.post(f"/api/v1/community/posts/{post_id}/reports", json={"reason": "ABUSE"})

    assert response.status_code == 400
    assert env.report_alerts.reports == []
