import httpx
import pytest

from app.schemas.frames import EndReason
from app.services import spring_client as spring_mod
from app.services.spring_client import SpringInternalClient


class _Settings:
    spring_boot_internal_url = "http://spring-app:8080"
    internal_secret = "test-secret"


def _client(handler) -> tuple[SpringInternalClient, list]:
    calls: list = []

    def _handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(len(calls), request)

    client = SpringInternalClient(
        _Settings(), transport=httpx.MockTransport(_handler)
    )
    return client, calls


async def _notify(client: SpringInternalClient) -> None:
    await client.notify_session_closed(
        "sess-retry",
        reason=EndReason.USER_END,
        transcript=[{"role": "user", "text": "여보세요"}],
        silence_total=0.0,
    )


@pytest.fixture(autouse=True)
def _no_backoff_delay(monkeypatch):
    monkeypatch.setattr(spring_mod, "_RETRY_BASE_DELAY_SECONDS", 0.0)


@pytest.mark.asyncio
async def test_transient_5xx_then_success(caplog) -> None:
    client, calls = _client(
        lambda n, _req: httpx.Response(500 if n < 3 else 200)
    )
    await _notify(client)
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_all_attempts_fail_swallows_and_logs_error(caplog) -> None:
    import logging

    client, calls = _client(lambda n, _req: httpx.Response(500))
    with caplog.at_level(logging.ERROR, logger="app.services.spring_client"):
        await _notify(client)
    assert len(calls) == 3
    assert any("최종 실패" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_4xx_is_not_retried() -> None:
    client, calls = _client(lambda n, _req: httpx.Response(400))
    await _notify(client)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_network_error_is_retried() -> None:
    def _handler(n: int, req: httpx.Request) -> httpx.Response:
        if n < 2:
            raise httpx.ConnectError("connection refused", request=req)
        return httpx.Response(200)

    client, calls = _client(_handler)
    await _notify(client)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_community_callback_failure_is_retried_and_swallowed(caplog) -> None:
    import logging

    client, calls = _client(lambda _n, _req: httpx.Response(500))
    with caplog.at_level(logging.ERROR, logger="app.services.spring_client"):
        await client.notify_community_notification(
            notification_type="COMMENT",
            recipient_user_id=7,
            actor_user_id=8,
            post_id=10,
            comment_id=25,
        )

    assert len(calls) == 3
    assert any("커뮤니티 알림 콜백 최종 실패" in record.getMessage() for record in caplog.records)
