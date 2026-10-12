import asyncio
import json
import logging
import logging.config
import time
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
import websockets

import app.core.logging as app_logging
from app.core.logging import setup_logging

_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiI3IiwidHlwZSI6IkFDQ0VTUyJ9.c2lnbmF0dXJlLXZhbHVl"
_CLIENT = ("10.0.0.1", 51234)


def _loggers() -> list[logging.Logger]:
    manager = logging.root.manager
    return [logging.root] + [lg for lg in manager.loggerDict.values() if isinstance(lg, logging.Logger)]


@pytest.fixture(autouse=True)
def _restore_logging(monkeypatch):
    monkeypatch.setattr(logging.config, "_clearExistingHandlers", lambda: None)
    saved = {id(lg): (lg.handlers[:], lg.filters[:], lg.level, lg.propagate, lg.disabled) for lg in _loggers()}
    yield
    for lg in _loggers():
        handlers, filters, level, propagate, disabled = saved.get(id(lg), ([], [], logging.NOTSET, True, False))
        lg.handlers[:] = handlers
        lg.filters[:] = filters
        lg.setLevel(level)
        lg.propagate = propagate
        lg.disabled = disabled


def _setup(capsys) -> None:
    capsys.readouterr()
    setup_logging()


def test_ws_accepted_line_hides_token(capsys) -> None:
    _setup(capsys)
    logging.getLogger("uvicorn.error").info(
        '%s - "WebSocket %s" [accepted]', _CLIENT, f"/ws/voice/s1?token={_JWT}"
    )

    out = capsys.readouterr().out
    assert _JWT not in out
    assert "uvicorn.error: ('10.0.0.1', 51234) - \"WebSocket /ws/voice/s1?token=[REDACTED]\" [accepted]" in out


def test_ws_rejected_line_keeps_other_params(capsys) -> None:
    _setup(capsys)
    logging.getLogger("uvicorn.error").info(
        '%s - "WebSocket %s" 403', _CLIENT, f"/ws/voice/s1?lang=ko&token={_JWT}"
    )

    out = capsys.readouterr().out
    assert _JWT not in out
    assert '"WebSocket /ws/voice/s1?lang=ko&token=[REDACTED]" 403' in out


def test_access_line_hides_token_and_keeps_following_params(capsys) -> None:
    _setup(capsys)
    logging.getLogger("uvicorn.access").info(
        '%s - "%s %s HTTP/%s" %d', "10.0.0.1:51234", "GET", f"/ws/voice/s1?token={_JWT}&x=1", "1.1", 404
    )

    out = capsys.readouterr().out
    assert _JWT not in out
    assert '"GET /ws/voice/s1?token=[REDACTED]&x=1 HTTP/1.1" 404' in out


def test_non_jwt_token_value_in_query_is_hidden(capsys) -> None:
    _setup(capsys)
    logging.getLogger("uvicorn.error").info('%s - "WebSocket %s" 403', _CLIENT, "/ws/voice/s1?TOKEN=opaque-123")

    out = capsys.readouterr().out
    assert "opaque-123" not in out
    assert '"WebSocket /ws/voice/s1?TOKEN=[REDACTED]" 403' in out


def test_jwt_outside_query_is_hidden(capsys) -> None:
    _setup(capsys)
    logging.getLogger("app.anything").warning("Authorization: Bearer %s", _JWT)

    out = capsys.readouterr().out
    assert _JWT not in out
    assert "Authorization: Bearer [REDACTED]" in out


def test_every_secret_in_one_message_is_hidden(capsys) -> None:
    unsigned = "eyJhbGciOiJub25lIn0.eyJzdWIiOiI3In0."
    _setup(capsys)
    logging.getLogger("app.x").warning(
        "a ?token=one&token=two b %s %s c %s d ?token=val\\\" e", _JWT, _JWT, unsigned
    )

    out = capsys.readouterr().out
    assert "one" not in out and "two" not in out and _JWT not in out and unsigned not in out
    assert (
        'app.x: a ?token=[REDACTED]&token=[REDACTED] b [REDACTED] [REDACTED] c [REDACTED] d ?token=[REDACTED]\\" e'
        in out
    )


def test_long_eyj_run_is_scanned_in_linear_time() -> None:
    record = logging.LogRecord(
        "uvicorn.access", logging.INFO, __file__, 0,
        '%s - "%s %s HTTP/%s" %d', ("10.0.0.1:51234", "GET", "/x?q=" + "eyJ" * 21845, "1.1", 404), None,
    )

    start = time.perf_counter()
    app_logging.RedactSecretsFilter().filter(record)
    assert time.perf_counter() - start < 0.2


def test_record_propagated_from_unconfigured_logger_is_hidden(capsys) -> None:
    _setup(capsys)
    logging.getLogger("websockets.client").warning("> GET /stream?token=%s HTTP/1.1", _JWT)

    out = capsys.readouterr().out
    assert _JWT not in out
    assert "> GET /stream?token=[REDACTED] HTTP/1.1" in out


def test_lines_without_secrets_are_unchanged(capsys) -> None:
    _setup(capsys)
    logging.getLogger("uvicorn.error").info("connection open")
    logging.getLogger("uvicorn.access").info(
        '%s - "%s %s HTTP/%s" %d', "10.0.0.1:51234", "GET", "/community/posts?page=1&size=20", "1.1", 200
    )
    logging.getLogger("app.core.metrics").info("metric=llm_usage input_tokens=%d output_tokens=%d", 420, 12)
    logging.getLogger("app.x").info("progress %d%% of %s", 50, "token_user_id=7")

    lines = [line.split("] ", 1)[1] for line in capsys.readouterr().out.splitlines()]
    assert lines == [
        "uvicorn.error: connection open",
        'uvicorn.access: 10.0.0.1:51234 - "GET /community/posts?page=1&size=20 HTTP/1.1" 200',
        "app.core.metrics: metric=llm_usage input_tokens=420 output_tokens=12",
        "app.x: progress 50% of token_user_id=7",
    ]


def test_broken_log_call_does_not_raise_to_caller(capsys) -> None:
    _setup(capsys)
    logging.getLogger("app.x").info("%d", "not-a-number")

    assert "Logging error" in capsys.readouterr().err


def test_json_formatter_message_hides_token(capsys, monkeypatch) -> None:
    monkeypatch.setattr(app_logging, "get_settings", lambda: SimpleNamespace(env="prod"))
    _setup(capsys)
    logging.getLogger("uvicorn.error").info(
        '%s - "WebSocket %s" [accepted]', _CLIENT, f"/ws/voice/s1?token={_JWT}"
    )

    out = capsys.readouterr().out
    assert _JWT not in out
    record = json.loads(out.strip().splitlines()[-1])
    assert record["message"] == "('10.0.0.1', 51234) - \"WebSocket /ws/voice/s1?token=[REDACTED]\" [accepted]"


async def test_real_uvicorn_handshake_lines_hide_token(capsys) -> None:
    from app.deps.redis import get_redis
    from app.main import app

    _setup(capsys)
    app.dependency_overrides[get_redis] = lambda: None
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, lifespan="off", log_config=None))
    task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            assert not task.done(), "uvicorn 이 뜨지 못함"
            await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        base = f"127.0.0.1:{port}/ws/voice/s1"

        async with websockets.connect(f"ws://{base}?token={_JWT}") as ws:
            with pytest.raises(websockets.ConnectionClosed):
                await ws.recv()
        with pytest.raises(websockets.InvalidStatus):
            async with websockets.connect(f"ws://{base}?Tok={_JWT}"):
                pass
        async with httpx.AsyncClient() as client:
            assert (await client.get(f"http://{base}?token={_JWT}")).status_code == 404
    finally:
        server.should_exit = True
        await task
        app.dependency_overrides.clear()

    out = capsys.readouterr().out
    assert _JWT not in out
    assert '"WebSocket /ws/voice/s1?token=[REDACTED]" [accepted]' in out
    assert '"WebSocket /ws/voice/s1?Tok=[REDACTED]" 403' in out
    assert '"GET /ws/voice/s1?token=[REDACTED] HTTP/1.1" 404' in out
