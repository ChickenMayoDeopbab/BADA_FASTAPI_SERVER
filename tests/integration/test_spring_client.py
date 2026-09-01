import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

from app.schemas.frames import EndReason
from app.schemas.training_analysis import (
    AnalysisQualityStatus,
    TrainingAnalysisPayload,
)
from app.services.spring_client import SpringInternalClient

_SECRET = "test-internal-secret"


class _CapturingHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        self.server.captured = {
            "path": self.path,
            "headers": dict(self.headers),
            "body": json.loads(body) if body else None,
        }
        self.send_response(self.server.status_code)
        self.end_headers()

    def log_message(self, *args) -> None:
        pass

class _MockSpring:
    def __init__(self, status_code: int = 200) -> None:
        self._server = HTTPServer(("127.0.0.1", 0), _CapturingHandler)
        self._server.status_code = status_code
        self._server.captured = None
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    @property
    def captured(self) -> dict | None:
        return self._server.captured

    def __enter__(self) -> "_MockSpring":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()


def _client_for(base_url: str) -> SpringInternalClient:
    settings = SimpleNamespace(
        spring_boot_internal_url=base_url,
        internal_secret=_SECRET,
    )
    return SpringInternalClient(settings)


async def test_notify_session_closed_sends_correct_request() -> None:
    with _MockSpring(status_code=200) as spring:
        client = _client_for(spring.base_url)
        analysis = TrainingAnalysisPayload(
            stability_score=82.5,
            conversation_score=75.0,
            fluency_score=80.0,
            user_speech_duration_ms=18200,
            ai_speech_duration_ms=24100,
            server_wait_duration_ms=3200,
            valid_user_turn_count=4,
            user_tremor_duration_ms=900,
            user_sustained_speech_duration_ms=5100,
            completed_script_steps=3,
            script_step_count=4,
            analysis_quality_status=AnalysisQualityStatus.PASS,
            analysis_exclusion_reason=None,
            analyzer_version="SPEECH_ANALYZER_V1",
            analysis_policy_version="ANALYSIS_POLICY_V1",
        )
        await client.notify_session_closed(
            "sess-123",
            reason=EndReason.SCENARIO_DONE,
            transcript=[
                {"role": "user", "text": "여보세요"},
                {"role": "assistant", "text": "네, 안녕하세요"},
            ],
            silence_total=3.5,
            shake_count=2,
            recording_key="recordings/sess-123/a1b2c3d4.wav",
            good_segments=[{"start": 1.0, "end": 4.0, "good_point": "잘 이야기했어요"}],
            session_type="SCENARIO",
            analysis=analysis,
        )

        captured = spring.captured
        assert captured is not None, "mock Spring이 요청을 못 받음"
        assert captured["path"] == "/internal/v1/sessions/sess-123/closed"
        assert captured["headers"].get("X-Internal-Secret") == _SECRET
        assert captured["body"] == {
            "type": "SCENARIO",
            "reason": "SCENARIO_DONE",
            "transcript": [
                {"role": "user", "text": "여보세요"},
                {"role": "assistant", "text": "네, 안녕하세요"},
            ],
            "silence_total": 3.5,
            "shake_count": 2,
            "recording_key": "recordings/sess-123/a1b2c3d4.wav",
            "good_segments": [
                {"start": 1.0, "end": 4.0, "good_point": "잘 이야기했어요"}
            ],
            "analysis": {
                "stability_score": 82.5,
                "conversation_score": 75.0,
                "fluency_score": 80.0,
                "user_speech_duration_ms": 18200,
                "ai_speech_duration_ms": 24100,
                "server_wait_duration_ms": 3200,
                "valid_user_turn_count": 4,
                "user_tremor_duration_ms": 900,
                "user_sustained_speech_duration_ms": 5100,
                "completed_script_steps": 3,
                "script_step_count": 4,
                "analysis_quality_status": "PASS",
                "analysis_exclusion_reason": None,
                "analyzer_version": "SPEECH_ANALYZER_V1",
                "analysis_policy_version": "ANALYSIS_POLICY_V1",
            },
        }


async def test_notify_community_notification_sends_correct_request() -> None:
    with _MockSpring(status_code=200) as spring:
        client = _client_for(spring.base_url)

        await client.notify_community_notification(
            notification_type="REPLY",
            recipient_user_id=7,
            actor_user_id=8,
            post_id=10,
            comment_id=25,
        )

        captured = spring.captured
        assert captured is not None, "mock Spring이 요청을 못 받음"
        assert captured["path"] == "/internal/v1/notifications/community"
        assert captured["headers"].get("X-Internal-Secret") == _SECRET
        assert captured["body"] == {
            "type": "REPLY",
            "recipientUserId": 7,
            "actorUserId": 8,
            "postId": 10,
            "commentId": 25,
        }


async def test_trailing_slash_in_base_url_is_normalized() -> None:
    with _MockSpring(status_code=200) as spring:
        client = _client_for(spring.base_url + "/")
        await client.notify_session_closed(
            "abc", reason=EndReason.USER_END, transcript=[], silence_total=0
        )
        assert spring.captured["path"] == "/internal/v1/sessions/abc/closed"
        assert spring.captured["body"]["recording_key"] is None
        assert spring.captured["body"]["analysis"] is None


async def test_server_error_is_swallowed() -> None:
    with _MockSpring(status_code=500) as spring:
        client = _client_for(spring.base_url)
        await client.notify_session_closed(
            "x", reason=EndReason.ERROR, transcript=[], silence_total=0
        )


async def test_unreachable_server_is_swallowed() -> None:
    with _MockSpring() as spring:
        dead_url = spring.base_url
    client = _client_for(dead_url)
    await client.notify_session_closed(
        "y", reason=EndReason.TIMEOUT, transcript=[], silence_total=0
    )
