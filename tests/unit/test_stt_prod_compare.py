import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core import logging as app_logging
from app.core.metrics import log_metric, now_ms
from app.services.pipeline import VoicePipeline
from app.services.stt import GoogleSTTClient, stt_config_fields
from scripts.stt_compare import wilson
from scripts.stt_noise import prod_compare

ROOT = Path(__file__).resolve().parents[2]


class _Lines(logging.Handler):
    def __init__(self, formatter: logging.Formatter) -> None:
        super().__init__(logging.DEBUG)
        self.setFormatter(formatter)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


@pytest.fixture
def prod_log(monkeypatch):
    """운영(ENV=dev) 텍스트 포맷 그대로 찍는 로그 — setup_logging 이 dictConfig 에 넘기는 설정에서 포맷을 꺼낸다"""
    seen: list[dict] = []
    monkeypatch.setattr(app_logging, "dictConfig", seen.append)
    monkeypatch.setattr(app_logging, "get_settings", lambda: SimpleNamespace(env="dev"))
    app_logging.setup_logging()
    fmt = seen[0]["formatters"][seen[0]["handlers"]["console"]["formatter"]]
    formatter = logging.Formatter(fmt["format"], fmt["datefmt"])
    clock = {"t": 0.0}
    formatter.converter = lambda _secs: time.gmtime(clock["t"])
    handler = _Lines(formatter)
    names = ("app.metrics", "app.services.pipeline", "app.services.stt")
    saved = {}
    for name in names:
        lg = logging.getLogger(name)
        saved[name] = lg.level
        lg.setLevel(logging.DEBUG)
        lg.addHandler(handler)
    monkeypatch.setattr("app.services.stt.SpeechAsyncClient", lambda **kw: object())

    def at(when: str) -> None:
        clock["t"] = datetime.fromisoformat(when).replace(tzinfo=UTC).timestamp()

    yield SimpleNamespace(lines=handler.lines, at=at)
    for name in names:
        logging.getLogger(name).removeHandler(handler)
        logging.getLogger(name).setLevel(saved[name])


def _config(sid: str, denoise: bool) -> None:
    client = GoogleSTTClient(project_id="p", location="asia-northeast1", model="chirp_3", language="ko-KR",
                             denoise=denoise)
    log_metric("stt_config", session_id=sid, **stt_config_fields(client))


def _drop(sid: str, reason: str, text: str, *, state: str = "speaking", since_ms: float | None = None) -> None:
    fake = SimpleNamespace(_session_id=sid, _state=SimpleNamespace(value=state),
                           _turn_final_at=None if since_ms is None else now_ms() - since_ms)
    VoicePipeline._log_final_dropped(fake, reason, text)


def _turn(sid: str, stt_ms: float, response_ms: float) -> None:
    log_metric("voice_turn", session_id=sid, step=1, error=False, stt_ms=stt_ms, response_ms=response_ms,
               turn_total_ms=None)


def _end(sid: str, reason: str) -> None:
    log_metric("session_usage", session_id=sid, user_id=7, scenario_id=3, reason=reason, stt_engine="chirp", turns=1)


def _write_prod_log(prod_log, path: Path) -> Path:
    """on 2세션(10/21) · off 1세션(10/20) · 스위치 이전 1세션(10/19) · 세션 없는 STT 줄"""
    prod_log.at("2026-10-19T09:00:00")
    try:
        raise RuntimeError("grpc 끊김")
    except RuntimeError:
        logging.getLogger("app.services.pipeline").exception("STT 스트리밍 실패", extra={"session_id": "old1"})
    prod_log.at("2026-10-19T10:00:00")
    _turn("old1", 30.0, 900.0)
    prod_log.at("2026-10-20T10:00:00")
    _config("off1", False)
    for ms in (10.0, 20.0, 30.0, 40.0, 50.0):
        _turn("off1", ms, 1000.0)
    _drop("off1", "time_up", "늦은 말", state="listening")
    _drop("off1", "closing", "", state="closing")
    _end("off1", "USER_END")
    prod_log.at("2026-10-21T10:00:00")
    _config("on1", True)
    for ms in (60.0, 70.0, 80.0, 90.0):
        _turn("on1", ms, 1100.0)
    _drop("on1", "empty", "", state="listening")
    _drop("on1", "empty", "", state="listening")
    _drop("on1", "state", "나머지예요", state="thinking", since_ms=800.0)
    _drop("on1", "state", "요", state="thinking", since_ms=1500.0)
    _end("on1", "SCENARIO_DONE")
    prod_log.at("2026-10-21T10:05:00")
    for _ in range(2):
        logging.getLogger("app.services.stt").info("STT 무요청 ABORTED: 스트림 재오픈 대상")
    prod_log.at("2026-10-21T11:00:00")
    _config("on2", True)
    _turn("on2", 100.0, 1200.0)
    _turn("on2", 110.0, 1300.0)
    _drop("on2", "state", "끼어든 말", state="speaking", since_ms=5000.0)
    _end("on2", "ERROR")
    path.write_text("\n".join(prod_log.lines) + "\n", encoding="utf-8")
    return path


def test_log_lines_are_the_prod_text_format(prod_log, tmp_path) -> None:
    lines = _write_prod_log(prod_log, tmp_path / "prod.log").read_text(encoding="utf-8").splitlines()
    config = next(x for x in lines if "metric=stt_config session_id=on1" in x)
    assert config.startswith("2026-10-21 10:00:00.") and "[INFO] app.metrics: metric=stt_config" in config
    ts, row, kind = prod_compare.parse_line(config)
    assert ts == datetime(2026, 10, 21, 10, 0, 0) and kind is None
    assert (row["engine"], row["denoise"], row["snr_threshold"]) == ("chirp", True, 0.0)
    drop = next(x for x in lines if "reason=state" in x and "on1" in x)
    _, row, _ = prod_compare.parse_line(drop)
    assert (row["chars"], row["state"]) == (5.0, "thinking") and 700 < row["since_final_ms"] < 3000
    assert any(x.startswith("Traceback") for x in lines)  # 예외 줄의 traceback 은 시각 없이 붙는다


def test_groups_by_stt_config_and_counts_per_100_turns(prod_log, tmp_path) -> None:
    report = prod_compare.compare(prod_compare.read_logs([str(_write_prod_log(prod_log, tmp_path / "p.log"))]))
    assert list(report["groups"]) == ["off", "on", "none"]
    on, off, none = (report["groups"][k] for k in ("on", "off", "none"))
    assert (on["sessions"], on["turns"], off["sessions"], off["turns"], none["sessions"], none["turns"]) == (
        2, 6, 1, 5, 1, 1)
    d = on["drops"]
    assert (d["empty"]["count"], d["empty"]["per100"]) == (2, pytest.approx(100 * 2 / 6))
    assert d["empty"]["sessions"] == {"k": 1, "n": 2, **dict(zip(("p", "lo", "hi"), wilson(1, 2), strict=True))}
    assert (d["state"]["count"], d["state_text"]["count"], d["state_3s"]["count"]) == (3, 3, 2)
    assert (d["state_3s"]["per100"], d["state_3s"]["sessions"]["k"]) == (pytest.approx(100 * 2 / 6), 1)
    assert d["state"]["sessions"]["k"] == 2 and d["time_up"]["count"] == 0
    assert on["state_by_state"] == {"speaking": 1, "thinking": 2}
    assert on["end_reasons"] == {"ERROR": 1, "SCENARIO_DONE": 1}
    assert (on["error_sessions"]["k"], on["error_sessions"]["n"]) == (1, 2)
    assert on["error_sessions"]["lo"] == pytest.approx(wilson(1, 2)[1])
    assert on["latency"]["stt_ms"]["p50"] == pytest.approx(85.0)
    assert on["latency"]["turn_total_ms"] == {"n": 0, "p50": None, "p90": None}
    o = off["drops"]
    assert (o["time_up"]["per100"], o["closing"]["per100"], o["empty"]["count"]) == (20.0, 20.0, 0)
    assert off["end_reasons"] == {"USER_END": 1} and off["error_sessions"]["k"] == 0
    assert none["end_reasons"] == {"없음": 1} and none["drops"]["empty"]["per100"] == 0.0
    assert report["general"]["aborted_no_requests"] == 2 and report["general"]["stream_failed"] == 1


def test_since_until_pick_sessions_by_first_line_and_general_lines_by_own_time(prod_log, tmp_path) -> None:
    data = prod_compare.read_logs([str(_write_prod_log(prod_log, tmp_path / "p.log"))])
    after = prod_compare.compare(data, since=datetime(2026, 10, 20))
    assert list(after["groups"]) == ["off", "on"] and after["general"]["stream_failed"] == 0
    window = prod_compare.compare(data, since=datetime(2026, 10, 20), until=datetime(2026, 10, 21, 10, 30))
    assert {k: v["sessions"] for k, v in window["groups"].items()} == {"off": 1, "on": 1}
    assert window["general"]["aborted_no_requests"] == 2
    before = prod_compare.compare(data, until=datetime(2026, 10, 20, 10, 0, 0))  # off1 첫 줄 시각 = until → 제외
    assert list(before["groups"]) == ["none"] and before["general"]["stream_failed"] == 1


def test_state_text_counts_only_state_drops_with_characters() -> None:
    def drop(chars):
        return (None, {"metric": "final_dropped", "session_id": "s", "reason": "state", "chars": chars})

    rows = [(None, {"metric": "voice_turn", "session_id": "s"}), drop(0.0), drop(3.0), drop(None)]
    s = prod_compare.compare({"rows": rows, "general": [], "lines": 4, "dup": 0})["groups"]["none"]["drops"]
    assert (s["state"]["count"], s["state_text"]["count"]) == (3, 1)


def test_session_straddling_the_window_is_kept_whole() -> None:
    def row(ts, metric, **kw):
        return (datetime.fromisoformat(ts), {"metric": metric, "session_id": "s", **kw})

    rows = [row("2026-10-21 10:00:00", "stt_config", engine="chirp", denoise=True),
            row("2026-10-21 10:40:00", "voice_turn"), row("2026-10-21 10:41:00", "final_dropped", reason="empty")]
    data = {"rows": rows, "general": [], "lines": 3, "dup": 0}
    inside = prod_compare.compare(data, until=datetime(2026, 10, 21, 10, 30))
    assert (inside["groups"]["on"]["turns"], inside["groups"]["on"]["drops"]["empty"]["count"]) == (1, 1)
    assert prod_compare.compare(data, since=datetime(2026, 10, 21, 10, 30))["groups"] == {}


def test_overlapping_log_files_count_each_line_once(prod_log, tmp_path) -> None:
    path = _write_prod_log(prod_log, tmp_path / "p.log")
    copy = tmp_path / "backup.log"
    copy.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    once = prod_compare.compare(prod_compare.read_logs([str(path)]))
    twice = prod_compare.read_logs([str(path), str(copy)])
    assert twice["dup"] > 0
    assert prod_compare.compare(twice)["groups"] == once["groups"]
    line = "2026-10-21 10:00:00.001 [INFO] app.metrics: metric=final_dropped session_id=s reason=empty chars=0"
    same = tmp_path / "same.log"
    same.write_text(f"{line}\n{line}\n", encoding="utf-8")
    assert len(prod_compare.read_logs([str(same)])["rows"]) == 2  # 한 파일 안의 같은 줄은 둘 다 센다
    assert len(prod_compare.read_logs([str(same), str(same)])["rows"]) == 2


def test_general_patterns_exist_in_app_logs() -> None:
    source = "".join((ROOT / "app" / "services" / f).read_text(encoding="utf-8") for f in ("stt.py", "pipeline.py"))
    for _, pattern in prod_compare.GENERAL:
        assert pattern in source


def test_docker_timestamp_prefix_and_json_lines() -> None:
    text = ("2026-10-21T01:00:00.123456789Z 2026-10-21 10:00:00.123 [INFO] app.metrics: metric=stt_config "
            "session_id=x engine=chirp model=chirp_3 location=asia-northeast1 denoise=False snr_threshold=None")
    ts, row, _ = prod_compare.parse_line(text)
    assert ts == datetime(2026, 10, 21, 1, 0, 0) and row["denoise"] is False and row["snr_threshold"] is None
    js = json.dumps({"asctime": "2026-10-21T10:00:00", "levelname": "INFO", "name": "app.metrics",
                     "message": "metric=voice_turn session_id=y", "metric": "voice_turn", "session_id": "y",
                     "stt_ms": 12.5})
    ts, row, _ = prod_compare.parse_line(js)
    assert ts == datetime(2026, 10, 21, 10, 0, 0) and (row["metric"], row["session_id"], row["stt_ms"]) == (
        "voice_turn", "y", 12.5)
    ts, row, _ = prod_compare.parse_line("2026-10-21T01:00:00.5Z " + js)
    assert ts == datetime(2026, 10, 21, 1, 0, 0) and (row["metric"], row["stt_ms"]) == ("voice_turn", 12.5)
    ts, row, kind = prod_compare.parse_line(json.dumps({"asctime": "2026-10-21T10:00:01",
                                                        "message": "STT 무요청 ABORTED: 스트림 재오픈 대상"}))
    assert row is None and kind == "aborted_no_requests"
    assert prod_compare.parse_line("2026-10-21 10:00:00.1 [INFO] app.services.pipeline: STT FINAL 수신")[1:] == (
        None, None)


def test_gemini_and_mixed_sessions_are_kept_apart() -> None:
    def cfg(sid, engine, denoise=None):
        return (None, {"metric": "stt_config", "session_id": sid, "engine": engine, "denoise": denoise})

    rows = [cfg("g", "gemini_live"), cfg("m", "chirp", True), cfg("m", "chirp", False), cfg("c", "chirp", False),
            cfg("c", "chirp", False), (None, {"metric": "ws_auth", "session_id": "w"})]
    report = prod_compare.compare({"rows": rows, "general": [], "lines": 6, "dup": 0})
    assert {k: v["sessions"] for k, v in report["groups"].items()} == {"off": 1, "gemini": 1, "mixed": 1}
    assert report["groups"]["off"]["drops"]["empty"]["per100"] is None


def test_main_prints_groups_and_writes_json(prod_log, tmp_path, monkeypatch, capsys) -> None:
    path = _write_prod_log(prod_log, tmp_path / "p.log")
    out = tmp_path / "s.json"
    monkeypatch.setattr(sys, "argv", ["prod_compare.py", str(path), "--since", "2026-10-20", "--json-out", str(out)])
    prod_compare.main()
    text = capsys.readouterr().out
    assert "## on — denoise=True  세션 2" in text and "## off" in text and "## none" not in text
    assert "| on | 2 | 6 |" in text and "aborted_no_requests 2" in text
    assert json.loads(out.read_text(encoding="utf-8"))["groups"]["on"]["turns"] == 6


@pytest.mark.parametrize("value", ["2026-10-21T00:00Z", "2026-10-21T09:00+09:00", "2026-10-21 00:00:00+00:00"])
def test_since_until_with_a_time_zone_is_rejected_not_crashed(prod_log, tmp_path, monkeypatch, capsys, value) -> None:
    path = _write_prod_log(prod_log, tmp_path / "p.log")
    for flag in ("--since", "--until"):
        monkeypatch.setattr(sys, "argv", ["prod_compare.py", str(path), flag, value])
        with pytest.raises(SystemExit) as e:
            prod_compare.main()
        assert e.value.code == 2
        assert "시간대 없이" in capsys.readouterr().err


def test_when_reads_naive_iso_times() -> None:
    assert prod_compare._when("2026-10-21") == datetime(2026, 10, 21)
    assert prod_compare._when("2026-10-21 09:30") == datetime(2026, 10, 21, 9, 30)
    assert prod_compare._when("2026-10-21T09:30:05") == datetime(2026, 10, 21, 9, 30, 5)
    with pytest.raises(argparse.ArgumentTypeError, match="ISO"):
        prod_compare._when("어제")


def test_main_until_picks_sessions_before(prod_log, tmp_path, monkeypatch, capsys) -> None:
    path = _write_prod_log(prod_log, tmp_path / "p.log")
    monkeypatch.setattr(sys, "argv", ["prod_compare.py", str(path), "--until", "2026-10-20 10:00"])
    prod_compare.main()
    text = capsys.readouterr().out
    assert "## none" in text and "## off" not in text and "## on" not in text
