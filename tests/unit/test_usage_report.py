import asyncio
import gzip
import importlib.util
import json
import sys
from datetime import UTC
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.pricing import (
    ELEVEN_PLAN,
    FX,
    STT_ENGINE_MODEL,
    cost_lines,
    get_price,
    llm_cost_lines,
    session_cost_lines,
    tts_cost_lines,
)
from app.db.base import Base
from app.services.usage_service import record_event

_SPEC = importlib.util.spec_from_file_location(
    "usage_report", Path(__file__).resolve().parents[2] / "scripts" / "usage_report.py"
)
usage_report = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = usage_report
_SPEC.loader.exec_module(usage_report)

KST = ZoneInfo("Asia/Seoul")
M = 1_000_000

_SESSION = {
    "session_id": "s1", "user_id": 77, "scenario_id": 3, "reason": "USER_END",
    "stt_engine": "chirp", "llm_model": "gemini-3.5-flash-lite", "tts_model": "eleven_flash_v2_5",
    "turns": 4, "stt_sec": 12.5,
    "llm_prompt_tokens": 4000, "llm_cached_tokens": 3000, "llm_output_tokens": 200, "llm_thought_tokens": 50,
    "feedback_prompt_tokens": 0, "feedback_cached_tokens": 0, "feedback_output_tokens": 0,
    "feedback_thought_tokens": 0,
    "tts_chars_eleven": 120, "tts_chars_qwen": 300, "tts_audio_sec_eleven": 4.0, "tts_audio_sec_qwen": 9.0,
}
_LLM = {
    "provider": "anthropic", "model": "claude-sonnet-4-6", "purpose": "scenario_gen",
    "user_id": 7, "scenario_id": 42, "attempt": 1, "ok": True, "images": 0,
    "input_tokens": 700, "output_tokens": 220, "cache_read_tokens": 0, "cache_write_tokens": 0, "thought_tokens": 0,
}


def _by_item(lines):
    return {ln.item: ln for ln in lines}


def test_session_cost_matches_hand_calculation() -> None:
    lines = _by_item(session_cost_lines(_SESSION))
    assert lines["stt_sec[chirp]"].usd_low == pytest.approx(12.5 * 0.016 / 60)
    assert lines["llm_input_tokens"].quantity == 1000, "prompt − cached"
    assert lines["llm_input_tokens"].usd_low == pytest.approx(1000 * 0.30 / M)
    assert lines["llm_cached_tokens"].usd_low == pytest.approx(3000 * 0.03 / M)
    assert lines["llm_output_tokens"].quantity == 250, "candidates + thoughts"
    assert lines["llm_output_tokens"].usd_low == pytest.approx(250 * 2.50 / M)
    eleven = lines["tts_chars[eleven]"]
    assert (eleven.usd_low, eleven.usd_high) == pytest.approx((120 * 0.5 * 6 / 30000, 120 * 1.0 * 6 / 30000))
    payg = lines["tts_chars[eleven] PAYG 참고"]
    assert payg.reference and payg.usd_low == pytest.approx(120 / 1000 * 0.05)
    qwen = lines["tts_chars[qwen]"]
    assert qwen.priced and qwen.usd_low == 0.0 and qwen.quantity == 300
    assert all(ln.priced for ln in lines.values())
    assert eleven.krw_high == pytest.approx(eleven.usd_high * FX.krw_per_usd)


def test_anthropic_llm_cost_and_image_cost() -> None:
    llm = _by_item(llm_cost_lines(_LLM))
    assert llm["input_tokens[scenario_gen]"].usd_low == pytest.approx(700 * 3.0 / M)
    assert llm["output_tokens[scenario_gen]"].usd_low == pytest.approx(220 * 15.0 / M)
    assert "cache_read_tokens[scenario_gen]" not in llm, "0 수량 캐시 줄은 생략"

    image = _by_item(llm_cost_lines({
        "provider": "gemini", "model": "gemini-2.5-flash-image", "purpose": "thumbnail_image",
        "images": 1, "input_tokens": 30, "output_tokens": 1290,
    }))
    assert image["image[thumbnail_image]"].usd_low == pytest.approx(0.039)
    assert image["input_tokens[thumbnail_image]"].priced is False, "이미지 모델 입력 토큰은 단가 미조회"
    assert "output_tokens[thumbnail_image]" not in image, "이미지 단가에 포함"


def test_unknown_model_is_unpriced_not_zero() -> None:
    lines = llm_cost_lines({"provider": "gemini", "model": "gemini-9-flash", "purpose": "x",
                            "input_tokens": 10, "output_tokens": 5})
    assert lines and all(ln.priced is False and ln.usd_low is None for ln in lines)
    session = _by_item(session_cost_lines({**_SESSION, "llm_model": "gemini-9-flash"}))
    assert session["llm_input_tokens"].priced is False and session["stt_sec[chirp]"].priced is True


def test_tts_usage_lines() -> None:
    eleven = _by_item(tts_cost_lines({"engine": "eleven", "model": "eleven_flash_v2_5",
                                      "purpose": "example_audio", "chars": 1000}))
    ln = eleven["example_audio_chars[eleven]"]
    assert (ln.usd_low, ln.usd_high) == pytest.approx((0.1, 0.2))
    qwen = tts_cost_lines({"engine": "qwen", "model": "qwen", "purpose": "example_audio", "chars": 50})
    assert qwen[0].usd_low == 0.0 and qwen[0].priced
    with pytest.raises(ValueError):
        cost_lines("voice_turn", {})


_PROD_MODELS = {
    "llm_realtime_model": "gemini-3.5-flash-lite",
    "llm_analysis_model": "claude-sonnet-4-6",
    "elevenlabs_model": "eleven_flash_v2_5",
}


def _config_default(name: str) -> str:
    from app.core.config import Settings

    return Settings.model_fields[name].default


@pytest.mark.parametrize("source", ["config_default", "prod_env"])
def test_every_configured_model_has_price_rows(source) -> None:
    pick = _config_default if source == "config_default" else _PROD_MODELS.__getitem__
    realtime, analysis, eleven = pick("llm_realtime_model"), pick("llm_analysis_model"), pick("elevenlabs_model")
    for item in ("input_tokens", "cached_tokens", "output_tokens"):
        assert get_price("gemini", realtime, item), (realtime, item)
    for item in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
        assert get_price("anthropic", analysis, item), (analysis, item)
    assert get_price("eleven", eleven, "chars_payg"), eleven
    assert get_price("gemini", _config_default("gemini_image_model"), "image")
    engine = _config_default("stt_engine")
    provider, model = STT_ENGINE_MODEL[engine]
    if engine == "chirp":
        assert model == _config_default("google_stt_model")
    assert get_price(provider, model, "stt_sec")


def test_llm_lines_are_priced_for_the_config_default_analysis_model() -> None:
    lines = llm_cost_lines({**_LLM, "model": _config_default("llm_analysis_model")})
    assert lines and all(ln.priced for ln in lines), "config 기본 모델이 단가 미설정으로 빠지면 안 된다"


_P = "[INFO] app.metrics: metric="
_LOG = "\n".join([
    " ".join([
        f"2026-09-15 11:36:00.123 {_P}session_usage session_id=s1 user_id=77 scenario_id=3 reason=USER_END",
        "stt_engine=chirp llm_model=gemini-3.5-flash-lite tts_model=eleven_flash_v2_5 turns=4 stt_sec=12.5",
        "llm_prompt_tokens=4000 llm_cached_tokens=3000 llm_output_tokens=200 llm_thought_tokens=None",
        "feedback_prompt_tokens=0 feedback_cached_tokens=0 feedback_output_tokens=0 feedback_thought_tokens=0",
        "tts_chars_eleven=120 tts_chars_qwen=0 tts_audio_sec_eleven=4.0 tts_audio_sec_qwen=0.0",
    ]),
    " ".join([
        f"2026-09-30 15:30:00.000 {_P}llm_usage provider=anthropic model=claude-sonnet-4-6 purpose=scenario_gen",
        "user_id=7 scenario_id=42 attempt=1 ok=True images=0 duration_ms=None input_tokens=700 output_tokens=220",
        "cache_read_tokens=0 cache_write_tokens=0 thought_tokens=0",
    ]),
    " ".join([
        f"2026-09-15 15:30:01.000 {_P}tts_usage engine=qwen model=qwen purpose=example_audio user_id=None",
        "scenario_id=1 trigger=prebake ok=True chars=33 audio_sec=4.2 turns=2",
    ]),
    " ".join([
        f"2026-09-15 15:30:02.000 {_P}llm_usage provider=gemini model=gemini-9-flash purpose=x user_id=77",
        "scenario_id=None attempt=1 ok=True images=0 duration_ms=None input_tokens=10 output_tokens=5",
        "cache_read_tokens=0 cache_write_tokens=0 thought_tokens=0",
    ]),
    "2026-09-15 15:30:03.000 [INFO] app.services.pipeline: 턴 완료 step=1 user='x' ai='y'",
    f"2026-09-15 15:30:04.000 {_P}voice_turn session_id=s1 step=1",
]) + "\n"


def _write_log(tmp_path: Path, *, gz: bool = False) -> Path:
    path = tmp_path / ("app.log.gz" if gz else "app.log")
    if gz:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(_LOG)
    else:
        path.write_text(_LOG, encoding="utf-8")
    return path


def test_log_parser_extracts_timestamp_kind_and_types(tmp_path) -> None:
    events = usage_report.load_from_logs([_write_log(tmp_path)])
    assert [e.kind for e in events] == ["session_usage", "llm_usage", "tts_usage", "llm_usage"]
    s = events[0]
    assert s.ts_utc.isoformat() == "2026-09-15T11:36:00+00:00"
    assert s.payload["user_id"] == 77 and s.payload["stt_sec"] == 12.5
    assert s.payload["llm_thought_tokens"] is None and s.payload["llm_model"] == "gemini-3.5-flash-lite"
    assert events[1].payload["ok"] is True
    assert events[2].user_key == usage_report.UNATTRIBUTED


def test_log_parser_reads_gzip_and_log_tz(tmp_path) -> None:
    events = usage_report.load_from_logs([_write_log(tmp_path, gz=True)], ZoneInfo("Asia/Seoul"))
    assert len(events) == 4
    assert events[0].ts_utc.isoformat() == "2026-09-15T02:36:00+00:00", "KST 로그 → UTC"


def test_aggregate_groups_by_user_and_kst_month(tmp_path) -> None:
    events = usage_report.load_from_logs([_write_log(tmp_path)])
    groups = usage_report.aggregate(events, tz=KST)
    assert set(groups) == {(77, "2026-09"), (7, "2026-10"), (usage_report.UNATTRIBUTED, "2026-09")}
    g = groups[(77, "2026-09")]
    assert g.sessions == 1 and g.quantities["turns"] == 4 and g.quantities["tts_chars_eleven"] == 120
    low, high = g.totals()
    expected_low = 12.5 * 0.016 / 60 + 1000 * 0.30 / M + 3000 * 0.03 / M + 200 * 2.50 / M + 120 * 0.5 * 6 / 30000
    assert low == pytest.approx(expected_low)
    assert high == pytest.approx(expected_low + 120 * 0.5 * 6 / 30000)
    assert [ln.item for ln in g.unpriced()] == ["gen[x]_input_tokens", "gen[x]_cached_tokens", "gen[x]_output_tokens"]

    utc_groups = usage_report.aggregate(events, tz=UTC)
    assert (7, "2026-09") in utc_groups, "UTC 기준이면 9/30 15:30 은 9월"
    only = usage_report.aggregate(events, tz=KST, month="2026-10", user=7)
    assert list(only) == [(7, "2026-10")]


def test_cli_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        usage_report.main(["--help"])
    assert exc.value.code == 0


def test_cli_log_source_renders_all_sections_and_csv(tmp_path, capsys) -> None:
    log = _write_log(tmp_path)
    out_dir = tmp_path / "csv"
    assert usage_report.main(["--source", "log", "--log", str(log), "--month", "2026-09", "--csv", str(out_dir)]) == 0
    out = capsys.readouterr().out
    assert "user=77" in out and "user=미귀속" in out and "user=7" not in out.replace("user=77", "")
    assert "단가 미설정" in out and "gen[x]_input_tokens" in out
    assert f"ElevenLabs {ELEVEN_PLAN.name}" in out and "⑤ 단가 출처" in out
    assert "tts_chars[eleven] PAYG 참고" in out and "합계 미포함" in out
    files = sorted(p.name for p in out_dir.iterdir())
    assert files == ["usage_costs.csv", "usage_quantities.csv", "usage_unpriced.csv"]
    assert "gen[x]_input_tokens" in (out_dir / "usage_unpriced.csv").read_text(encoding="utf-8")


def test_cli_without_inputs_fails_cleanly(monkeypatch, capsys) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert usage_report.main(["--source", "log"]) == 2
    assert usage_report.main(["--source", "db"]) == 2


def test_cli_reads_from_sqlite_db(tmp_path, capsys) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'usage.db'}"

    async def _seed() -> None:
        engine = create_async_engine(url)
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            sessions = async_sessionmaker(engine, expire_on_commit=False)
            assert await record_event("session_usage", _SESSION, session_factory=sessions)
            assert await record_event("llm_usage", _LLM, session_factory=sessions)
        finally:
            await engine.dispose()

    asyncio.run(_seed())
    assert usage_report.main(["--source", "db", "--database-url", url, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    by_user = {row["user"]: row for row in data}
    assert by_user[77]["sessions"] == 1 and by_user[77]["quantities"]["stt_sec"] == 12.5
    assert by_user[7]["quantities"]["gen_input_tokens"] == 700
    assert by_user[77]["usd_low"] < by_user[77]["usd_high"], "EL 크레딧 범위"


def test_month_bounds_follow_the_report_timezone() -> None:
    lo, hi = usage_report.month_bounds_utc("2026-09", KST)
    assert lo.isoformat() == "2026-08-31T15:00:00+00:00" and hi.isoformat() == "2026-09-30T15:00:00+00:00"
    lo_dec, hi_dec = usage_report.month_bounds_utc("2026-12", UTC)
    assert (lo_dec.year, lo_dec.month, hi_dec.year, hi_dec.month) == (2026, 12, 2027, 1)


def test_load_from_db_filters_month_and_user_in_the_query(tmp_path) -> None:
    from datetime import datetime

    from app.db.models import UsageEventORM

    url = f"sqlite+aiosqlite:///{tmp_path / 'usage.db'}"

    def _row(user_id: int, created_at: datetime) -> UsageEventORM:
        return UsageEventORM(kind="session", session_id=f"s-{user_id}-{created_at.month}", user_id=user_id,
                             provider=None, model=None, purpose=None,
                             payload={"user_id": user_id, "turns": 1}, created_at=created_at)

    async def _seed() -> None:
        engine = create_async_engine(url)
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            sessions = async_sessionmaker(engine, expire_on_commit=False)
            async with sessions() as db:
                db.add_all([
                    _row(77, datetime(2026, 8, 31, 15, 30, tzinfo=UTC)),  # KST 9/1 00:30 → 9월
                    _row(77, datetime(2026, 7, 15, 0, 0, tzinfo=UTC)),    # 7월
                    _row(8, datetime(2026, 9, 10, 0, 0, tzinfo=UTC)),     # 9월, 다른 사용자
                ])
                await db.commit()
        finally:
            await engine.dispose()

    asyncio.run(_seed())
    sept = usage_report.load_from_db(url, month="2026-09", tz=KST)
    assert sorted(e.payload["user_id"] for e in sept) == [8, 77], "7월 행은 DB 에서 걸러진다"
    assert [e.ts_utc.month for e in sept] == [8, 9], "UTC 8/31 15:30 은 KST 9월"
    only_77 = usage_report.load_from_db(url, month="2026-09", tz=KST, user=77)
    assert [e.payload["user_id"] for e in only_77] == [77]
    assert len(usage_report.load_from_db(url)) == 3, "필터 없으면 전체"
