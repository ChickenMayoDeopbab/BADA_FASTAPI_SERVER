import logging
import random
from datetime import timedelta

from app.core.timeutil import now_utc
from app.services.biosignal.trend import summarize_trend, trend_of
from app.services.biosignal_service import save_call_timeline
from app.services.session import parse_ai_personality
from tests.unit.test_avti_save import _factory
from tests.unit.test_biosignal_api import _env, _seed
from tests.unit.test_biosignal_pipeline import _closing_call


def _metrics(call: float | None = 10.0, anticipation: float | None = 5.0, recovery: float | None = 80.0,
             eda: float | None = 20.0) -> dict:
    return {
        "cardiac": {"hr": {
            "call_delta": {"status": "MEASURED", "value": call},
            "anticipation_delta": {"status": "MEASURED", "value": anticipation},
            "peak": {"status": "MEASURED", "delta": None if call is None else call + 5},
            "listen_minus_speak": {"status": "MEASURED", "value": -2.0},
            "recovery_pct": {"status": "MEASURED", "value": recovery},
        }},
        "eda": {
            "change_vs_baseline_pct": {"call": {"status": "MEASURED", "value": eda}},
            "rises": {"call": {"per_min": 1.5}},
        },
    }


def _row(day: int, *, scenario: int = 1, difficulty: str = "easy", personality: str = "KIND",
         session_type: str = "SCENARIO", **metrics) -> dict:
    return {
        "session_id": f"s{day}",
        "at": f"2026-11-{day:02d}T10:00:00+00:00",
        "scenario_id": scenario,
        "difficulty": difficulty,
        "ai_personality": personality,
        "session_type": session_type,
        "metrics": _metrics(**metrics),
    }


def _group(result: dict, scenario: int, difficulty: str, personality: str) -> dict:
    return next(
        g for g in result["groups"]
        if (g["scenario_id"], g["difficulty"], g["ai_personality"]) == (scenario, difficulty, personality)
    )


def test_trends_compare_only_the_same_scenario_difficulty_and_attitude() -> None:
    rows = [
        _row(1, call=12.0),
        _row(2, call=10.0),
        _row(3, difficulty="hard", call=15.0),
        _row(4, personality="RUDE", call=14.0),
        _row(5, call=8.0),
    ]

    result = summarize_trend(rows)

    same = _group(result, 1, "easy", "KIND")
    assert [s["session_id"] for s in same["sessions"]] == ["s1", "s2", "s5"]
    assert same["trend"]["hr_call_delta"] == {
        "status": "MEASURED", "sessions": 3, "first": 12.0, "last": 8.0, "change": -4.0, "slope_per_session": -2.0,
    }
    assert _group(result, 1, "hard", "KIND")["trend"]["hr_call_delta"] == {"status": "INSUFFICIENT_DATA", "sessions": 1}
    assert _group(result, 1, "easy", "RUDE")["trend"]["hr_call_delta"]["status"] == "INSUFFICIENT_DATA"
    assert len(result["groups"]) == 3


def test_warmups_are_left_out_and_input_order_does_not_matter() -> None:
    rows = [_row(day, call=float(20 - day)) for day in range(1, 6)] + [_row(9, session_type="WARMUP", call=99.0)]
    shuffled = rows[:]
    random.Random(4).shuffle(shuffled)

    result = summarize_trend(shuffled)

    assert result == summarize_trend(rows)
    assert result["sessions"] == 5
    assert [s["session_id"] for s in result["groups"][0]["sessions"]] == ["s1", "s2", "s3", "s4", "s5"]
    assert all(s["session_id"] != "s9" for s in result["first_encounters"]["sessions"])


def test_first_encounters_take_each_scenarios_first_session_regardless_of_condition() -> None:
    rows = [
        _row(1, scenario=1, call=18.0),
        _row(2, scenario=1, call=11.0),
        _row(3, scenario=2, difficulty="hard", call=16.0),
        _row(4, scenario=2, call=9.0),
        _row(5, scenario=3, personality="RUDE", call=14.0),
    ]

    first = summarize_trend(rows)["first_encounters"]

    assert [s["session_id"] for s in first["sessions"]] == ["s1", "s3", "s5"]
    assert first["trend"]["hr_call_delta"]["change"] == -4.0


def test_anticipation_and_recovery_use_every_session() -> None:
    rows = [
        _row(1, scenario=1, anticipation=9.0, recovery=50.0),
        _row(2, scenario=2, difficulty="hard", anticipation=7.0, recovery=60.0),
        _row(3, scenario=3, personality="RUDE", anticipation=5.0, recovery=70.0),
    ]

    result = summarize_trend(rows)

    assert (result["anticipation"]["slope_per_session"], result["anticipation"]["change"]) == (-2.0, -4.0)
    assert (result["recovery"]["slope_per_session"], result["recovery"]["last"]) == (10.0, 70.0)


def test_missing_values_shrink_the_series_and_can_make_it_insufficient() -> None:
    rows = [_row(1, eda=None), _row(2, eda=30.0), _row(3, eda=None), _row(4, eda=20.0), _row(5, call=None, eda=None)]

    trend = summarize_trend(rows)["groups"][0]["trend"]

    assert trend["eda_call_change_pct"] == {"status": "INSUFFICIENT_DATA", "sessions": 2}
    assert trend["hr_call_delta"]["sessions"] == 4
    assert trend["hr_peak_delta"]["sessions"] == 4


def test_no_sessions_gives_empty_insufficient_trends() -> None:
    result = summarize_trend([])

    assert (result["sessions"], result["groups"], result["first_encounters"]["sessions"]) == (0, [], [])
    assert result["anticipation"] == {"status": "INSUFFICIENT_DATA", "sessions": 0}
    assert trend_of([None, None, 1.0]) == {"status": "INSUFFICIENT_DATA", "sessions": 1}


# --- 상대 태도 저장 -----------------------------------------------------------


def test_attitude_parsing_matches_the_prompt_rules(caplog) -> None:
    caplog.set_level(logging.WARNING)

    assert parse_ai_personality({"aiPersonality": "kind"}).value == "KIND"
    assert parse_ai_personality({}).value == "NORMAL"
    assert parse_ai_personality({"aiPersonality": "weird"}).value == "NORMAL"
    assert any("알 수 없는 aiPersonality" in r.getMessage() for r in caplog.records)


async def test_the_timeline_row_keeps_the_calls_attitude(monkeypatch) -> None:
    session = {"type": "SCENARIO", "userId": 7, "scenarioId": 3, "aiPersonality": "rude"}
    p, _, calls = _closing_call(monkeypatch, session=session)

    await p._teardown()

    assert calls[0]["ai_personality"] == "RUDE"


async def test_save_call_timeline_writes_the_attitude() -> None:
    store: dict = {}

    await save_call_timeline(
        session_id="sess-1", user_id=7, scenario_id=3, session_type="SCENARIO", difficulty="easy",
        timeline={}, ai_personality="TOUGH", session_factory=_factory(store),
    )

    assert "TOUGH" in set(map(str, store["params"].values()))


# --- 조회 API ----------------------------------------------------------------


async def test_trend_route_is_disabled_with_the_flag() -> None:
    async with _env(enabled=False) as env:
        assert (await env.client.get("/api/v1/biosignals/trend")).status_code == 404


async def test_trend_route_summarizes_only_my_analyzed_calls() -> None:
    async with _env() as env:
        base = now_utc() - timedelta(days=10)
        for i, call in enumerate((12.0, 10.0, 8.0)):
            await _seed(
                env, f"mine-{i}", status="ANALYZED", metrics=_metrics(call=call),
                created_at=base + timedelta(days=i), ai_personality="KIND", difficulty="easy",
            )
        await _seed(env, "mine-waiting")
        await _seed(env, "theirs", 9, status="ANALYZED", metrics=_metrics(call=40.0), created_at=base)

        response = await env.client.get("/api/v1/biosignals/trend")

        assert response.status_code == 200
        body = response.json()
        assert (body["version"], body["sessions"]) == ("BIOSIGNAL_TREND_V1", 3)
        group = body["groups"][0]
        assert (group["scenario_id"], group["difficulty"], group["ai_personality"]) == (3, "easy", "KIND")
        assert [s["session_id"] for s in group["sessions"]] == ["mine-0", "mine-1", "mine-2"]
        assert group["trend"]["hr_call_delta"]["slope_per_session"] == -2.0
