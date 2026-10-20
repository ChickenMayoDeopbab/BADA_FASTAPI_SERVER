"""운영 텍스트 로그로 chirp denoiser 켜기 전후를 비교한다 (네트워크 없음, 맥에서).

  .venv/bin/python scripts/stt_noise/prod_compare.py stt-before.log stt-after.log [--since 2026-10-21] [--until ...]

세션을 metric=stt_config 로 묶는다: off(denoise=False) · on(denoise=True) · none(stt_config 없음 = 스위치 이전)
· gemini(engine=gemini_live) · mixed(한 세션에 값이 둘).
묶음마다 세션·voice_turn 수, final_dropped 사유별 건수·100턴당·하나라도 있었던 세션 비율(Wilson 95%),
state 중 글자>0 · 턴 FINAL 뒤 3초 안(끊긴 FINAL 의 나머지가 AI 차례에 온 신호),
session_usage reason 분포와 ERROR 세션 비율, voice_turn 지연 p50/p90.
세션 시각 = 그 세션 첫 줄의 로그 시각. --since(포함)·--until(제외)은 그 시각으로 세션을 고른다(로그 시각 그대로,
시간대 없이 — Z·+09:00 이 붙으면 거절).
session_id 가 없는 일반 줄(STT 무요청 ABORTED·STT 스트리밍 실패·STT 스트림 재오픈 등)은 그 줄 시각으로 걸러
전체 수만 낸다. 시각 있는 같은 줄은 파일마다 센 개수의 최댓값만(백업 로그와 docker logs 가 겹쳐도 한 번).
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.stt_compare import percentile, wilson  # noqa: E402

_TS = re.compile(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})")
_DOCKER_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\S+Z\s+")
_METRIC = re.compile(r"metric=(\w+)\s*(.*)$")
_KV = re.compile(r"(\w+)=(\S+)")
SESSION_METRICS = ("stt_config", "voice_turn", "final_dropped", "session_usage")
REASONS = ("empty", "state", "time_up", "closing")
LATENCY = ("stt_ms", "response_ms", "turn_total_ms")
SPLIT_TAIL_MS = 3000
GROUP_ORDER = ("off", "on", "none", "gemini", "mixed")
GENERAL = (
    ("aborted_no_requests", "STT 무요청 ABORTED"),
    ("stream_failed", "STT 스트리밍 실패"),
    ("reopen_after_drop", "STT 스트림 끊김"),
    ("reopen_backoff", "STT 스트림 재오픈("),
    ("reopen_gave_up", "복구 포기"),
    ("hard_recycle", "STT 하드 재활용"),
)
_BOOL = {"denoise"}
_NUM = {"chars", "since_final_ms", "turns", "snr_threshold", *LATENCY}


def _coerce(k: str, v):
    if v in ("None", "null", None):
        return None
    if k in _BOOL:
        return v in (True, "True", "true")
    if k in _NUM:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    return str(v)


def _ts(line: str) -> datetime | None:
    m = _TS.search(line)
    return datetime.strptime(f"{m[1]} {m[2]}", "%Y-%m-%d %H:%M:%S") if m else None


def parse_line(line: str) -> tuple[datetime | None, dict | None, str | None]:
    """(시각, 메트릭 dict, 일반 줄 종류). 관계없는 줄은 (시각, None, None)"""
    ts = _ts(line)
    body = _DOCKER_TS.sub("", line, count=1)
    if body.startswith("{"):
        try:
            d = json.loads(body)
        except json.JSONDecodeError:
            return ts, None, None
        if "metric" in d:
            return ts, {k: _coerce(k, v) for k, v in d.items()}, None
        body = str(d.get("message", ""))
    m = _METRIC.search(body)
    if m:
        row = {"metric": m[1]}
        row.update({k: _coerce(k, v) for k, v in _KV.findall(m[2])})
        return ts, row, None
    kind = next((k for k, pattern in GENERAL if pattern in body), None)
    return ts, None, kind


def read_logs(paths: list[str]) -> dict:
    """{rows: [(시각, 메트릭)], general: [(시각, 종류)], lines, dup}.
    시각 있는 같은 줄은 파일마다 센 횟수의 최댓값만 — 겹친 파일은 한 번, 한 파일 안의 같은 줄은 그대로"""
    keep: dict[str, tuple[int, tuple]] = {}
    rows, general = [], []
    lines = total = 0
    for path in paths:
        counts: dict[str, int] = {}
        parsed: dict[str, tuple] = {}
        with open(path, encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                lines += 1
                item = parsed.get(line) or parse_line(line)
                ts, row, kind = item
                if row is None and kind is None:
                    continue
                if ts is None:
                    if row is not None:
                        rows.append((None, row))
                    else:
                        general.append((None, kind))
                    continue
                parsed[line] = item
                counts[line] = counts.get(line, 0) + 1
        for line, k in counts.items():
            total += k
            if k > keep.get(line, (0,))[0]:
                keep[line] = (k, parsed[line])
    for k, (ts, row, kind) in keep.values():
        for _ in range(k):
            if row is not None:
                rows.append((ts, row))
            else:
                general.append((ts, kind))
    return {"rows": rows, "general": general, "lines": lines, "dup": total - sum(k for k, _ in keep.values())}


def _in_window(ts: datetime | None, since: datetime | None, until: datetime | None) -> bool:
    if since is None and until is None:
        return True
    if ts is None:
        return False
    return (since is None or ts >= since) and (until is None or ts < until)


def sessions_of(rows: list[tuple[datetime | None, dict]]) -> dict[str, dict]:
    """session_id → {first, configs, turns, drops, usage}"""
    out: dict[str, dict] = {}
    for ts, row in rows:
        sid = row.get("session_id")
        if row["metric"] not in SESSION_METRICS or sid is None:
            continue
        s = out.setdefault(sid, {"first": ts, "configs": [], "turns": [], "drops": [], "usage": []})
        if ts is not None and (s["first"] is None or ts < s["first"]):
            s["first"] = ts
        key = {"stt_config": "configs", "voice_turn": "turns", "final_dropped": "drops", "session_usage": "usage"}
        s[key[row["metric"]]].append(row)
    return out


def group_of(session: dict) -> str:
    configs = session["configs"]
    if not configs:
        return "none"
    engines = {c.get("engine") for c in configs}
    if engines == {"gemini_live"}:
        return "gemini"
    values = {bool(c.get("denoise")) for c in configs}
    if len(engines) > 1 or len(values) > 1:
        return "mixed"
    return "on" if values.pop() else "off"


def _rate(k: int, n: int) -> dict:
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "p": p, "lo": lo, "hi": hi}


def summarize(sessions: list[dict]) -> dict:
    n = len(sessions)
    turns = [t for s in sessions for t in s["turns"]]
    n_turns = len(turns)
    drops: dict[str, dict] = {}

    def add(name: str, pick) -> None:
        per = [sum(1 for d in s["drops"] if pick(d)) for s in sessions]
        count = sum(per)
        drops[name] = {"count": count, "per100": 100 * count / n_turns if n_turns else None,
                       "sessions": _rate(sum(1 for k in per if k), n)}

    for reason in REASONS:
        add(reason, lambda d, r=reason: d.get("reason") == r)
    add("state_text", lambda d: d.get("reason") == "state" and (d.get("chars") or 0) > 0)
    add("state_3s", lambda d: d.get("reason") == "state" and (d.get("chars") or 0) > 0
        and d.get("since_final_ms") is not None and d["since_final_ms"] <= SPLIT_TAIL_MS)
    state_drops = [d for s in sessions for d in s["drops"] if d.get("reason") == "state"]
    by_state: dict[str, int] = {}
    for d in state_drops:
        by_state[str(d.get("state"))] = by_state.get(str(d.get("state")), 0) + 1
    ends: dict[str, int] = {}
    for s in sessions:
        reason = s["usage"][-1].get("reason") if s["usage"] else "없음"
        ends[str(reason)] = ends.get(str(reason), 0) + 1
    latency = {}
    for key in LATENCY:
        xs = [t[key] for t in turns if t.get(key) is not None]
        latency[key] = {"n": len(xs), "p50": percentile(xs, 50), "p90": percentile(xs, 90)}
    return {
        "sessions": n,
        "turns": n_turns,
        "drops": drops,
        "state_by_state": by_state,
        "state_since_final_p50": percentile([d["since_final_ms"] for d in state_drops
                                            if d.get("since_final_ms") is not None], 50),
        "end_reasons": ends,
        "error_sessions": _rate(ends.get("ERROR", 0), n),
        "latency": latency,
    }


def compare(data: dict, *, since: datetime | None = None, until: datetime | None = None) -> dict:
    sessions = {sid: s for sid, s in sessions_of(data["rows"]).items() if _in_window(s["first"], since, until)}
    groups: dict[str, list[dict]] = {}
    for s in sessions.values():
        groups.setdefault(group_of(s), []).append(s)
    general = {k: 0 for k, _ in GENERAL}
    for ts, kind in data["general"]:
        if _in_window(ts, since, until):
            general[kind] += 1
    stamps = [s["first"] for s in sessions.values() if s["first"] is not None]
    return {
        "since": None if since is None else since.isoformat(sep=" "),
        "until": None if until is None else until.isoformat(sep=" "),
        "first_session": None if not stamps else min(stamps).isoformat(sep=" "),
        "last_session": None if not stamps else max(stamps).isoformat(sep=" "),
        "lines": data["lines"],
        "dup": data["dup"],
        "groups": {g: summarize(groups[g]) for g in GROUP_ORDER if g in groups},
        "general": general,
    }


def _pct(r: dict) -> str:
    return f"{r['k']}/{r['n']} = {r['p'] * 100:.1f}% ({r['lo'] * 100:.0f}~{r['hi'] * 100:.0f}%)"


def _num(v: float | None, fmt: str = ".0f") -> str:
    return "—" if v is None else format(v, fmt)


LABELS = {"off": "denoise=False", "on": "denoise=True", "none": "stt_config 없음(스위치 이전)",
          "gemini": "engine=gemini_live", "mixed": "한 세션에 값 둘"}
DROP_LABELS = {"empty": "empty(빈 FINAL)", "state": "state(AI 차례에 온 FINAL)", "state_text": "state 중 글자>0",
               "state_3s": f"state 중 턴 FINAL 뒤 {SPLIT_TAIL_MS // 1000}초 안", "time_up": "time_up",
               "closing": "closing"}


def print_report(report: dict) -> None:
    print(f"읽은 줄 {report['lines']} · 겹친 줄 {report['dup']} 건너뜀 · 세션 시각 {report['first_session']} ~ "
          f"{report['last_session']} · 고른 범위 {report['since'] or '처음'} ~ {report['until'] or '끝'}")
    if not report["groups"]:
        print("세션이 없다 — 로그에 metric=stt_config|voice_turn|final_dropped|session_usage 줄이 있는지 볼 것")
    for name, s in report["groups"].items():
        per = s["turns"] / s["sessions"] if s["sessions"] else 0
        print(f"\n## {name} — {LABELS[name]}  세션 {s['sessions']} · voice_turn {s['turns']} (세션당 {per:.1f})")
        print("| final_dropped | 건수 | 100턴당 | 하나라도 있던 세션 (Wilson 95%) |\n|---|---|---|---|")
        for key, label in DROP_LABELS.items():
            d = s["drops"][key]
            print(f"| {label} | {d['count']} | {_num(d['per100'], '.1f')} | {_pct(d['sessions'])} |")
        if s["state_by_state"]:
            detail = " · ".join(f"{k} {v}" for k, v in sorted(s["state_by_state"].items()))
            print(f"state 세부: {detail} · since_final_ms p50 {_num(s['state_since_final_p50'])}")
        print("session_usage reason: " + " · ".join(f"{k} {v}" for k, v in sorted(s["end_reasons"].items())))
        print(f"ERROR 종료 세션: {_pct(s['error_sessions'])}")
        print("지연 p50/p90 ms: " + " · ".join(
            f"{k} {_num(v['p50'])}/{_num(v['p90'])} (n={v['n']})" for k, v in s["latency"].items()))
    if len(report["groups"]) > 1:
        print("\n| 묶음 | 세션 | 턴 | " + " | ".join(f"{k}/100턴" for k in DROP_LABELS) + " | ERROR 세션 |")
        print("|---" * (len(DROP_LABELS) + 4) + "|")
        for name, s in report["groups"].items():
            cells = " | ".join(_num(s["drops"][k]["per100"], ".1f") for k in DROP_LABELS)
            err = s["error_sessions"]
            print(f"| {name} | {s['sessions']} | {s['turns']} | {cells} | {err['k']}/{err['n']} |")
    general = " · ".join(f"{k} {v}" for k, v in report["general"].items())
    print(f"\n세션에 못 붙이는 STT 줄(고른 범위 전체): {general}")


def _when(text: str) -> datetime:
    """로그 시각과 견줄 시각 — 로그엔 시간대가 없으니 Z·+09:00 이 붙은 값은 거절한다"""
    try:
        when = datetime.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"ISO 시각이 아니다: {text!r}") from None
    if when.tzinfo is not None:
        raise argparse.ArgumentTypeError(f"시간대 없이 로그에 찍힌 시각 그대로 쓴다(Z·+09:00 빼기): {text!r}")
    return when


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="운영 로그 파일(docker logs 를 grep 한 것, 여러 개 가능)")
    ap.add_argument("--since", type=_when, default=None,
                    help="이 시각부터(포함), 로그 시각 그대로(시간대 없이) 예 2026-10-21 또는 '2026-10-21 09:00'")
    ap.add_argument("--until", type=_when, default=None, help="이 시각 전까지(제외), 형식은 --since 와 같다")
    ap.add_argument("--json-out", default=None, help="요약을 JSON 으로 저장")
    a = ap.parse_args()

    report = compare(read_logs(a.logs), since=a.since, until=a.until)
    print_report(report)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"json → {a.json_out}")


if __name__ == "__main__":
    main()
