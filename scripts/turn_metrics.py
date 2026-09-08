import argparse
import csv
import json
import math
import re
import sys

_KV = re.compile(r"(\w+)=(\S+)")
_NUM = ("stt_ms", "llm_ttft_ms", "llm_total_ms", "tts_ttfb_ms", "tts_total_ms", "response_ms",
        "turn_total_ms", "pcm_chunks", "pcm_bytes", "audio_ms", "odd_chunks", "send_wall_ms",
        "arrival_rtf", "max_gap_ms", "gap0_count", "gap300_count", "engine_chunks", "step",
        "chunks", "gaps80", "total_gap_ms", "played_ms", "dropped_chunks", "first_play_ms", "turn")
_BOOL = ("error", "watchdog", "fallback", "tts_failed")


def _coerce(k, v):
    if v in ("None", "null", None):
        return None
    if k in _BOOL:
        return v in (True, "True", "true")
    if k in _NUM:
        try:
            return float(v) if "." in str(v) or k == "arrival_rtf" else int(float(v))
        except ValueError:
            return None
    return v


def parse(paths):
    rows = []
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if "metric" not in line:
                    continue
                if line.startswith("{"):
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "metric" not in d:
                        continue
                    rows.append({k: _coerce(k, v) for k, v in d.items()})
                    continue
                m = re.search(r"metric=(\w+)\s*(.*)$", line)
                if not m:
                    continue
                d = {"metric": m.group(1)}
                for k, v in _KV.findall(m.group(2)):
                    d[k] = _coerce(k, v)
                rows.append(d)
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    i = (len(xs) - 1) * q
    lo, hi = math.floor(i), math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def summarize(turns, label):
    n = len(turns)
    bad_gap = sum(1 for t in turns if (t.get("gap0_count") or 0) >= 1)
    bad_srv = sum(1 for t in turns if t.get("tts_failed") or t.get("watchdog") or t.get("fallback"))
    def _bad(t):
        return (t.get("gap0_count") or 0) >= 1 or t.get("tts_failed") or t.get("watchdog") or t.get("fallback")

    bad_any = sum(1 for t in turns if _bad(t))
    bad300 = sum(1 for t in turns if (t.get("gap300_count") or 0) >= 1)
    odd = sum(1 for t in turns if (t.get("odd_chunks") or 0) >= 1)
    print(f"\n## {label}  (n={n} 턴)")
    for name, k in (("L1 끊김 턴(gap0≥1)", bad_gap), ("서버 확정 불량(tts_failed/watchdog/fallback)", bad_srv),
                    ("L1 불량 턴(합집합)", bad_any), ("프리버퍼 300ms 가정 끊김 턴(gap300≥1)", bad300),
                    ("홀수 바이트 청크 있는 턴", odd)):
        p, lo, hi = wilson(k, n)
        print(f"- {name}: {k}/{n} = {p*100:.1f}%  (Wilson 95% CI {lo*100:.1f}~{hi*100:.1f}%)")
    print("| 지표 | p50 | p90 | max |\n|---|---|---|---|")
    for f in ("tts_ttfb_ms", "response_ms", "audio_ms", "pcm_chunks", "engine_chunks", "send_wall_ms", "arrival_rtf",
              "max_gap_ms", "gap0_count", "gap300_count", "odd_chunks"):
        xs = [t.get(f) for t in turns if t.get(f) is not None]
        if not xs:
            continue
        fmt = (lambda v: f"{v:.3f}") if f == "arrival_rtf" else (lambda v: f"{v:.0f}")
        print(f"| {f} | {fmt(pct(xs, .5))} | {fmt(pct(xs, .9))} | {fmt(max(xs))} |")
    engines = {}
    for t in turns:
        engines.setdefault(t.get("tts_engine"), []).append(t)
    if len(engines) > 1:
        for e, ts in engines.items():
            summarize(ts, f"{label} / tts_engine={e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--label", default="all")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--session", default=None, help="session_id 접두사 필터")
    a = ap.parse_args()
    rows = parse(a.logs)
    turns = [r for r in rows if r.get("metric") == "voice_turn"]
    if a.session:
        turns = [t for t in turns if str(t.get("session_id", "")).startswith(a.session)]
    others = {}
    for r in rows:
        if r.get("metric") in ("fallback_audio", "client_playback", "realtime_tts_switch", "realtime_tts_engine"):
            others.setdefault(r["metric"], []).append(r)
    print(f"voice_turn {len(turns)}행 · " + " · ".join(f"{k} {len(v)}행" for k, v in others.items()))
    if not turns:
        sys.exit(1)
    summarize(turns, a.label)
    if others.get("realtime_tts_switch"):
        reasons = {}
        for r in others["realtime_tts_switch"]:
            reasons[r.get("reason")] = reasons.get(r.get("reason"), 0) + 1
        print("\nrealtime_tts_switch reason:", reasons)
    if others.get("realtime_tts_engine"):
        skips = {}
        for r in others["realtime_tts_engine"]:
            key = (r.get("engine"), r.get("skip_reason"))
            skips[key] = skips.get(key, 0) + 1
        print("realtime_tts_engine (engine, skip_reason):", skips)
    if a.csv:
        keys = sorted({k for t in turns for k in t})
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(turns)
        print(f"csv → {a.csv}")


if __name__ == "__main__":
    main()
