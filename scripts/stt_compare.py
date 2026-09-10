"""STT 엔진 비교 집계 (계획 0052 F76).

ws_bench `--json --json-out` 결과 여러 개를 (engine, tail_silence_ms) 조건별로 묶어
FINAL 성공률(Wilson 95% CI)·stt_final_ms p50/p95·오인식 건수(정규화 후 불일치)·평균 CER 을 표로 낸다.

    .venv/bin/python scripts/stt_compare.py --manifest audio/manifest.json runs/*.json
"""
import argparse
import json
import math
import re
from pathlib import Path

_STRIP = re.compile(r"[\s.,?!·…\-~:;\"'()\[\]]+")


def normalize(text: str) -> str:
    """공백·구두점 제거. 띄어쓰기/문장부호 차이는 오인식으로 치지 않는다."""
    return _STRIP.sub("", text or "")


def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    """문자 오류율 = 편집거리 / 정답 길이."""
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def load_runs(paths: list[str]) -> dict[tuple[str, int], list[dict]]:
    runs: dict[tuple[str, int], list[dict]] = {}
    for path in paths:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        key = (str(d.get("engine")), int(d.get("tail_silence_ms", 0)))
        runs.setdefault(key, []).extend(d.get("turns", []))
    return runs


_NOT_ATTEMPTED = {"end", "error", "CLOSED"}


def summarize_condition(turns: list[dict], manifest: dict[str, str]) -> dict:
    """세션이 이미 끝난 뒤의 턴(end/error/CLOSED)은 STT 시도가 아니므로 제외한다. TIMEOUT 은 시도(FINAL 없음)."""
    attempted = [t for t in turns if t.get("terminal") not in _NOT_ATTEMPTED]
    finals = [t for t in attempted if t.get("stt_final_ms") is not None]
    n, k = len(attempted), len(finals)
    rate, lo, hi = wilson(k, n)
    latencies = [float(t["stt_final_ms"]) for t in finals]
    cers: list[float] = []
    mismatches = 0
    for t in finals:
        ref = manifest.get(t.get("audio", ""))
        if ref is None:
            continue
        r, h = normalize(ref), normalize(t.get("transcript") or "")
        c = cer(r, h)
        cers.append(c)
        if r != h:
            mismatches += 1
    return {
        "n": n,
        "excluded": len(turns) - n,
        "finals": k,
        "final_rate": rate,
        "final_rate_lo": lo,
        "final_rate_hi": hi,
        "stt_final_p50": percentile(latencies, 50),
        "stt_final_p95": percentile(latencies, 95),
        "mismatches": mismatches,
        "mean_cer": (sum(cers) / len(cers)) if cers else None,
    }


def _fmt_ms(v: float | None) -> str:
    return f"{v:.0f}" if v is not None else "—"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="ws_bench --json-out 파일들")
    ap.add_argument("--manifest", required=True, help="오디오 파일명 → 정답 텍스트 JSON")
    ap.add_argument("--details", action="store_true", help="오인식 턴을 한 줄씩 출력")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    runs = load_runs(args.runs)
    print(
        f"{'engine':<12} {'silence':>7} {'n':>4} {'FINAL%':>7} {'95% CI':>15} "
        f"{'p50':>6} {'p95':>6} {'mis':>4} {'CER':>6}"
    )
    for (engine, silence), turns in sorted(runs.items()):
        s = summarize_condition(turns, manifest)
        ci = f"[{s['final_rate_lo'] * 100:.0f}, {s['final_rate_hi'] * 100:.0f}]"
        cer_s = f"{s['mean_cer'] * 100:.1f}%" if s["mean_cer"] is not None else "—"
        print(
            f"{engine:<12} {silence:>7} {s['n']:>4} {s['final_rate'] * 100:>6.0f}% {ci:>15} "
            f"{_fmt_ms(s['stt_final_p50']):>6} {_fmt_ms(s['stt_final_p95']):>6} {s['mismatches']:>4} {cer_s:>6}"
        )
        if args.details:
            for t in turns:
                if t.get("terminal") in _NOT_ATTEMPTED:
                    continue
                ref = manifest.get(t.get("audio", ""), "")
                hyp = t.get("transcript")
                if hyp is None or normalize(ref) != normalize(hyp):
                    print(f"    [{t.get('audio')}] {t.get('terminal') or ''} 정답={ref!r} 전사={hyp!r}")


if __name__ == "__main__":
    main()
