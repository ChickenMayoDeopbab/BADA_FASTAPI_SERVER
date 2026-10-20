"""run_chirp 결과를 (잡음, SNR, denoiser, 청크)별로 채점한다. ws_bench --json-out 결과(로컬 서버 경로)도 받는다.

  .venv/bin/python scripts/stt_noise/score.py .harness/stt-noise-2026-10/runs/*.json
  .venv/bin/python scripts/stt_noise/score.py --manifest .harness/stt-compare-2026-09/audio/manifest.json ws-*.json

턴 = 첫 비어 있지 않은 FINAL(운영과 같다). EOS 뒤에 온 FINAL(after_eos)은 끝점이 아니라 반쯤 닫기로 나온 것이라 뺀다.
스트림 에러가 난 클립(stopped=error 이거나 error 이벤트)은 n 에서 빼고 err 로 따로 센다.
비어 있지 않은 FINAL 은 덮은 오디오의 끝(stream_open_s + end_offset_s, 없으면 도착 시각)으로 나눈다:
noise = 말소리 시작 전에 끝났거나 말소리가 끝난 뒤 열린 스트림 · fragment = 말소리 끝 − 0.3초 전에 끝남 ·
complete = 그 뒤.
발화 클립: FINAL 성공률(Wilson 95%, 표본 수 = 서로 다른 오디오 파일 수) · 턴 CER(정규화 후, 턴 없는 클립 제외 /
턴 없음=1 포함) · 정답과 다른 턴 수 · 말소리 끝→턴 FINAL p50/p95(complete 턴만) · noiseT(턴이 잡음) ·
early(턴이 발화 조각 — 운영에선 나머지가 버려진다) · split(발화가 비어 있지 않은 FINAL 2개 이상으로 쪼개짐) ·
클립당 빈 FINAL·SPEECH_BEGIN.
잡음만 클립: 거짓 FINAL(비어 있지 않은 FINAL) 수·분당·하나라도 나온 클립 수, 빈 FINAL, SPEECH_BEGIN 분당.
ws_bench 결과는 (engine, 잡음, SNR, 꼬리, 청크, 앞 잡음)별로 scripts/stt_compare.py 의 summarize_condition 그대로.
noise 가 없는 ws_bench 결과(원본 파일 그대로)는 raw, 잡음 모드의 꼬리(tail_silence_ms=null)는 응답까지 잡음이다.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.stt_compare import cer, normalize, percentile, summarize_condition, wilson  # noqa: E402

NOISE_ORDER = {"raw": -1, "clean": 0, "white": 1, "pink": 2, "babble": 3}
END_MARGIN_S = 0.3


def finals(clip: dict) -> list[dict]:
    return [e for e in clip["events"] if e["type"] == "final" and not e.get("after_eos")]


def turn_of(clip: dict) -> dict | None:
    return next((e for e in finals(clip) if e["text"].strip()), None)


def failed(clip: dict) -> bool:
    return clip.get("stopped") == "error" or any(e["type"] == "error" for e in clip["events"])


def final_end_s(e: dict) -> float:
    """FINAL 이 덮은 오디오의 끝(클립 기준 초). end_offset 이 없으면 도착 시각"""
    off = e.get("end_offset_s")
    if off and off > 0 and e.get("stream_open_s") is not None:
        return e["stream_open_s"] + off
    return e["t"]


def kind_of(clip: dict, e: dict) -> str:
    """비어 있지 않은 FINAL 의 종류: noise · fragment · complete"""
    if (e.get("stream_open_s") or 0.0) >= clip["speech_end_s"]:
        return "noise"
    end = final_end_s(e)
    if end < clip["speech_start_s"]:
        return "noise"
    return "fragment" if end < clip["speech_end_s"] - END_MARGIN_S else "complete"


def _count(clip: dict, kind: str) -> int:
    return sum(1 for e in clip["events"] if e["type"] == kind and not e.get("after_eos"))


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def summarize_utterances(clips: list[dict]) -> dict:
    ok = [c for c in clips if not failed(c)]
    n = len(ok)
    n_audio = len({c["file"] for c in ok})
    turns = [(c, turn_of(c)) for c in ok]
    with_turn = [(c, t, kind_of(c, t)) for c, t in turns if t is not None]
    rate, lo, hi = wilson(len(with_turn) * n_audio / n, n_audio) if n else wilson(0, 0)
    cers: list[float] = []
    mismatches = 0
    for c, t, _ in with_turn:
        ref, hyp = normalize(c["text"]), normalize(t["text"])
        cers.append(cer(ref, hyp))
        mismatches += ref != hyp
    complete = [(c, t) for c, t, k in with_turn if k == "complete"]
    late = [(t["t"] - c["speech_end_s"]) * 1000 for c, t in complete]
    gaps = [(final_end_s(t) - c["speech_end_s"]) * 1000 for c, t in complete]
    noise_turns = sum(1 for *_, k in with_turn if k == "noise")
    early = sum(1 for *_, k in with_turn if k == "fragment")
    split = sum(
        1 for c in ok if sum(1 for e in finals(c) if e["text"].strip() and kind_of(c, e) != "noise") >= 2
    )
    return {
        "n": n,
        "errors": len(clips) - n,
        "n_audio": n_audio,
        "finals": len(with_turn),
        "final_rate": rate,
        "final_rate_lo": lo,
        "final_rate_hi": hi,
        "mean_cer": _mean(cers),
        "mean_cer_all": _mean(cers + [1.0] * (n - len(with_turn))),
        "mismatches": mismatches,
        "final_p50": percentile(late, 50),
        "final_p95": percentile(late, 95),
        "turn_end_gap_p50": percentile(gaps, 50),
        "noise_turns": noise_turns,
        "noise_turn_rate": noise_turns / n if n else 0.0,
        "early_turns": early,
        "early_rate": early / n if n else 0.0,
        "split": split,
        "split_rate": split / n if n else 0.0,
        "empty_finals_per_clip": _mean([sum(1 for e in finals(c) if not e["text"].strip()) for c in ok]),
        "speech_begin_per_clip": _mean([_count(c, "speech_begin") for c in ok]),
        "audio_sent_s": round(sum(c["audio_sent_s"] for c in clips), 3),
    }


def summarize_noise_only(clips: list[dict]) -> dict:
    ok = [c for c in clips if not failed(c)]
    minutes = sum(c["audio_sent_s"] for c in ok) / 60
    false_by_clip = [sum(1 for e in finals(c) if e["text"].strip()) for c in ok]
    false = sum(false_by_clip)
    begins = sum(_count(c, "speech_begin") for c in ok)
    return {
        "clips": len(ok),
        "errors": len(clips) - len(ok),
        "minutes": round(minutes, 3),
        "false_finals": false,
        "false_per_min": false / minutes if minutes else None,
        "clips_with_false": sum(1 for k in false_by_clip if k),
        "empty_finals": sum(1 for c in ok for e in finals(c) if not e["text"].strip()),
        "speech_begins": begins,
        "speech_begin_per_min": begins / minutes if minutes else None,
        "false_texts": [e["text"] for c in ok for e in finals(c) if e["text"].strip()],
    }


def cond_key(row: dict) -> tuple:
    """(잡음, SNR, denoiser, 청크 ms, denoiser snr_threshold)"""
    denoise = bool(row.get("denoise"))
    threshold = row.get("denoise_snr_threshold") if denoise else None
    return (row["noise"], row["snr_db"], denoise, row.get("chunk_ms"), threshold)


def _sort_key(key: tuple) -> tuple:
    noise, snr, *rest = key
    return (NOISE_ORDER.get(noise, 9), -(snr if snr is not None else 1e9), *(-1 if v is None else v for v in rest))


def group(clips: list[dict]) -> tuple[dict, dict]:
    utterances: dict[tuple, list[dict]] = {}
    noise_only: dict[tuple, list[dict]] = {}
    for c in clips:
        (noise_only if c["noise_only"] else utterances).setdefault(cond_key(c), []).append(c)
    return utterances, noise_only


def ws_key(d: dict) -> tuple:
    """(engine, 잡음, SNR, 꼬리 ms, 청크 ms, 앞 잡음 ms)"""
    return (str(d.get("engine")), d.get("noise") or "raw", d.get("snr_db"), d.get("tail_silence_ms"),
            d.get("chunk_ms"), d.get("lead_ms"))


def load(paths: list[str]) -> tuple[list[dict], dict[tuple, list[dict]]]:
    """(run_chirp 클립들, ws_bench 턴 ws_key 별)"""
    clips: list[dict] = []
    ws: dict[tuple, list[dict]] = {}
    for path in paths:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if "clips" in d:
            clips.extend(d["clips"])
        elif "turns" in d:
            ws.setdefault(ws_key(d), []).extend(d["turns"])
        else:
            raise SystemExit(f"run_chirp·ws_bench 결과가 아니다: {path}")
    return clips, ws


def _pct(v: float | None) -> str:
    return f"{v * 100:.1f}%" if v is not None else "—"


def _ms(v: float | None) -> str:
    return f"{v:.0f}" if v is not None else "—"


def _snr(v: float | None) -> str:
    return "—" if v is None else f"{v:g}"


def _dn(denoise: bool, threshold: float | None) -> str:
    if not denoise:
        return "off"
    return "on" if not threshold else f"on{threshold:g}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run_chirp 결과 JSON 또는 ws_bench --json-out 파일")
    ap.add_argument("--manifest", default=None, help="ws_bench 결과 채점용 파일명→정답 JSON(깨끗한 발화 manifest)")
    ap.add_argument("--json-out", default=None, help="요약을 JSON 으로 저장")
    ap.add_argument("--details", action="store_true", help="정답과 다른 턴·거짓 FINAL 을 줄마다")
    args = ap.parse_args()

    clips, ws = load(args.runs)
    utterances, noise_only = group(clips)
    report: dict = {"utterance": [], "noise_only": [], "ws_bench": []}

    if utterances:
        print(
            f"{'noise':<7} {'snr':>4} {'dn':>5} {'chunk':>5} {'n':>4} {'err':>3} {'aud':>4} {'FINAL%':>7} "
            f"{'95% CI':>11} {'p50':>6} {'p95':>6} {'CER':>6} {'CER*':>6} {'mis':>4} {'noiseT':>6} {'early':>6} "
            f"{'split':>6} {'emptyF':>6} {'SB':>5}"
        )
        for key in sorted(utterances, key=_sort_key):
            s = summarize_utterances(utterances[key])
            noise, snr, denoise, chunk_ms, threshold = key
            report["utterance"].append({"noise": noise, "snr_db": snr, "denoise": denoise, "chunk_ms": chunk_ms,
                                        "denoise_snr_threshold": threshold, **s})
            ci = f"[{s['final_rate_lo'] * 100:.0f},{s['final_rate_hi'] * 100:.0f}]"
            print(
                f"{noise:<7} {_snr(snr):>4} {_dn(denoise, threshold):>5} {_snr(chunk_ms):>5} {s['n']:>4} "
                f"{s['errors']:>3} {s['n_audio']:>4} {s['final_rate'] * 100:>6.0f}% {ci:>11} "
                f"{_ms(s['final_p50']):>6} {_ms(s['final_p95']):>6} {_pct(s['mean_cer']):>6} "
                f"{_pct(s['mean_cer_all']):>6} {s['mismatches']:>4} {_pct(s['noise_turn_rate']):>6} "
                f"{_pct(s['early_rate']):>6} {_pct(s['split_rate']):>6} {s['empty_finals_per_clip'] or 0:>6.2f} "
                f"{s['speech_begin_per_clip'] or 0:>5.2f}"
            )
            if args.details:
                for c in utterances[key]:
                    t = turn_of(c)
                    if failed(c) or t is None or normalize(t["text"]) != normalize(c["text"]):
                        print(f"    r{c['rep']} [{c['utt']}] {c['stopped']} 정답={c['text']!r} 턴={t and t['text']!r}")
        print("n = 에러 클립(err)을 뺀 클립 · aud = 서로 다른 오디오 수(CI 표본 수) · CER* = 턴 없음을 1 로")
        print("p50/p95 = 말소리 끝→턴 FINAL ms(complete 턴만) · noiseT = 턴이 잡음 · early = 턴이 발화 조각")
        print("split = 발화가 비어 있지 않은 FINAL 2개 이상 · emptyF·SB = 클립당")

    if noise_only:
        print(
            f"\n{'잡음만':<7} {'snr':>4} {'dn':>5} {'chunk':>5} {'clips':>5} {'err':>3} {'분':>6} {'거짓F':>5} "
            f"{'/분':>6} {'클립':>4} {'빈F':>4} {'SB/분':>6}"
        )
        for key in sorted(noise_only, key=_sort_key):
            s = summarize_noise_only(noise_only[key])
            noise, snr, denoise, chunk_ms, threshold = key
            report["noise_only"].append({"noise": noise, "snr_db": snr, "denoise": denoise, "chunk_ms": chunk_ms,
                                         "denoise_snr_threshold": threshold, **s})
            per_min = "—" if s["false_per_min"] is None else f"{s['false_per_min']:.2f}"
            sb = "—" if s["speech_begin_per_min"] is None else f"{s['speech_begin_per_min']:.2f}"
            print(
                f"{noise:<7} {_snr(snr):>4} {_dn(denoise, threshold):>5} {_snr(chunk_ms):>5} {s['clips']:>5} "
                f"{s['errors']:>3} {s['minutes']:>6.2f} {s['false_finals']:>5} {per_min:>6} "
                f"{s['clips_with_false']:>4} {s['empty_finals']:>4} {sb:>6}"
            )
            if args.details and s["false_texts"]:
                print(f"    거짓 FINAL: {s['false_texts']!r}")

    if ws:
        if args.manifest is None:
            raise SystemExit("ws_bench 결과 채점엔 --manifest(파일명→정답)가 필요하다")
        texts = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        print(
            f"\n{'ws engine':<12} {'noise':<7} {'snr':>4} {'tail':>5} {'chunk':>5} {'lead':>5} {'n':>4} "
            f"{'FINAL%':>7} {'p50':>6} {'p95':>6} {'mis':>4} {'CER':>6}"
        )
        for key in sorted(ws, key=lambda k: (k[0], *_sort_key(k[1:]))):
            s = summarize_condition(ws[key], texts)
            engine, noise, snr, tail_ms, chunk_ms, lead_ms = key
            report["ws_bench"].append({"engine": engine, "noise": noise, "snr_db": snr, "tail_silence_ms": tail_ms,
                                       "chunk_ms": chunk_ms, "lead_ms": lead_ms, **s})
            tail = "reply" if tail_ms is None and noise != "raw" else _snr(tail_ms)
            print(
                f"{engine:<12} {noise:<7} {_snr(snr):>4} {tail:>5} {_snr(chunk_ms):>5} {_snr(lead_ms):>5} "
                f"{s['n']:>4} {s['final_rate'] * 100:>6.0f}% {_ms(s['stt_final_p50']):>6} "
                f"{_ms(s['stt_final_p95']):>6} {s['mismatches']:>4} {_pct(s['mean_cer']):>6}"
            )
        print("tail reply = 잡음 모드(응답이 올 때까지 잡음) · raw = 잡음 manifest 없이 원본 파일")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\njson → {args.json_out}")


if __name__ == "__main__":
    main()
