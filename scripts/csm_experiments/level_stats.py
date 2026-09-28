# -*- coding: utf-8 -*-
"""E-C — 학습 wav zip 한 조각의 세션 N개에 level.analyze 만 돌려 이득·상한·weak 분포를 본다(토큰화 없음, GPU 없음). 집 PC 에서.
  python level_stats.py --label <라벨 zip…> --wav <wav zip> --sessions 100 [--target -26]
"""
import argparse, collections, os, re, sys, zipfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tok_ktel import label_index, load_session, build_turns, SessionSkip, SR_IN
from level import analyze


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--label", nargs="+", required=True); ap.add_argument("--wav", required=True)
    ap.add_argument("--sessions", type=int, default=100); ap.add_argument("--target", type=float, default=-26.0); ap.add_argument("--gap-s", type=float, default=0.3); ap.add_argument("--max-turn-s", type=float, default=20.0)
    a = ap.parse_args(); m = re.search(r"_(D\d+)_", os.path.basename(a.wav)); labels, _ = label_index(a.label, {m.group(1)} if m else set())
    wz = zipfile.ZipFile(os.path.expanduser(a.wav)); sessions = sorted({os.path.dirname(i.filename) for i in wz.infolist() if i.filename.lower().endswith(".wav")})[: a.sessions]
    rows, skipped = [], collections.Counter()
    for sess in sessions:
        try: utts, meta = load_session(sess, labels, wz)
        except SessionSkip as e: skipped[e.reason] += 1; continue
        for t in build_turns(utts, a.gap_s, a.max_turn_s):
            lv = analyze(t["pcm"].astype(np.float32) / 32768.0, SR_IN, a.target); lv["role"] = meta["spk"][t["spk"]].get("type"); lv["dur_s"] = t["pcm"].size / SR_IN; rows.append(lv)
    L = np.array([r["level_db"] for r in rows]); G = np.array([r["gain_db"] for r in rows]); F = np.array([r["floor_db"] + r["gain_db"] for r in rows])
    q = lambda v: f"{np.percentile(v, 10):+.1f} / {np.median(v):+.1f} / {np.percentile(v, 90):+.1f} / 최대 {v.max():+.1f}"
    print(f"세션 {len(sessions) - sum(skipped.values())}개 · 턴 {len(rows):,}개 · 건너뜀 {dict(skipped)} · 목표 {a.target} dBFS")
    print(f"원본 말소리 RMS p10/p50/p90/최대: {q(L)} dBFS · < −38 인 턴 {np.mean(L < -38):.0%}")
    print(f"이득 p10/p50/p90/최대: {q(G)} dB · 피크 상한 걸림 {np.mean([r['capped'] for r in rows]):.1%} · 이득 ≥ +25 {np.mean(G >= 25):.1%} · weak {np.mean([r['weak'] for r in rows]):.1%}")
    print(f"정규화 뒤 잡음 바닥 p50/p90/최대: {np.median(F):+.1f} / {np.percentile(F, 90):+.1f} / {F.max():+.1f} dBFS · SNR(말소리−바닥) p10 {np.percentile([r['level_db'] - r['floor_db'] for r in rows], 10):.1f} dB")
    for role in sorted({r["role"] for r in rows}, key=str):
        rr = [r for r in rows if r["role"] == role]; print(f"  {role}: 턴 {len(rr):,} · 원본 레벨 p50 {np.median([r['level_db'] for r in rr]):+.1f} · 이득 p50 {np.median([r['gain_db'] for r in rr]):+.1f} · weak {np.mean([r['weak'] for r in rr]):.1%}")
    short = [r for r in rows if r["dur_s"] < 1.0]; print(f"  1 s 미만 턴 {len(short):,}개: weak {np.mean([r['weak'] for r in short]) if short else 0:.0%} · 이득 p90 {np.percentile([r['gain_db'] for r in short], 90) if short else 0:+.1f}")


if __name__ == "__main__":
    main()
