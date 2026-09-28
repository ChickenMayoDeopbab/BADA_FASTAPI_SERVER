# -*- coding: utf-8 -*-
"""level.py 골든 테스트 — 같은 소리를 8/16/24 kHz 로 넣으면 같은 이득, 파고율 큰 입력은 피크 상한, 무음은 weak. python test_level.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import level as L


def synth(sr, seed=0):
    """0.5 s 무음 + 2 s 톤 묶음(말소리 흉내, 진폭 0.05) + 0.5 s 무음 + 잡음 바닥(−60 dBFS)"""
    rng = np.random.default_rng(seed); t = np.arange(int(2.0 * sr)) / sr
    tone = 0.05 * (np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 660 * t)) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    x = np.concatenate([np.zeros(int(0.5 * sr)), tone, np.zeros(int(0.5 * sr))]) + 1e-3 * rng.standard_normal(int(3.0 * sr))
    return x.astype(np.float32)


def main():
    g = {sr: L.analyze(synth(sr), sr) for sr in (8000, 16000, 24000)}
    assert max(v["gain_db"] for v in g.values()) - min(v["gain_db"] for v in g.values()) <= 0.2, g          # 샘플레이트 무관
    a = g[8000]; assert abs(a["level_db"] + a["gain_db"] + 26.0) < 1e-6 and not a["capped"] and not a["weak"] and 1.8 <= a["speech_s"] <= 2.2, a
    print(f"  ✓ 8/16/24 kHz 이득 일치 {[v['gain_db'] for v in g.values()]} · 레벨 {a['level_db']} · 말소리 {a['speech_s']} s")
    y = L.apply_gain(synth(8000), a["gain_db"]); assert abs(L.speech_rms_db(y, 8000) + 26.0) < 0.05 and y.dtype == np.float32
    print("  ✓ apply_gain 뒤 말소리 RMS = −26 dBFS")
    x = synth(8000); x[8000] = 0.9                                                                          # 파고율 큰 입력: 피크 상한이 이득을 줄인다
    c = L.analyze(x, 8000); assert c["capped"] and abs(c["peak_db"] + c["gain_db"] + 2.0) < 1e-6 and c["gain_db"] < a["gain_db"], c
    print(f"  ✓ 피크 상한 −2 dBFS: 이득 {a['gain_db']} → {c['gain_db']}")
    q = L.analyze(synth(8000) * 10 ** (-40 / 20), 8000); assert q["capped"] and q["gain_db"] == 30.0, q     # 이득 상한 +30
    print("  ✓ 이득 상한 +30 dB")
    s = L.analyze(1e-4 * np.random.default_rng(1).standard_normal(8000), 8000); assert s["weak"], s          # 잡음뿐 → weak
    sh = L.analyze(np.concatenate([np.zeros(4000, np.float32), synth(8000)[4000:5200], np.zeros(4000, np.float32)]), 8000); assert sh["weak"] and sh["speech_s"] < 0.3, sh
    print("  ✓ 잡음뿐·0.15 s 말소리 → weak")
    lv = L.analyze(np.zeros(1000, np.float32), 8000); assert np.isfinite(lv["gain_db"]) and lv["weak"]      # 완전 무음도 죽지 않는다
    print("  ✓ 무음 입력 안전")
    print("전부 통과 ✓")


if __name__ == "__main__":
    main()
