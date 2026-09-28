# -*- coding: utf-8 -*-
"""광대역 확인 — wav 폴더마다 4~8 kHz 에너지 / 0~4 kHz 에너지(dB) 의 중앙값을 낸다. 전화 대역(8 kHz 데이터)으로 끌려갔으면 값이 뚜렷이 낮다.
  python band_ratio.py ~/g1/set/prompt ~/g1/a2_p ~/g1/b1_p      # 폴더 여러 개, 같은 파일 이름끼리 짝 비교도 낸다
wav 는 어떤 샘플레이트든 받되 8 kHz 위 대역이 없는 파일(≤ 8 kHz 샘플레이트)은 '측정 불가' 로 표시한다.
"""
import glob, os, sys, wave
import numpy as np


def read(path):
    with wave.open(path) as w:
        sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth(); b = w.readframes(n)
    x = np.frombuffer(b, dtype={2: "<i2", 4: "<i4"}[sw]).astype(np.float64) / (2 ** (8 * sw - 1))
    return (x.reshape(-1, ch).mean(1) if ch > 1 else x), sr


def ratio_db(x, sr, lo=4000.0, hi=8000.0):
    if sr < 2 * hi: return None
    n = 4096; hop = 2048; win = np.hanning(n); acc = np.zeros(n // 2 + 1)
    for s in range(0, max(1, len(x) - n), hop):
        acc += np.abs(np.fft.rfft(x[s:s + n] * win)) ** 2
    f = np.fft.rfftfreq(n, 1 / sr); low, high = acc[(f >= 300) & (f < lo)].sum(), acc[(f >= lo) & (f < hi)].sum()
    return 10 * np.log10(max(high, 1e-20) / max(low, 1e-20))


def main():
    dirs = [os.path.expanduser(d) for d in sys.argv[1:]]; per = {}
    for d in dirs:
        per[d] = {os.path.basename(p): ratio_db(*read(p)) for p in sorted(glob.glob(os.path.join(d, "*.wav")))}
        v = [x for x in per[d].values() if x is not None]
        print(f"{d}: 파일 {len(per[d])} · 4~8 kHz/0.3~4 kHz 중앙값 {np.median(v):+.1f} dB · p10 {np.percentile(v, 10):+.1f} · p90 {np.percentile(v, 90):+.1f}" if v else f"{d}: 측정 불가(샘플레이트 < 16 kHz)")
    if len(dirs) >= 2:
        base = per[dirs[0]]
        for d in dirs[1:]:
            pairs = [(per[d][k] - base[k]) for k in per[d] if k in base and per[d][k] is not None and base[k] is not None]
            if pairs: print(f"  {os.path.basename(d)} − {os.path.basename(dirs[0])} (같은 파일 이름 짝 {len(pairs)}개): 중앙값 {np.median(pairs):+.1f} dB")


if __name__ == "__main__":
    main()
