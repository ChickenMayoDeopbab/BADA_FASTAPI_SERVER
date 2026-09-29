# -*- coding: utf-8 -*-
"""G1e(71631 검증) 세트의 사람 원본 — set.jsonl 의 target(start_s·end_s·chan)으로 원천 zip 의 stereo wav 에서 그 화자 채널을 잘라 human/<id>.wav(16 kHz mono) 로. 집 PC 에서 돈다(zip 을 풀지 않는다).
  python g1b_human_emo.py --set <set 폴더(set.jsonl)> --wav <VS_02.실외.zip …> --out <set 폴더>/human [--pad-s 0.05]"""
import argparse, io, json, os, sys, wave, zipfile
import numpy as np
SR = 16000


class BadWav(Exception):
    """stereo 16-bit 16 kHz 가 아니거나 wav 로 못 읽는 파일 — 건너뛰고 센다(tok_emo 의 SessionSkip 과 같은 패턴)."""


def cut_from_zip(zips, session, start_s, end_s, chan, pad_s):
    for z in zips:
        for n in z.namelist():
            if os.path.splitext(os.path.basename(n))[0] == session and n.lower().endswith(".wav"):
                try:
                    with wave.open(io.BytesIO(z.read(n))) as w:
                        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (2, 2, SR): raise BadWav(f"{n}: {w.getnchannels()} ch·{w.getsampwidth()*8} bit·{w.getframerate()} Hz")
                        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").reshape(-1, 2)
                except (wave.Error, EOFError) as e: raise BadWav(f"{n}: wav 아님({e})")
                a = max(0, int(round((start_s - pad_s) * SR))); b = min(len(x), int(round((end_s + pad_s) * SR)))
                return x[a:b, chan].copy()
    raise KeyError(session)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True); ap.add_argument("--wav", nargs="+", required=True); ap.add_argument("--out", default=None); ap.add_argument("--pad-s", type=float, default=0.05)
    a = ap.parse_args(); out = a.out or os.path.join(a.set, "human"); os.makedirs(out, exist_ok=True)
    zips = [zipfile.ZipFile(p) for p in a.wav]; n_ok = n_miss = n_bad = 0
    for line in open(os.path.join(a.set, "set.jsonl"), encoding="utf-8"):
        it = json.loads(line); t = it["target"]
        if t.get("start_s") is None or t.get("chan") is None: n_miss += 1; continue
        try: pcm = cut_from_zip(zips, it["session"], float(t["start_s"]), float(t["end_s"]), int(t["chan"]), a.pad_s)
        except KeyError: n_miss += 1; continue
        except BadWav as e: n_bad += 1; print(f"  건너뜀 {it['id']}: {e}", file=sys.stderr); continue
        with wave.open(os.path.join(out, it["id"] + ".wav"), "wb") as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
        n_ok += 1
    print(f"사람 원본 {n_ok}개 → {out}" + (f" · 못 찾음 {n_miss}" if n_miss else "") + (f" · wav 형식 아님 {n_bad}" if n_bad else ""))


if __name__ == "__main__":
    main()
