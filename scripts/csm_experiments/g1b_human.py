# -*- coding: utf-8 -*-
"""G1b 사람 원본 — set.jsonl 의 session·utts 로 검증 D60 wav zip 에서 원본 발화를 꺼내 0.3 초 무음으로 이어 human/<id>.wav(8 kHz)를 쓴다. 집 PC 에서 돈다.
  python g1b_human.py --set ~/g1b/set --wav ~/aihub/012.상담_음성_데이터/01.데이터/2.Validation/원천데이터_1129_add/KtelSpeech_valid_D60_wav_0.zip
"""
import argparse, io, json, os, wave, zipfile
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True); ap.add_argument("--wav", required=True); ap.add_argument("--gap-s", type=float, default=0.3)
    a = ap.parse_args(); set_dir = os.path.expanduser(a.set); os.makedirs(os.path.join(set_dir, "human"), exist_ok=True)
    zf = zipfile.ZipFile(os.path.expanduser(a.wav)); gap = np.zeros(int(round(a.gap_s * 8000)), np.int16); n = 0
    for line in open(os.path.join(set_dir, "set.jsonl"), encoding="utf-8"):
        it = json.loads(line); pcm = []
        for u in it["target"]["utts"]:
            with wave.open(io.BytesIO(zf.read(f"{it['session']}/{u}.wav"))) as w:
                assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 8000); pcm += [np.frombuffer(w.readframes(w.getnframes()), "<i2"), gap]
        with wave.open(os.path.join(set_dir, "human", it["id"] + ".wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000); w.writeframes(np.concatenate(pcm[:-1]).tobytes())
        n += 1
    print(f"human/ {n}개 → {set_dir}/human")


if __name__ == "__main__":
    main()
