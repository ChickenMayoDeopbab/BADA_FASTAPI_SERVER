"""g1b_human_emo.py 시험 — 정상 stereo 파일은 잘라 쓰고, mono·wav 아님 파일은 죽지 않고 건너뛰어 센다(리뷰 2026-09-29)."""
import io, json, os, subprocess, sys, tempfile, wave, zipfile
import numpy as np
SR = 16000


def wav_bytes(x, ch):
    b = io.BytesIO()
    with wave.open(b, "wb") as w: w.setnchannels(ch); w.setsampwidth(2); w.setframerate(SR); w.writeframes(np.asarray(x, dtype="<i2").tobytes())
    return b.getvalue()


def test_skip_bad():
    d = tempfile.mkdtemp(); st = np.zeros((SR * 3, 2), dtype="<i2"); st[SR:2 * SR, 1] = 1000                # 1~2 s 에 ch1 만 소리
    with zipfile.ZipFile(os.path.join(d, "VS.zip"), "w") as z:
        z.writestr("good.wav", wav_bytes(st, 2)); z.writestr("mono.wav", wav_bytes(np.zeros(SR, dtype="<i2"), 1)); z.writestr("junk.wav", b"RIFF not really a wav")
    sd = os.path.join(d, "set"); os.makedirs(sd)
    with open(os.path.join(sd, "set.jsonl"), "w", encoding="utf-8") as f:
        for i, s in enumerate(["good", "mono", "junk", "absent"]): f.write(json.dumps(dict(id=f"e{i}", session=s, target=dict(start_s=1.0, end_s=2.0, chan=1))) + "\n")
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "g1b_human_emo.py"), "--set", sd, "--wav", os.path.join(d, "VS.zip"), "--pad-s", "0"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "사람 원본 1개" in r.stdout and "못 찾음 1" in r.stdout and "wav 형식 아님 2" in r.stdout, r.stdout
    with wave.open(os.path.join(sd, "human", "e0.wav")) as w: y = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    assert w.getnchannels() == 1 and len(y) == SR and y.min() == y.max() == 1000, (len(y), y.min(), y.max())
    assert sorted(os.listdir(os.path.join(sd, "human"))) == ["e0.wav"]
    print("  ✓ stereo 1개 자름(ch1·1~2 s) · mono·깨진 wav 2개 건너뜀 · 없는 세션 1개 못 찾음, 종료 코드 0")


if __name__ == "__main__":
    test_skip_bad(); print("g1b_human_emo 시험 1/1 통과")
