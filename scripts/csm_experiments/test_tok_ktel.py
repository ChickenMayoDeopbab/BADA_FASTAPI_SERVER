# -*- coding: utf-8 -*-
"""tok_ktel.py 시험(맥, 모델 없이 돈다). pytest 가 있으면 pytest 로, 없으면 그냥 실행. 실제 Mimi 스모크는 KTEL_REAL=1 일 때만.
  source labs/env.sh && python labs/data/test_tok_ktel.py
"""
import fcntl, io, json, math, os, pathlib, subprocess, sys, tempfile, wave, zipfile
import numpy as np, torch
try: import pytest
except ImportError: pytest = None

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import tok_ktel as tk

SESS = ["D60/J91/S00000001", "D60/J91/S00000002", "D60/J91/S00000003", "D60/J91/S00000004"]
SPK = [dict(id="9855", gender="남", type="상담원", age="20대", residence="서울"), dict(id="tczpppab", gender="남", type="고객", age="60대", residence="경기")]


def tone(hz, sec, sr=8000, amp=0.5):
    return amp * np.sin(2 * np.pi * hz * np.arange(int(sec * sr)) / sr)


def pcm16(sec, seed=0, sr=8000):
    return (np.random.default_rng(seed).standard_normal(int(sec * sr)) * 3000).astype(np.int16)


def wav_bytes(pcm, sr=8000):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())
    return b.getvalue()


def test_resampler_gain_passband_and_images():
    k = tk.resample_kernel("cpu")
    assert k.numel() == 2 * tk.HALF + 1 and abs(k.sum().item() - tk.UP) < 1e-3          # DC 이득 3 = 0 채우기 보상
    for hz in (300.0, 1000.0, 3400.0):
        x = torch.tensor(tone(hz, 2.0), dtype=torch.float32); y = tk.resample_8k_to_24k(x, k)
        assert y.numel() == 3 * x.numel()
        mid = y[24000:36000]                                                            # 가운데 0.5 s(과도 구간 제외). 톤 주기가 정수 개 들어간다
        rms = mid.pow(2).mean().sqrt().item(); assert abs(rms / (0.5 / math.sqrt(2)) - 1) < 0.01, (hz, rms)
        p = torch.fft.rfft(mid.double()).abs() ** 2; f = torch.fft.rfftfreq(mid.numel(), 1 / 24000)
        leak = p[f > 4000].sum() / p.sum()
        assert leak < 1e-8, (hz, leak.item())                                           # 4 kHz 위(영상 성분) ≤ -80 dB


def test_build_turns_merges_same_speaker_with_gap_and_cap():
    one = lambda s: np.ones(int(s * 8000), dtype=np.int16)
    utts = [("0001", "A", "안녕", one(2)), ("0002", "B", "네", one(1)), ("0003", "B", "저기요", one(1.5)),
            ("0004", "A", "예", one(1)), ("0005", "A", "긴 말", one(12)), ("0006", "A", "더 긴 말", one(12)), ("0007", "A", "혼자 긴 말", one(25))]
    t = tk.build_turns(utts, 0.3, 20.0)
    assert [x["spk"] for x in t] == ["A", "B", "A", "A", "A"]
    assert [x["utts"] for x in t] == [["0001"], ["0002", "0003"], ["0004", "0005"], ["0006"], ["0007"]]   # 1+0.3+12 ≤ 20, +12 는 넘음, 25 s 혼자는 그대로
    assert t[1]["pcm"].size == 8000 + 2400 + 12000 and int(t[1]["pcm"][8000:10400].max()) == 0 and t[1]["utt_s"] == [1.0, 1.5]
    assert t[1]["raws"] == ["네", "저기요"]


def make_zips(tmp):
    """세션 4개: S1 정상(A,B,B,A) · S2 wav 하나 빠짐 · S3 16 kHz wav · S4 정상(A,B). 라벨 없는 S5 는 wav 만."""
    lab, wav = os.path.join(tmp, "KtelSpeech_valid_D60_label_0.zip"), os.path.join(tmp, "KtelSpeech_valid_D60_wav_0.zip")
    texts = {SESS[0]: [("0001", "9855", "안녕하세요 n/ 상담원입니다.", 2.0), ("0002", "tczpppab", "네 (1권)/(한 권) 어/ 문의요", 1.0), ("0003", "tczpppab", "b/ 배송이요   ", 1.5), ("0004", "9855", "예", 1.0)],
             SESS[1]: [("0001", "9855", "여보세요", 1.0), ("0002", "tczpppab", "네", 1.0)],
             SESS[2]: [("0001", "9855", "여보세요", 1.0)],
             SESS[3]: [("0001", "9855", "안녕하세요", 1.2), ("0002", "tczpppab", "네 안녕하세요", 1.4)]}
    with zipfile.ZipFile(lab, "w") as lz, zipfile.ZipFile(wav, "w") as wz:
        for sess, utts in texts.items():
            dialogs = [dict(speaker=spk, audioPath=f"KtelSpeech/{sess}/{uid}.wav", textPath=f"KtelSpeech/{sess}/{uid}.txt") for uid, spk, _, _ in utts]
            lz.writestr(f"{sess}/{os.path.basename(sess)}.json", json.dumps(dict(dataSet=dict(version="1.0", typeInfo=dict(category="교육", speakers=SPK), dialogs=dialogs)), ensure_ascii=False, indent=4))
            for i, (uid, spk, text, sec) in enumerate(utts):
                lz.writestr(f"{sess}/{uid}.txt", text)
                if sess == SESS[1] and uid == "0002": continue                                    # 빠진 wav
                sr = 16000 if sess == SESS[2] else 8000
                wz.writestr(f"{sess}/{uid}.wav", wav_bytes(pcm16(sec, seed=i, sr=sr), sr))
        wz.writestr("D60/J91/S00000005/0001.wav", wav_bytes(pcm16(1.0)))
    return lab, wav, texts


def run(args, **kw):
    return subprocess.run([sys.executable, os.path.join(HERE, "tok_ktel.py")] + args, capture_output=True, text=True, **kw)


def test_end_to_end_fake_codec(tmp_path=None):
    tmp_path = pathlib.Path(tmp_path or tempfile.mkdtemp()); lab, wav, texts = make_zips(str(tmp_path)); out = str(tmp_path / "tok")
    base = ["--label", lab, "--wav", wav, "--out", out, "--codec", "fake", "--sessions-per-shard", "2", "--device", "cpu"]
    r = run(base); assert r.returncode == 0, r.stderr
    names = sorted(os.listdir(os.path.join(out, "manifest"))); assert names == ["KtelSpeech_valid_D60_wav_0_0001.jsonl", "KtelSpeech_valid_D60_wav_0_0002.jsonl"]     # 3번째 샤드(S5 만)는 안 쓴다
    rows = [json.loads(l) for l in open(os.path.join(out, "manifest", names[0]), encoding="utf-8")]
    assert len(rows) == 3 and [r["role"] for r in rows] == ["상담원", "고객", "상담원"] and [r["utts"] for r in rows] == [["0001"], ["0002", "0003"], ["0004"]]
    b = rows[1]
    assert b["raw"] == "네 (1권)/(한 권) 어/ 문의요 b/ 배송이요" and b["spell"] == "네 1권 어 문의요 [숨] 배송이요" and b["pron"] == "네 한 권 어 문의요 [숨] 배송이요"
    assert b["flags"]["dual"] == 1 and b["flags"]["filler"] == 1 and b["flags"]["breath"] == 1 and b["n_utt"] == 2 and b["utt_s"] == [1.0, 1.5]
    assert b["dur_s"] == round((8000 + 2400 + 12000) / 8000, 3) and b["frames"] == math.ceil(3 * (8000 + 2400 + 12000) / tk.FRAME)
    assert all(r["session"] == SESS[0] and r["domain"] == "D60" and r["category"] == "교육" and r["n_turns"] == 3 and r["order_ok"] and r["spk_id"] for r in rows)
    assert rows[0]["gender"] == "남" and rows[0]["age"] == "20대" and rows[1]["age"] == "60대"
    z = np.load(os.path.join(out, "codes", names[0][:-6] + ".npz")); off = z["offsets"]
    assert z["codes"].shape == (32, sum(r["frames"] for r in rows)) and off[-1] == z["codes"].shape[1] and len(off) - 1 == 3 and list(z["ids"]) == [r["id"] for r in rows]
    assert z["codes"].dtype == np.int16 and 0 <= z["codes"].min() and z["codes"].max() < 2048
    rows2 = [json.loads(l) for l in open(os.path.join(out, "manifest", names[1]), encoding="utf-8")]; assert [r["session"] for r in rows2] == [SESS[3]] * 2
    assert "'라벨 없음': 1" in r.stdout and "'파일 빠짐(wav 또는 txt)': 1" in r.stdout and "'wav 형식 아님(8000 Hz·16 bit·mono 가 아니다)': 1" in r.stdout, r.stdout
    assert "세션 2 · 턴 5" in r.stdout and json.load(open(os.path.join(out, "meta.json")))["codec"] == "fake"
    # 이어 돌리기: 끝난 샤드는 건너뛴다(파일이 안 바뀐다)
    m0 = os.path.getmtime(os.path.join(out, "codes", names[0][:-6] + ".npz")); r2 = run(base); assert r2.returncode == 0 and "세션 0 · 턴 0" in r2.stdout
    assert os.path.getmtime(os.path.join(out, "codes", names[0][:-6] + ".npz")) == m0
    # 잠금: 다른 프로세스가 잡고 있으면 거부(종료 코드 2)
    lk = open(os.path.join(out, ".lock"), "a+"); fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    r3 = run(base); assert r3.returncode == 2 and "다른 프로세스" in r3.stderr, (r3.returncode, r3.stderr); lk.close()
    # check_tokens.py 가 그대로 통한다
    r4 = subprocess.run([sys.executable, os.path.join(HERE, "check_tokens.py"), out], capture_output=True, text=True); assert r4.returncode == 0, r4.stdout
    # stat_ktel.py 도 돈다
    r5 = subprocess.run([sys.executable, os.path.join(HERE, "stat_ktel.py"), out], capture_output=True, text=True); assert r5.returncode == 0 and "세션 2 · 턴 5" in r5.stdout, r5.stdout + r5.stderr


@(pytest.mark.skipif(not os.environ.get("KTEL_REAL"), reason="KTEL_REAL=1 일 때만(실제 Mimi, CPU)") if pytest else (lambda f: f))
def test_real_mimi_frames_per_second():
    import argparse
    enc = tk.make_codec(argparse.Namespace(codec="mimi", mimi="kyutai/mimi"), torch.device("cpu")); k = tk.resample_kernel("cpu")
    for sec in (2.0, 2.02, 5.16):
        x = tk.resample_8k_to_24k(torch.tensor(tone(440.0, sec), dtype=torch.float32), k); c = enc(x)
        assert c.shape[0] == 32 and c.dtype == np.int16, c.shape
        print(f"{sec} s → {c.shape[1]} 프레임 (ceil {math.ceil(x.numel() / tk.FRAME)})")


if __name__ == "__main__":
    tests = [test_resampler_gain_passband_and_images, test_build_turns_merges_same_speaker_with_gap_and_cap, test_end_to_end_fake_codec] + ([test_real_mimi_frames_per_second] if os.environ.get("KTEL_REAL") else [])
    for f in tests:
        f(); print(f"  ✓ {f.__name__}")
    print("전부 통과 ✓")
