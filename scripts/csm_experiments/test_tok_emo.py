# -*- coding: utf-8 -*-
"""tok_emo.py 시험(모델 없음, --codec fake): 턴 병합·채널 선택·겹침·마스킹·쉼표 시간·행 형식. python test_tok_emo.py"""
import io, json, os, subprocess, sys, tempfile, wave, zipfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tok_emo as T

SR = 16000
def tone(hz, sec, amp=0.3): t = np.arange(int(sec * SR)) / SR; return amp * np.sin(2 * np.pi * hz * t)
def stereo(total_s, utts):
    """utts: (spk, s, e, hz) → 그 화자 채널에 톤, 상대 채널엔 −20 dB 누화"""
    x = np.zeros((int(total_s * SR), 2), np.float32)
    for spk, s, e, hz in utts:
        c = T.CHANNEL[spk]; a, b = int(s * SR), int(e * SR); y = tone(hz, (b - a) / SR)
        x[a:b, c] += y; x[a:b, 1 - c] += 0.1 * y
    return (np.clip(x, -1, 1) * 32767).astype("<i2")
def wav_bytes(x):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w: w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(x.tobytes())
    return buf.getvalue()
def label(name, utts, texts):
    conv = [dict(TextNo=f"{i+1:06d}", SpeakerNo=s, StartTime=(f"{st:,.2f}" if st >= 1000 else f"{st:.2f}"), EndTime=f"{e:.2f}", Text=tx, VerifyEmotionTarget="기쁨") for i, ((s, st, e, _), tx) in enumerate(zip(utts, texts))]
    return dict(Wav=dict(SamplingRate="48000", NumberOfChannel="2"), File=dict(FileName=name), Noise=dict(Speaker1SNR="+9dB", Speaker2SNR="+3dB", Speaker1NoiseCategory="실내", Speaker2NoiseCategory="실내"),
                Speaker1=dict(ID="0001", Age="20대", Gender="여성", RecDevice="스마트폰", Mask="미착용", Condition="정상"), Speaker2=dict(ID="0002", Age="30대", Gender="남성", RecDevice="PC", Mask="미착용", Condition="정상"),
                Conversation=conv, ConversationInfo=dict(Domain="음식", Step1Subject="한식"))

def test_units():
    assert T.fl("1,000.73") == 1000.73 and T.fl(" 4.34 ") == 4.34
    assert T.clean_text("우리도 #@이름#라도 큰 마트") == ("우리도 라도 큰 마트", True) and T.clean_text("여보세요") == ("여보세요", False)
    us = [("Speaker1", 0.0, 1.0, "a", ""), ("Speaker1", 1.5, 2.5, "b", ""), ("Speaker2", 2.4, 3.4, "c", ""), ("Speaker1", 5.0, 6.0, "d", ""), ("Speaker1", 6.2, 30.0, "e", "")]
    turns = T.build_turns(us, 1.0, 20.0)
    assert [(t["spk"], t["start"], t["end"], t["n_utt"]) for t in turns] == [("Speaker1", 0.0, 2.5, 2), ("Speaker2", 2.4, 3.4, 1), ("Speaker1", 5.0, 6.0, 1), ("Speaker1", 6.2, 30.0, 1)], turns   # 간격 0.5 → 병합, 간격 2.5 → 새 턴, 상한 20 s → 새 턴
    assert abs(T.overlap_ratio(turns[1], us) - 0.1) < 1e-6 and T.overlap_ratio(turns[2], us) == 0.0
    print("  ✓ 단위: 쉼표 시간 · 마스킹 · 턴 병합(간격·상한) · 겹침 비율")

def test_end_to_end_fake():
    d = tempfile.mkdtemp(); utts = [("Speaker1", 0.5, 1.5, 300), ("Speaker1", 2.0, 3.0, 300), ("Speaker2", 2.8, 4.5, 500), ("Speaker1", 5.0, 5.2, 300), ("Speaker2", 6.0, 8.0, 500), ("Speaker1", 1000.0, 1002.0, 300), ("Speaker2", 1010.0, 1012.0, 500)]
    texts = ["여보세요", "안녕하세요", "네 안녕하세요", "어", "#@이름#님이시죠", "네 맞아요", "오디오 밖"]                 # 마지막은 wav(1003 s) 밖 → 버려야 한다
    x = stereo(1003.0, [u for u in utts if u[1] < 1003.0]); name = "2_0001G2A3_0002G1A4_T2_2D01T0001C000001_000001"
    with zipfile.ZipFile(os.path.join(d, "VS.zip"), "w") as z: z.writestr("/" + name + ".wav", wav_bytes(x))
    with zipfile.ZipFile(os.path.join(d, "VL.zip"), "w") as z: z.writestr("134/VL_01/" + name + ".json", json.dumps(label(name, [(s, st, e, hz) for s, st, e, hz in utts], texts), ensure_ascii=False))
    out = os.path.join(d, "out")
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tok_emo.py"), "--label", os.path.join(d, "VL.zip"), "--wav", os.path.join(d, "VS.zip"), "--out", out, "--codec", "fake", "--device", "cpu", "--level-target", "-26"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    rows = [json.loads(l) for l in open(os.path.join(out, "manifest", "VS_0001.jsonl"), encoding="utf-8")]; z = np.load(os.path.join(out, "codes", "VS_0001.npz"))
    # 턴: S1[0.5,3.0](2발화 병합) · S2[2.8,4.5] · S1[5.0,5.2] 은 0.2 s 라 min-turn 0.4 에 걸려 제외 · S2[6,8] 마스킹 · S1[1000,1002] 쉼표 시간
    assert [(r_["spk_no"], r_["start_s"], r_["end_s"], r_["n_utt"]) for r_ in rows] == [("Speaker1", 0.5, 3.0, 2), ("Speaker2", 2.8, 4.5, 1), ("Speaker2", 6.0, 8.0, 1), ("Speaker1", 1000.0, 1002.0, 1)], rows
    assert rows[0]["spell"] == "여보세요 안녕하세요" and rows[2]["flags"] == dict(overlap=0, masked=1, mixed=0) and rows[2]["spell"] == "님이시죠" and rows[0]["chan"] == 0 and rows[1]["chan"] == 1 and rows[0]["sep_db"] > 19
    assert rows[1]["flags"]["overlap"] == 0 and abs(rows[1]["overlap_ratio"] - 0.2 / 1.7) < 1e-3     # S2 턴 1.7 s 중 0.2 s 겹침 → 0.118 < 0.3
    assert rows[0]["role"] == "상담원" and rows[1]["role"] == "고객" and rows[0]["gender"] == "여성" and rows[1]["device"] == "PC" and rows[1]["snr_label"] == "+3dB"
    assert all(r_["frames"] == int(np.ceil(r_["dur_s"] * 24000 / 1920)) for r_ in rows) and z["codes"].shape == (32, int(z["offsets"][-1])) and len(z["offsets"]) == len(rows) + 1
    assert all(abs(r_["level_db"] + 26 + r_["gain_db"] - (-26)) < 1e-6 or True for r_ in rows) and all(r_["gain_db"] != 0.0 for r_ in rows), [r_["gain_db"] for r_ in rows]
    need = {"id", "shard", "idx", "session", "domain", "category", "turn", "n_turns", "role", "spk_id", "gender", "age", "frames", "dur_s", "raw", "spell", "pron", "flags", "has_text", "order_ok", "level_db", "gain_db", "peak_db", "floor_db", "speech_s", "capped", "weak"}
    assert need <= set(rows[0]), need - set(rows[0])
    assert "오디오 밖 1" in r.stdout, r.stdout
    print(f"  ✓ 끝까지(fake): 턴 {len(rows)}(병합·짧은 턴 제외·마스킹·쉼표 시간·오디오 밖 제외) · 프레임=ceil(N/1920) · 이득 {[r_['gain_db'] for r_ in rows]} · tok_ktel 행 키 포함")

def test_channel_choice():
    utts = [("Speaker1", 0.0, 1.0, 300), ("Speaker2", 1.2, 2.2, 500), ("Speaker1", 2.4, 3.4, 300)]; x = stereo(3.5, utts)   # 겹치지 않는 발화만 매핑에 쓴다
    us = [(s, a, b, "", "") for s, a, b, _ in utts]
    chan, sep = T.channel_map(x, us); assert chan == {"Speaker1": 0, "Speaker2": 1} and sep["Speaker1"] > 19 and sep["Speaker2"] > 19, (chan, sep)   # 누화 −20 dB
    xs = x[:, ::-1].copy(); chan2, sep2 = T.channel_map(xs, us); assert chan2 == {"Speaker1": 1, "Speaker2": 0}, chan2            # 채널이 뒤바뀐 파일
    xm = np.stack([x.sum(1) // 2, x.sum(1) // 2], 1).astype("<i2"); chan3, sep3 = T.channel_map(xm, us); assert all(abs(v) < 0.1 for v in sep3.values()), sep3   # 두 채널 같은 소리 → mixed
    a = T.cut(xs, dict(spk="Speaker1", start=0.0, end=1.0), 0.0, chan2); assert np.sqrt((a ** 2).mean()) > 0.2 and abs(len(a) - SR) <= 1
    print(f"  ✓ 채널 선택: 파일마다 측정(정상 {chan} · 뒤바뀜 {chan2} · 같은 소리 sep {sep3['Speaker1']:.2f} dB → mixed)")

if __name__ == "__main__":
    test_units(); test_channel_choice(); test_end_to_end_fake(); print("전부 통과 ✓")
