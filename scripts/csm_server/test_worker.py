# -*- coding: utf-8 -*-
"""워커 단위 시험(맥 CPU, 실 모델 sesame/csm-1b 캐시). HF_HUB_OFFLINE=1 python test_worker.py [--weights <ckpt>]"""
import argparse, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csm_worker as W


def fake_codes(T, seed):
    return torch.randint(1, 2048, (T, 32), generator=torch.Generator().manual_seed(seed))


def test_session_incremental_equals_oneshot(tok, model, sc):
    turns = [(0, "안녕하세요, 무엇을 도와드릴까요?", fake_codes(30, 1)), (1, "어 제가 어제 주문한 게 아직 안 왔어요.", fake_codes(50, 2)), (0, "네, 확인해 드리겠습니다.", None)]
    a = W.Session(model, sc, tok)                                            # (a) 한 번에: rebase 경로로 통째 프리필
    for tag, text, c in turns[:2]: a.turns.append(dict(tag=tag, ids=W.G.text_ids(tok, text, tag), codes=c, start=0))
    a.turns.append(dict(tag=0, ids=W.G.text_ids(tok, turns[2][1], 0), codes=None, start=0)); a.rebase()
    ea, pa = sc.s.emb.clone(), a.pos; fa = a.greedy_first_frame()                      # 은닉값은 depth 스텝 전에, 프레임은 프리필 직후에(상태 공유)
    b = W.Session(model, sc, tok)                                            # (b) 턴마다 이어 붙이기
    ms = [b.append_turn(tag, text, c) for tag, text, c in turns[:2]] + [b.append_text(0, turns[2][1])]
    eb, pb = sc.s.emb.clone(), b.pos; fb = b.greedy_first_frame()
    assert pa == pb, (pa, pb); assert torch.allclose(ea, eb, atol=1e-4, rtol=1e-4), (ea - eb).abs().max()
    assert torch.equal(fa, fb), f"탐욕 첫 프레임 불일치 {int((fa != fb).sum())}/32"
    print(f"  ✓ 이어 붙인 프리필 = 한 번에 프리필: 위치 {pb} · 은닉값 최대 차 {(ea - eb).abs().max():.1e} · 첫 프레임 32코드 일치 · 프리필 {[round(m) for m in ms]} ms")
    return b


def test_commit_and_rebase(tok, model, sc):
    s = W.Session(model, sc, tok, budget=260)                               # 예산을 작게 → 재프리필이 일어나게
    s.append_turn(0, "참조 문장입니다.", fake_codes(40, 9))                    # 참조 턴(항상 남는다)
    for k in range(6):
        s.append_turn(1 if k % 2 == 0 else 0, f"턴 {k} 입니다 {'네 ' * (k + 1)}", fake_codes(45, 10 + k))
    assert s.pos <= 260 and s.turns[0]["codes"].shape[0] == 40 and len(s.turns) < 7, (s.pos, len(s.turns))
    kept = [t["tag"] for t in s.turns]
    # 재프리필 결과 = 남은 턴만 새 세션에 한 번에 넣은 것
    r = W.Session(model, sc, tok, budget=260); r.turns = [dict(t) for t in s.turns]; r.rebase(); assert r.rebases == 1; fr = r.greedy_first_frame()
    s2 = W.Session(model, sc, tok, budget=260); s2.turns = [dict(t) for t in s.turns]; s2.rebase(); fs = s2.greedy_first_frame()
    assert r.pos == s.pos and torch.equal(fr, fs), (r.pos, s.pos)
    s.rebase()                                                                           # 공유 상태를 s 의 캐시로 되돌린다
    print(f"  ✓ 재프리필: 7턴 → 남은 {len(s.turns)}턴 {kept} · 위치 {s.pos} ≤ 260 · 참조 턴 유지")
    s.append_text(0, "이제 말할 차례입니다."); a0 = s.turns[-1]["start"] + len(s.turns[-1]["ids"]); before = s.pos
    gen = fake_codes(12, 77); s.commit_audio(gen)
    assert s.turns[-1]["codes"] is gen and s.pos == a0 + 13 and s.pos == before + 13
    # 확정 뒤 캐시 = 같은 턴들을 한 번에 넣은 것
    assert s.pos <= 260 + 250 + 1, s.pos                                                 # 확정 직후는 예산(문맥 상한)을 넘을 수 있다 — 생성 프레임 몫. 2,048 안이면 된다
    s.append_turn(1, "다음 사용자 턴입니다.", fake_codes(30, 78))                             # 다음 턴을 붙이면 재프리필로 예산 안
    assert s.pos <= 260, s.pos; e0 = sc.s.emb.clone(); f0 = s.greedy_first_frame()
    c = W.Session(model, sc, tok, budget=260); c.turns = [dict(t) for t in s.turns]; c.rebase(); e1 = sc.s.emb.clone(); f1 = c.greedy_first_frame()
    assert c.pos == s.pos and torch.equal(f0, f1) and torch.allclose(e0, e1, atol=1e-4), (c.pos, s.pos, int((f0 != f1).sum()))
    print(f"  ✓ commit_audio: 오디오 자리 {a0}.. 에 {gen.shape[0]}프레임 + eos 확정 → 다음 턴에서 재프리필(위치 {s.pos} ≤ 260) = 한 번에 넣은 캐시")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--weights", default="sesame/csm-1b"); a = ap.parse_args()
    t0 = time.time(); tok, model, sc = W.load_model(a.weights, device="cpu", greedy=True); print(f"모델 로드 {time.time() - t0:.0f} s (CPU fp32, 탐욕)")
    test_session_incremental_equals_oneshot(tok, model, sc)
    test_commit_and_rebase(tok, model, sc)
    test_codec(tok, model, sc)
    test_generator(tok, model, sc)
    print("전부 통과 ✓")



@torch.no_grad()
def test_codec(tok, model, sc):
    import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "csm_experiments")); from level import speech_rms_db
    codec = W.Codec(model); rng = np.random.default_rng(0)
    t = np.arange(int(5.0 * 16000)) / 16000; pcm = (0.02 * np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)) * 32767).astype(np.int16)   # 조용한(≈ −38) 5 s
    codes, lv = codec.encode_pcm16k(pcm); assert codes.shape[1] == 32 and 60 <= codes.shape[0] <= 64, codes.shape
    y = codec.mimi.decode(codes.T[None]).audio_values[0, 0].float().numpy(); l_out = speech_rms_db(y, 24000)
    assert abs(l_out + 26) < 2.0, f"정규화 인코드 뒤 레벨 {l_out:.1f} dBFS"
    print(f"  ✓ encode_pcm16k: 입력 {lv['level_db']:+.1f} dBFS → 이득 {lv['gain_db']:+.1f} → 코덱 왕복 뒤 {l_out:+.1f} dBFS · {codes.shape[0]}프레임")
    full = codec.mimi.decode(codes.T[None]).audio_values[0, 0].float().numpy()
    out = np.concatenate([codec.decode_tail(codes[: i + 2], 2) for i in range(0, codes.shape[0] - 1, 2)]); n = min(len(out), len(full))
    d = np.abs(out[:n] - full[:n]).max(); assert d < 1e-5, f"프리픽스 재디코드 꼬리 vs 전체 디코드 최대 차 {d:.2e}"
    print(f"  ✓ decode_tail(프리픽스 재디코드, 2프레임씩): 전체 디코드와 최대 차 {d:.1e}")


@torch.no_grad()
def test_generator(tok, model, sc):
    import threading, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "csm_experiments")); from level import speech_rms_db
    codec = W.Codec(model); t = np.arange(int(3.0 * 16000)) / 16000
    ref = (0.05 * np.sin(2 * np.pi * 180 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) * 32767).astype(np.int16)
    s = W.Session(model, sc, tok); codes, _ = codec.encode_pcm16k(ref); s.append_turn(0, "네, 안녕하세요. 무엇을 도와드릴까요?", codes); s.append_text(0, "네, 확인해 드리겠습니다.")
    g = W.Generator(model, sc, s, codec, max_frames=20, chunk_frames=2); chunks = list(g.run()); info = g.info
    n16 = sum(len(c) // 2 for c in chunks); assert all(len(c) % 2 == 0 for c in chunks) and chunks
    assert abs(n16 - info["frames"] * 1280) <= 1280, (n16, info["frames"])                    # 1프레임 = 80 ms = 1,280 샘플(16 k)
    assert s.turns[-1]["codes"] is not None and s.turns[-1]["codes"].shape[0] == info["frames"] and s.pos == s.turns[-1]["start"] + len(s.turns[-1]["ids"]) + info["frames"] + 1
    y = codec.mimi.decode(s.turns[-1]["codes"].T[None]).audio_values[0, 0].float().numpy(); lv = speech_rms_db(y, 24000)
    print(f"  ✓ Generator: {info['frames']}프레임 · 청크 {len(chunks)}개 · {n16} 샘플 · eos {info['eos']} · TTFA {info['ttfa_ms']:.0f} ms(CPU) · 캐시 확정 위치 {s.pos} · 재인코딩 턴 레벨 {lv:+.1f} dBFS(모델 원 레벨 {info['raw_level_db']:+.1f})")
    assert abs(lv + 26) < 2.5, lv
    ev = threading.Event(); s2 = W.Session(model, sc, tok); s2.append_turn(0, "네, 안녕하세요.", codes); s2.append_text(0, "네, 확인해 드리겠습니다.")
    g2 = W.Generator(model, sc, s2, codec, max_frames=20, chunk_frames=2); out = []
    for c in g2.run(ev):
        out.append(c)
        if len(out) == 1: ev.set()                                                                  # 첫 청크 뒤 취소
    assert g2.info["cancelled"] and g2.info["frames"] <= 4 and s2.turns[-1]["codes"].shape[0] == g2.info["frames"]
    print(f"  ✓ 취소: 첫 청크 뒤 플래그 → {g2.info['frames']}프레임에서 멈춤, 낸 만큼만 확정")


if __name__ == "__main__":
    main()
