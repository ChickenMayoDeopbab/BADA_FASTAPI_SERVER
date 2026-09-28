# -*- coding: utf-8 -*-
"""csm_data_b 단위 시험(모델 없음, 가짜 토크나이저·가짜 샤드). python test_csm_data_b.py  또는 pytest."""
import os, random, subprocess, sys, tempfile, types
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import csm_data_b as B


class FakeTok:
    bos_token, eos_token, pad_token_id = "<b>", "<e>", 0
    def __call__(self, s, add_special_tokens=False):
        return types.SimpleNamespace(input_ids=[ord(c) % 1000 + 1 for c in s])


CFG = types.SimpleNamespace(audio_token_id=128002, audio_eos_token_id=128003, codebook_eos_token_id=0)


def mk(T, tag=0, ok=True, text="가나다"):
    c = torch.randint(1, 2048, (T, 32))
    return dict(ids=B.turn_ids(FakeTok(), tag, text), codes=c, frames=T, tag=tag, role="상담원" if tag == 0 else "고객", has_text=True, target_ok=ok)


def test_turn_ids_and_len():
    ids = B.turn_ids(FakeTok(), 1, "네"); assert ids[:5] == [ord(c) % 1000 + 1 for c in "<b>[1"]
    t = mk(10); assert B.turn_len(t) == len(t["ids"]) + 11


def test_select_targets_is_disjoint_and_covers():
    turns = [mk(20) for _ in range(40)]; turns[5]["target_ok"] = False
    sel = [B.select_targets(turns, "D60/J91/S1", e, 0, 4) for e in range(4)]
    assert all(0 not in s and 5 not in s for s in sel) and sorted(sum(sel, [])) == [i for i in range(1, 40) if i != 5]
    assert sel[0] == B.select_targets(turns, "D60/J91/S1", 4, 0, 4) and sel[0] != B.select_targets(turns, "D60/J91/S1", 0, 1, 4)
    assert B.select_targets(turns, "D60/J91/S1", 0, 0, 1) == [i for i in range(1, 40) if i != 5]      # every=1 → 전부


def idx(turns, w): return [next(k for k, u in enumerate(turns) if u is t) for t in w]     # dict 안에 텐서가 있어 == 비교 대신 동일성으로


def test_window_budgets():
    turns = [mk(400) for _ in range(10)]                                  # 각 400프레임
    assert idx(turns, B.window(turns, 9, 1500, 100000)) == [6, 7, 8]      # 1200 ≤ 1500 < 1600
    assert idx(turns, B.window(turns, 2, 1500, 100000)) == [0, 1]         # 세션 처음
    small = [mk(100) for _ in range(30)]; L = B.turn_len(small[0])
    w = B.window(small, 29, 100000, 2048); assert sum(B.turn_len(t) for t in w) + L <= 2048 < sum(B.turn_len(t) for t in w) + 2 * L
    assert B.window(turns, 0, 1500, 2048) == []


def test_collate_labels_table():
    ctx, tgt = [mk(30, 0), mk(50, 1)], mk(40, 0)
    b = B.collate_b([(ctx, tgt)], CFG, 0, 1.0, random.Random(0)); lab, ids = b["labels"][0], b["input_ids"][0]
    n_text = sum(len(t["ids"]) for t in ctx + [tgt]); L = n_text + 120 + 3
    assert ids.shape[0] == L and int(b["attention_mask"].sum()) == L and int(b["audio_mask"].sum()) == 123
    assert (lab[:, 0] != -100).sum() == 123                               # 백본: 전 프레임 + eos 3
    assert (~(lab[:, 1:] == -100).all(-1)).sum() == 40 + 3                # depth: 목표 프레임 + eos 3(문맥 eos 도 전부 0 레이블)
    tm = b["target_mask"][0]; assert tm.sum() == 41 and (lab[tm][:, 1:] != -100).all()
    text_pos = (ids != CFG.audio_token_id) & (ids != CFG.audio_eos_token_id) & (b["attention_mask"][0] == 1); assert (lab[text_pos] == -100).all()
    assert torch.equal(b["codes"][0][b["audio_mask"][0]][:30], ctx[0]["codes"])          # 문맥 코드가 자리대로 들어간다
    eos_pos = (ids == CFG.audio_eos_token_id).nonzero().flatten(); assert (lab[eos_pos] == 0).all() and (b["codes"][0][eos_pos] == 0).all()
    b2 = B.collate_b([(ctx, tgt)], CFG, 0, 0.5, random.Random(0)); assert (~(b2["labels"][0][:, 1:] == -100).all(-1)).sum() == 40 - 20 + 3
    b3 = B.collate_b([([], tgt)], CFG, 0, 1.0, random.Random(0)); assert b3["input_ids"].shape[1] == len(tgt["ids"]) + 41
    b4 = B.collate_b([(ctx, tgt), ([], tgt)], CFG, 0, 1.0, random.Random(0))             # 오른쪽 패딩
    assert b4["input_ids"].shape == (2, L) and int(b4["attention_mask"][1].sum()) == len(tgt["ids"]) + 41 and (b4["labels"][1][len(tgt["ids"]) + 41:] == -100).all()


def test_dodge_depth_32():
    ctx = [mk(20, 0), mk(20, 1)]
    for T, want in ((29, 31), (30, 33), (5, 8)):                          # depth 프레임 = T + eos 3 → 32 일 때만 하나 줄인다
        lab = B.collate_b([(ctx, mk(T, 0))], CFG, 0, 1.0, random.Random(0))["labels"]; before = lab.clone()
        out = B.dodge_depth_32(lab); n = (~(out[:, :, 1:] == -100).all(-1)).sum().item()
        assert n == want and (out[:, :, 0] == before[:, :, 0]).all()        # 코드북 0(백본)은 그대로


def test_mixed_order():
    out = list(B.mixed(iter(["b%d" % i for i in range(7)]), iter(["a0", "a1"]), 4))
    assert [k for k, _ in out] == ["B", "B", "B", "A", "B", "B", "B", "A", "B"] and [v for _, v in out] == ["b0", "b1", "b2", "a0", "b3", "b4", "b5", "a1", "b6"]
    assert [k for k, _ in B.mixed(iter(["b0", "b1"]), iter([]), 4)] == ["B", "B"] and [k for k, _ in B.mixed(iter(["b0"]), iter(["a0"]), 0)] == ["B"]


def test_shards_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, os.path.join(HERE, "make_fake_tokens.py"), d, "--layout", "ktel", "--shards", "2", "--sessions", "5"], check=True, capture_output=True)
        tr = B.TurnShards(d, FakeTok(), CFG, split="train"); dv = B.TurnShards(d, FakeTok(), CFG, split="dev")
        assert len(tr.names) == 2 and dv.names == ["KtelSpeech_valid_D60_wav_0_0001"]
        sess = tr.load(tr.names[0], random.Random(0)); assert len(sess) == 5 and all(t["tag"] in (0, 1) for _, ts in sess for t in ts)
        assert all(t["tag"] == (0 if t["role"] == "상담원" else 1) for _, ts in sess for t in ts)
        ex = tr.examples(*sess[0], epoch=0, seed=0); assert ex and all(len(c) >= 1 for c, _ in ex)
        n = sum(1 for _ in tr.batches(2048, 1.0, seed=0, epoch=0)); assert n > 0
        b = next(tr.batches(2048, 1.0, seed=0, epoch=0)); assert b["input_ids"].shape[1] <= 2048
        b0 = next(dv.batches(1024, 1.0, seed=0, epoch=0, shuffle=False, no_ctx=True)); assert b0["target_mask"].sum() == b0["audio_mask"].sum()   # 문맥 없음 = 전부 목표
        assert [len(s) for _, s in tr.load(tr.names[0], random.Random(0), with_codes=False)] == [len(s) for _, s in sess]
        rnd = B.TurnShards(d, FakeTok(), CFG, split="train", tag_by="random").load(tr.names[0], random.Random(0))
        assert all(len({t["tag"] ^ (0 if t["role"] == "상담원" else 1) for t in ts}) == 1 for _, ts in rnd)   # 세션 안에서는 뒤집기가 일정


if __name__ == "__main__":
    for f in [v for k, v in list(globals().items()) if k.startswith("test_")]:
        f(); print(f"  ✓ {f.__name__}")
    print("전부 통과 ✓")
