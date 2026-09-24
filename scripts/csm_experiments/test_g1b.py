# -*- coding: utf-8 -*-
"""G1b 스크립트 시험(맥): 가짜 ktel 샤드로 세트 뽑기 규칙·접두어 구성·사람 원본 추출. 토크나이저는 캐시(HF_HUB_OFFLINE=1). G1B_REAL=1 이면 Mimi 디코드까지.
  source labs/env.sh && HF_HUB_OFFLINE=1 python labs/eval/test_g1b.py
"""
import io, json, os, subprocess, sys, tempfile, types, wave, zipfile
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); TRAIN = os.path.join(os.path.dirname(HERE), "train")
import g1b_generate as GG


def run(args, **kw):
    r = subprocess.run([sys.executable] + args, capture_output=True, text=True, env={**os.environ, "HF_HUB_OFFLINE": "1"}, **kw); assert r.returncode == 0, r.stderr; return r.stdout


def main():
    with tempfile.TemporaryDirectory() as d:
        data, out = os.path.join(d, "tok"), os.path.join(d, "set")
        run([os.path.join(TRAIN, "make_fake_tokens.py"), data, "--layout", "ktel", "--shards", "1", "--sessions", "60", "--seed", "1"])
        so = run([os.path.join(HERE, "g1b_pick.py"), "--data", data, "--out", out, "--n", "5", "--no-audio"]); print("  ", so.strip().splitlines()[0])
        items = [json.loads(l) for l in open(os.path.join(out, "set.jsonl"), encoding="utf-8")]; assert 1 <= len(items) <= 5, len(items)
        codes = np.load(os.path.join(out, "codes.npz")); meta = open(os.path.join(out, "meta.lst"), encoding="utf-8").read().splitlines()
        assert len(meta) == len(items) and all(len(m.split("|")) == 5 for m in meta) and meta[0].startswith(items[0]["id"] + "|")
        for it in items:
            t = it["target"]; assert t["tag"] == 0 and 38 <= t["frames"] <= 125 and it["ref"]["frames"] >= 25 and it["infer_text"] == t["spell"]
            f120 = sum(u["frames"] for u in it["ctx120"]); f60 = sum(u["frames"] for u in it["ctx60"])
            assert 750 <= f120 <= 1500 and f60 <= 750 and it["ctx60"] == it["ctx120"][len(it["ctx120"]) - len(it["ctx60"]):]      # ctx60 은 ctx120 의 끝부분
            assert any(u["shard"] == it["ref"]["shard"] and u["idx"] == it["ref"]["idx"] for u in it["ctx120"])                 # 참조 턴은 문맥 안
            for u in it["ctx120"] + [t, it["ref"]]: assert codes[f"{u['shard']}:{u['idx']}"].shape == (32, u["frames"])
        print(f"  ✓ g1b_pick: {len(items)}개 · 규칙(역할·길이·문맥 예산·참조·codes.npz)")
        from transformers import AutoProcessor                                                      # 뽑기와 같은 토크나이저여야 2,048 예산 검사가 뜻이 있다
        tok, cfg = AutoProcessor.from_pretrained("sesame/csm-1b").tokenizer, types.SimpleNamespace(audio_token_id=128002, audio_eos_token_id=128003); it = items[0]
        for ctx, turns in ((120, it["ctx120"]), (60, it["ctx60"]), (0, [])):
            ids, spans = GG.build_prefix(tok, cfg, it, codes, ctx)
            want = sum(len(GG.G.text_ids(tok, u["spell"], u["tag"])) + u["frames"] + 1 for u in turns) + len(GG.G.text_ids(tok, it["infer_text"], it["tag"]))
            assert len(ids) == want and len(spans) == len(turns) and all(ids[s + c.shape[0]] == cfg.audio_eos_token_id and ids[s] == cfg.audio_token_id for s, c in spans)
            assert len(ids) + 250 <= 2048
        print("  ✓ g1b_generate.build_prefix: 길이·spans·eos 자리 · 접두어 + 250 ≤ 2048")
        zp = os.path.join(d, "valid.zip"); it = items[0]
        with zipfile.ZipFile(zp, "w") as z:
            for jt in items:
                for u in jt["target"]["utts"]:
                    b = io.BytesIO()
                    with wave.open(b, "wb") as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000); w.writeframes(np.ones(8000, np.int16).tobytes())
                    z.writestr(f"{jt['session']}/{u}.wav", b.getvalue())
        run([os.path.join(HERE, "g1b_human.py"), "--set", out, "--wav", zp])
        with wave.open(os.path.join(out, "human", it["id"] + ".wav")) as w: assert w.getframerate() == 8000 and w.getnframes() == 8000 * len(it["target"]["utts"]) + 2400 * (len(it["target"]["utts"]) - 1)
        print("  ✓ g1b_human: 원본 이어 붙이기(0.3 s 무음)")
        if os.environ.get("G1B_REAL"):
            run([os.path.join(HERE, "g1b_pick.py"), "--data", data, "--out", out + "_real", "--n", "2", "--device", "cpu"])
            r0 = json.loads(open(os.path.join(out + "_real", "set.jsonl"), encoding="utf-8").readline())
            with wave.open(os.path.join(out + "_real", "mimi", r0["id"] + ".wav")) as w: assert w.getframerate() == 24000 and abs(w.getnframes() - r0["target"]["frames"] * 1920) <= 1920
            with wave.open(os.path.join(out + "_real", "prompt", r0["id"] + ".wav")) as w: assert abs(w.getnframes() - r0["ref"]["frames"] * 1920) <= 1920
            print("  ✓ Mimi 디코드 wav(24 kHz, 길이 = 프레임 × 1920)")
    print("전부 통과 ✓")


if __name__ == "__main__":
    main()
