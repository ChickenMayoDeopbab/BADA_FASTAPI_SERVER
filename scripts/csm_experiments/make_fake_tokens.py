# -*- coding: utf-8 -*-
"""GPU 경로 시험용 **가짜 토큰 셋** — 진짜 토큰이 오기 전에 `train_a.py` 의 bf16 autocast · 8-bit AdamW · 전체 FT 메모리 · 처리량을 서버에서 본다.
코드는 난수라 **loss 값은 뜻이 없다**(내려가지 않아도 정상). 보는 것은 GB · 위치/s · 에러 없이 20 업데이트 + 저장이 되는가.
tok_kspon.py 출력과 같은 모양: codes/<묶음>.npz(codes int16 [32,ΣT] · offsets · ids) + manifest/<묶음>.jsonl. 마지막 묶음은 dev 이름(KsponSpeech_0621).

  python make_fake_tokens.py ~/CSM/data/fake            # 묶음 3 + dev 1, 묶음당 300발화 (≈ 20 업데이트 × 170발화)
"""
import argparse, json, os, random
import numpy as np

SENT = ["여보세요, 저기 두 시에 예약을 했는데요.", "네 알겠습니다. 잠시만 기다려 주세요.", "그게 아니라 제 말은 어제 주문한 게 아직 안 왔다는 거예요.",
        "혹시 배송 조회는 어디서 할 수 있나요?", "아 그럼 다음 주 화요일 오후는 괜찮으세요?", "죄송한데 다시 한 번만 말씀해 주시겠어요?",
        "네네 그렇게 해 주시면 감사하겠습니다.", "어 그 병원 진료 시간이 몇 시까지예요?", "음 그러면 취소하고 새로 예약할게요.",
        "제가 지금 밖이라서 잘 안 들려요.", "아니요 괜찮아요. 그냥 확인만 하려고 전화했어요.", "네 그 주소 맞아요. 삼층 삼백이호요."]


def main():
    ap = argparse.ArgumentParser(description="train_a.py 시험용 가짜 토큰")
    ap.add_argument("out"); ap.add_argument("--shards", type=int, default=3); ap.add_argument("--per-shard", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rng, nr = random.Random(a.seed), np.random.default_rng(a.seed)
    os.makedirs(os.path.join(a.out, "codes"), exist_ok=True); os.makedirs(os.path.join(a.out, "manifest"), exist_ok=True)
    names = [f"KsponSpeech_{i:04d}" for i in range(1, a.shards + 1)] + ["KsponSpeech_0621"]
    for name in names:
        codes, offsets, rows = [], [0], []
        for i in range(a.per_shard):
            T = rng.randint(38, 125)                                         # 3~10 초
            c = nr.integers(0, 2048, (32, T), dtype=np.int16); c[:, 0] = np.maximum(c[:, 0], 1)   # 전부 0 인 프레임(EOS 모양)은 만들지 않는다
            s = rng.choice(SENT); uid = f"{name}_{i:04d}"
            codes.append(c); offsets.append(offsets[-1] + T)
            rows.append(dict(id=uid, shard=name, idx=i, frames=T, dur_s=round(T / 12.5, 3), raw=s, spell=s, pron=s,
                             flags=dict(breath=0, laugh=0, overlap=0, noise=0, unknown=0, filler=0, repeat=0, unclear=0, dual=0), has_text=True))
        np.savez(os.path.join(a.out, "codes", name + ".npz"), codes=np.concatenate(codes, 1), offsets=np.asarray(offsets, dtype=np.int64), ids=np.asarray([r["id"] for r in rows]))
        with open(os.path.join(a.out, "manifest", name + ".jsonl"), "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(dict(fake=True, codebooks=32, frame_hz=12.5), open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(f"가짜 토큰 {len(names)}묶음 × {a.per_shard}발화 (마지막은 dev) → {a.out}")


if __name__ == "__main__":
    main()
