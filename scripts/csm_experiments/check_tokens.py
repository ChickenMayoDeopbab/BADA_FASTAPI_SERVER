# -*- coding: utf-8 -*-
"""tok_kspon.py 출력의 무결성 검사. 토큰화가 끝난 뒤(또는 프로세스가 겹쳐 돈 사고 뒤) 한 번 돌린다.
  python check_tokens.py ~/tok/kspon
보는 것: offsets 끝 = 코드 길이 · 매니페스트 줄 수 = 발화 수 · 코드 범위 0~2047 · codes/manifest 짝 · 남은 임시 파일.
이상 묶음은 그 묶음의 .npz 와 .jsonl 을 지우고 토큰화 명령을 다시 돌리면 그 묶음만 다시 만든다.
"""
import glob, os, sys
import numpy as np

root = os.path.expanduser(sys.argv[1]); bad = []; n = frames = 0
codes = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(root, "codes", "*.npz")) if not p.endswith(".tmp.npz")}
mans = {os.path.basename(p)[:-6] for p in glob.glob(os.path.join(root, "manifest", "*.jsonl"))}
for name in sorted(codes | mans):
    if name not in codes or name not in mans:
        bad.append((name, "codes/manifest 중 한쪽만 있다")); continue
    try:
        z = np.load(os.path.join(root, "codes", name + ".npz")); c, off = z["codes"], z["offsets"]
        lines = sum(1 for _ in open(os.path.join(root, "manifest", name + ".jsonl"), encoding="utf-8"))
        why = ("offsets 끝 ≠ 코드 길이" if off[-1] != c.shape[1] else "매니페스트 줄 수 ≠ 발화 수" if lines != len(off) - 1 else
               "코드북이 32개가 아니다" if c.shape[0] != 32 else "코드 범위 밖" if c.min() < 0 or c.max() > 2047 else "")
        if why: bad.append((name, why))
        n += len(off) - 1; frames += c.shape[1]
    except Exception as e:                                       # 쓰다 만 파일은 여기서 걸린다
        bad.append((name, f"열 수 없다: {type(e).__name__}"))
tmp = glob.glob(os.path.join(root, "**", "*.tmp*"), recursive=True)
print(f"묶음 {len(codes)} · 발화 {n:,} · {frames / 12.5 / 3600:.1f} h · 이상 {len(bad)} · 남은 임시 파일 {len(tmp)}")
for name, why in bad: print(f"  이상: {name} — {why}")
for p in tmp: print(f"  임시: {p}")
sys.exit(1 if bad or tmp else 0)
