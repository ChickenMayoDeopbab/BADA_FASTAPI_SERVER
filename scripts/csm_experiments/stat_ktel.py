# -*- coding: utf-8 -*-
"""tok_ktel.py 출력의 분포. B단계 로더 설정(문맥 예산·턴 길이 상한)을 정할 때 본다.
  python stat_ktel.py ~/tok/ktel [문맥 예산 초, 기본 90]
"""
import collections, glob, json, os, sys
import numpy as np

root = os.path.expanduser(sys.argv[1]); budget = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0
rows = [json.loads(l) for p in sorted(glob.glob(os.path.join(root, "manifest", "*.jsonl"))) for l in open(p, encoding="utf-8")]
sess = collections.defaultdict(list)
for r in rows: sess[r["session"]].append(r)
q = lambda xs: " / ".join(f"{np.percentile(xs, p):.1f}" for p in (50, 90, 99)) + f" / 최대 {max(xs):.1f}"
dur = [r["dur_s"] for r in rows]; utt = [u for r in rows for u in r["utt_s"]]
print(f"세션 {len(sess):,} · 턴 {len(rows):,} · 발화 {sum(r['n_utt'] for r in rows):,} · {sum(dur)/3600:.2f} h(무음 포함) · 글 없는 턴 {sum(not r['has_text'] for r in rows)}")
print(f"세션당 턴 수    p50/p90/p99 {q([len(v) for v in sess.values()])}")
print(f"턴당 발화 수    p50/p90/p99 {q([r['n_utt'] for r in rows])} · 2개 이상 {sum(r['n_utt'] > 1 for r in rows) / len(rows):.0%}")
print(f"턴 길이(초)     p50/p90/p99 {q(dur)}")
print(f"발화 길이(초)   p50/p90/p99 {q(utt)}")
for role in sorted({r["role"] for r in rows}, key=str):
    rr = [r for r in rows if r["role"] == role]; print(f"  {role}: 턴 {len(rr):,} · {sum(r['dur_s'] for r in rr)/3600:.2f} h · 턴 길이 p50 {np.percentile([r['dur_s'] for r in rr], 50):.1f} s")
ctx = []                                                            # 목표 턴마다 직전 턴을 예산 안까지 붙이면 몇 턴이 들어가나
for v in sess.values():
    for i in range(1, len(v)):
        s = n = 0
        for r in reversed(v[:i]):
            if s + r["dur_s"] > budget: break
            s += r["dur_s"]; n += 1
        ctx.append(n)
if ctx: print(f"문맥 예산 {budget:.0f} s 안에 드는 직전 턴 수 p50/p90/p99 {q(ctx)} · 0턴(직전 턴이 예산보다 김) {sum(c == 0 for c in ctx)}")
fl = collections.Counter(); [fl.update({k: v for k, v in r["flags"].items() if v}) for r in rows]
print(f"전사 표기 합계 {dict(fl)}")
print(f"역할 조합이 (고객, 상담원) 이 아닌 세션 {sum(sorted(map(str, {r['role'] for r in v})) != ['고객', '상담원'] for v in sess.values())}")
