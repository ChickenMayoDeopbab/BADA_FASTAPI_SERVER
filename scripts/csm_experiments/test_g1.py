# -*- coding: utf-8 -*-
"""g1_score / g1_pick 의 순수 함수 시험. 모델·네트워크 없이 돈다:  python test_g1.py"""
import json, math, os, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1_score as S
import g1_pick as P

ok = True
def check(name, cond, extra=""):
    global ok; ok &= bool(cond); print(f"  {'✓' if cond else '✗'} {name} {extra}")


print("[정규화]")
check("문장부호·태그 제거", S.normalize("[숨] 네, 그래요?  [웃음] 좋아요!") == "네 그래요 좋아요")
check("NFC (자모 분리형 → 완성형)", S.normalize("가") == "가")
check("영문 소문자화", S.normalize("OK 좋아") == "ok 좋아")
check("물결·말줄임표 제거", S.normalize("음… 그건~ 좀") == "음 그건 좀")

print("[편집거리 — 손으로 센 값]")
check("같음 = 0", S.edits(list("가나다"), list("가나다")) == 0)
check("치환 1", S.edits(list("가나다"), list("가너다")) == 1)
check("삭제 1", S.edits(list("가나다"), list("가다")) == 1)
check("삽입 2", S.edits(list("가나"), list("가나다라")) == 2)
check("빈 가설 = 참조 길이", S.edits(list("가나다"), []) == 3)

print("[발화별 WER/CER — seed-tts-eval 식: (S+D+I)/N]")
r = S.error_rates("오늘 날씨 어때요?", "오늘 날씨가 어때요")
check("WER 1/3", math.isclose(r["wer"], 1 / 3), f"({r['wer']:.4f})")
check("CER 1/7 (띄어쓰기 제거, 음절 단위)", math.isclose(r["cer"], 1 / 7), f"({r['cer']:.4f})")
r = S.error_rates("네 알겠습니다", "")
check("빈 전사 → 1.0", r["wer"] == 1.0 and r["cer"] == 1.0)
r = S.error_rates("네", "네 네 네 네")
check("삽입이 많으면 1 을 넘는다(자르지 않는다)", r["wer"] == 3.0)

print("[집계]")
rows = [dict(cer=0.0, wer=0.0, n_char=10, e_char=0, n_word=3, e_word=0), dict(cer=0.5, wer=1.0, n_char=2, e_char=1, n_word=1, e_word=1)]
a = S.aggregate(rows)
check("발화별 산술평균(seed-tts-eval) CER 25.0", math.isclose(a["cer_mean"], 25.0))
check("전체 합산 CER 1/12", math.isclose(a["cer_corpus"], 100 / 12))
check("50% 초과 발화 수(CER 기준 '초과' → 0.5 는 아님)", a["n_over50"] == 0)
lo, hi = S.bootstrap_ci([0.0, 0.5] * 50, seed=0)
lo2, hi2 = S.bootstrap_ci([0.0, 0.5] * 50, seed=0)
check("부트스트랩 구간 재현 가능 · 평균을 포함", (lo, hi) == (lo2, hi2) and lo < 25.0 < hi, f"[{lo:.1f}, {hi:.1f}]")

print("[리샘플 24k → 16k]")
t = np.arange(24000) / 24000
y = S.resample(np.sin(2 * np.pi * 1000 * t).astype(np.float32), 24000, 16000)
check("길이 = 16000", y.shape[0] == 16000, f"({y.shape[0]})")
mid = y[2000:14000]; check("1 kHz 진폭 보존", abs(np.sqrt(2 * np.mean(mid ** 2)) - 1) < 0.01, f"({np.sqrt(2*np.mean(mid**2)):.4f})")
y = S.resample(np.sin(2 * np.pi * 10000 * t).astype(np.float32), 24000, 16000)
check("10 kHz(새 나이퀴스트 밖)는 -60 dB 아래", 20 * np.log10(np.sqrt(np.mean(y[2000:14000] ** 2)) + 1e-12) < -60)
check("16k → 16k 는 그대로", S.resample(np.ones(10, dtype=np.float32), 16000, 16000).shape[0] == 10)

print("[캐시]")
with tempfile.TemporaryDirectory() as d:
    c = S.Cache(os.path.join(d, "c.jsonl")); calls = []
    f = lambda: calls.append(1) or "전사"
    check("처음엔 계산", c.get_or("judge:fake", "abc", f) == "전사" and len(calls) == 1)
    check("둘째엔 캐시", c.get_or("judge:fake", "abc", f) == "전사" and len(calls) == 1)
    c2 = S.Cache(os.path.join(d, "c.jsonl"))
    check("파일에서 다시 읽힌다", c2.get_or("judge:fake", "abc", f) == "전사" and len(calls) == 1)
    check("지표 이름이 다르면 따로", c2.get_or("judge:other", "abc", f) == "전사" and len(calls) == 2)

print("[뽑기: 거르기와 짝짓기]")
def row(i, text, dur=5.0, **fl):
    flags = dict(breath=0, laugh=0, overlap=0, noise=0, unknown=0, filler=0, repeat=0, unclear=0, dual=0); flags.update(fl)
    return dict(id=f"U{i:04d}", shard="eval_clean", idx=i, frames=int(dur * 12.5), dur_s=dur, raw=text, spell=text, pron=text, flags=flags, has_text=True)
good = [row(i, f"이것은 시험용 문장 번호 {'가나다라마바사아자차'[i % 10]} 입니다.") for i in range(30)]
bad = [row(100, "숫자 3개가 있어요 정말로요."), row(101, "영어 OK 가 들어간 문장이에요."), row(102, "겹친 말이 있는 문장입니다 네.", overlap=1),
       row(103, "너무 짧다."), row(104, "길이가 너무 긴 발화입니다 정말.", dur=12.0), row(105, "숨소리가 있는 문장입니다 네네.", breath=1), row(106, "간투어가 있는 문장입니다 어.", filler=1)]
cand = [r for r in good + bad if P.usable(r)]
check("나쁜 7종을 전부 거른다", {r["id"] for r in cand} == {r["id"] for r in good}, f"(남은 {len(cand)})")
items = P.pair(cand, n=10, seed=0); items2 = P.pair(cand, n=10, seed=0)
check("짝 10개 · 재현 가능", len(items) == 10 and items == items2)
ids = [x["target"]["id"] for x in items] + [x["prompt"]["id"] for x in items]
check("프롬프트와 목표가 겹치지 않는다", len(set(ids)) == 20)
check("meta.lst 한 줄 = seed-tts-eval 5칸", P.meta_line(items[0]).count("|") == 4 and P.meta_line(items[0]).split("|")[2] == f"prompt/{items[0]['id']}.wav")
try:
    P.pair(cand, n=16, seed=0); check("후보가 모자라면 에러", False)
except SystemExit:
    check("후보가 모자라면 에러", True)

print("\n전부 통과 ✓" if ok else "\n실패한 항목이 있다 ✗"); sys.exit(0 if ok else 1)
