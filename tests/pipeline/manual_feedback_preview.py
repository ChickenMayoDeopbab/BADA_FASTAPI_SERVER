"""구간별 피드백이 실제로 어떻게 나가는지 눈으로 보는 스크립트.

마이크도 통화도 필요 없다. 턴별 AVTI 와 발화를 넣고 최종 출력까지 찍는다.

  턴별 AVTI → 구간 선택(GOOD/IMPROVE) → AI 문구 → 검증 → good_segments

실행:
  python -m tests.pipeline.manual_feedback_preview            # 실제 Gemini 호출
  python -m tests.pipeline.manual_feedback_preview --offline  # 미리 준비한 응답으로
  python -m tests.pipeline.manual_feedback_preview --case 2   # 한 케이스만
"""
import argparse
import asyncio
import json

from app.services.feedback_points import is_safe
from app.services.pipeline import VoicePipeline, _segment_fallback

W = 78


class Case:
    def __init__(self, name, scenario, turns, avti_by_turn, clean_spans, canned):
        self.name = name
        self.scenario = scenario
        self.turns = turns          # [(start, end, 발화)]
        self.avti_by_turn = avti_by_turn
        self.clean_spans = clean_spans
        self.canned = canned


CASES = [
    Case(
        name="배달 문의 · 주소 말할 때 목소리가 흔들림",
        scenario={"title": "배달 문의", "callTarget": "중국집",
                  "callPurpose": "안 온 배달 확인하기"},
        turns=[
            (0.5, 5.2, "여보세요, 어제 배달 시킨 게 아직 안 와서 전화드렸어요"),
            (9.0, 16.4, "아 그... 주소가 행복아파트 101동인데요"),
            (21.0, 25.5, "네 그럼 기다릴게요, 감사합니다"),
        ],
        avti_by_turn={1: 2.1, 2: 8.3},
        clean_spans=[(0.8, 5.0), (21.2, 25.3)],
        canned=[
            ("첫마디를 바로 꺼냈어요", "망설이지 않고 상황을 설명해서 상대가 바로 알아들었어요."),
            ("주소를 말할 때 흔들렸어요", "상대방이 되묻고 싶었을 수 있어요. 다음엔 주소를 한 글자씩 천천히 말해봐요."),
            ("마무리 인사를 챙겼어요", "끝인사까지 하고 끊어서 통화가 깔끔하게 끝났어요."),
        ],
    ),
    Case(
        name="병원 예약 · 전 구간 안정적",
        scenario={"title": "병원 예약", "callTarget": "치과 접수처",
                  "callPurpose": "스케일링 예약 잡기"},
        turns=[
            (0.4, 4.8, "안녕하세요, 스케일링 예약하려고 전화드렸습니다"),
            (8.2, 15.0, "박준석이고요, 다음 주 화요일 오후면 좋겠어요"),
            (19.0, 23.0, "네 그때 뵙겠습니다, 감사합니다"),
        ],
        avti_by_turn={1: 1.9, 3: 2.4},
        clean_spans=[(0.6, 4.6), (8.4, 14.8), (19.2, 22.8)],
        canned=[
            ("용건을 먼저 밝혔어요", "무슨 일로 걸었는지 앞에 말해서 상대가 바로 알아들었어요."),
            ("원하는 날짜를 준비해 갔어요", "요일과 시간대까지 정해서 말해 예약이 빨리 잡혔어요."),
            ("차분하게 마무리했어요", "끝까지 흔들림 없이 이야기해서 편안하게 들렸을 거예요."),
        ],
    ),
    Case(
        name="면접 문의 · AVTI 하나도 못 잼",
        scenario={"title": "면접 일정 문의", "callTarget": "회사 인사팀",
                  "callPurpose": "면접 시간 변경 요청"},
        turns=[
            (0.6, 3.2, "여보세요, 어제 면접 안내 문자 받은 지원자인데요"),
            (7.0, 9.1, "박준석입니다"),
            (13.0, 18.6, "혹시 면접 시간을 오후로 바꿀 수 있을까 해서요"),
        ],
        avti_by_turn={},
        clean_spans=[(0.8, 3.0), (13.2, 18.4)],
        canned=[
            ("전화 건 이유를 밝혔어요", "지원자라고 먼저 말해서 상대가 바로 확인할 수 있었어요."),
            ("요청을 분명하게 말했어요", "무엇을 바꾸고 싶은지 정확히 말해서 상대가 되묻지 않았어요."),
        ],
    ),
]


def rule(char: str = "─") -> str:
    return char * W


def _pipeline_for(case: Case) -> VoicePipeline:
    p = VoicePipeline.__new__(VoicePipeline)
    p._user_turn_intervals = [(s, e) for s, e, _ in case.turns]
    p._user_turn_texts = [t for _, _, t in case.turns]
    return p


async def run_case(case: Case, *, offline: bool) -> None:
    print()
    print(rule("━"))
    print(f"  {case.name}")
    print(rule("━"))

    print("\n[1] 턴별 AVTI")
    print(rule())
    for i, (start, end, said) in enumerate(case.turns, start=1):
        avti = case.avti_by_turn.get(i)
        mark = f"{avti:.1f}" if avti is not None else "측정 안 됨"
        print(f"  턴{i}  {start:5.1f}~{end:5.1f}초   AVTI {mark:<10} \"{said}\"")

    p = _pipeline_for(case)
    segments = p._pick_segments(case.clean_spans, case.avti_by_turn)

    print("\n[2] 고른 구간")
    print(rule())
    for seg in segments:
        why = {"voice": "AVTI 로 판단", "clean": "떨림 없는 구간",
               "fallback": "근거 없어 보충"}.get(seg.get("reason"), "-")
        print(f"  {seg['type']:<8} {seg['start']:5.1f}~{seg['end']:5.1f}초   ({why})")

    print(f"\n[3] AI 문구  ({'미리 준비한 응답' if offline else '실제 Gemini 호출'})")
    print(rule())
    items = [
        {"type": seg["type"], "utterance": p._utterance_for(seg["start"], seg["end"]),
         "avti": seg.get("avti"), "reason": seg.get("reason")}
        for seg in segments
    ]
    if offline:
        pairs = (case.canned + [("", "")] * len(items))[: len(items)]
    else:
        from app.services.llm import LLMClient
        pairs = await LLMClient().segment_feedback(
            items,
            scenario_title=case.scenario["title"],
            call_target=case.scenario["callTarget"],
            call_purpose=case.scenario["callPurpose"],
        )

    for seg, (title, content) in zip(segments, pairs, strict=True):
        if not is_safe(title, content):
            print(f"  ✗ 검증 탈락 → 폴백 문구로 대체: {title} | {content}")
            title, content = _segment_fallback(seg["type"])
        seg["title"], seg["content"] = title, content
        seg["good_point"] = content
        seg.pop("reason", None)
        seg.pop("turn", None)
        seg.pop("avti", None)

    print("\n[4] 사용자에게 나가는 구간 피드백")
    print(rule())
    for seg in segments:
        mark = "＋" if seg["type"] == "GOOD" else "－"
        print(f"  {mark} [{seg['type']}]  {seg['start']:.1f}~{seg['end']:.1f}초")
        print(f"     title  : {seg['title']}")
        print(f"     content: {seg['content']}")

    print("\n[5] end 프레임 · Spring 콜백에 실리는 모양")
    print(rule())
    for line in json.dumps(
        {"good_segments": segments}, ensure_ascii=False, indent=2
    ).splitlines():
        print(f"  {line}")


async def main() -> None:
    ap = argparse.ArgumentParser(description="구간 피드백 미리보기")
    ap.add_argument("--offline", action="store_true", help="Gemini 호출 없이")
    ap.add_argument("--case", type=int, help="특정 케이스만 (1부터)")
    args = ap.parse_args()

    cases = CASES if args.case is None else [CASES[args.case - 1]]
    for case in cases:
        await run_case(case, offline=args.offline)
    print()


if __name__ == "__main__":
    asyncio.run(main())
