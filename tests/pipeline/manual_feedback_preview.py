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
    Case(
        name="환불 요청 · 처음부터 끝까지 흔들림",
        scenario={"title": "환불 요청", "callTarget": "쇼핑몰 고객센터",
                  "callPurpose": "잘못 온 물건 환불받기"},
        turns=[
            (1.0, 6.5, "저기... 어제 받은 물건이 주문한 거랑 다른데요"),
            (11.0, 18.0, "그... 환불이 되는지 여쭤보려고요"),
            (23.0, 29.5, "아 네... 그럼 어떻게 하면 되나요"),
        ],
        avti_by_turn={1: 6.8, 2: 9.1, 3: 7.4},
        clean_spans=[],
        canned=[
            ("첫마디를 꺼내기 어려웠어요", "상대방이 잘 듣지 못했을 것 같아요. 다음엔 한 호흡 쉬고 시작해봐요."),
            ("환불이라는 말을 흐렸어요", "말끝이 흔들려서 상대방이 되물었을 수 있어요. 다음엔 문장을 끝까지 말해봐요."),
            ("되묻는 말이 작아졌어요", "상대방이 잘 듣지 못했을 것 같아요. 다음엔 또박또박 물어봐요."),
        ],
    ),
    Case(
        name="관공서 문의 · 발화 인식 안 된 구간 섞임",
        scenario={"title": "서류 문의", "callTarget": "주민센터",
                  "callPurpose": "등본 떼는 방법 물어보기"},
        turns=[
            (0.5, 4.0, ""),
            (8.0, 14.2, "등본을 떼려면 뭘 가져가야 하는지 여쭤보려고요"),
            (18.0, 22.0, ""),
        ],
        avti_by_turn={1: 7.9, 2: 2.2},
        clean_spans=[(18.2, 21.8)],
        canned=[
            ("첫마디가 잘 안 들렸어요", "다음엔 조금 더 큰 목소리로 시작해봐요."),
            ("무엇이 필요한지 물었어요", "필요한 것을 콕 집어 물어서 상대방이 바로 답할 수 있었어요."),
            ("끝까지 차분했어요", "마지막까지 흔들림 없이 말했어요."),
        ],
    ),
    Case(
        name="미용실 예약 · 단답만 하고 끊음",
        scenario={"title": "미용실 예약", "callTarget": "미용실",
                  "callPurpose": "커트 예약 잡기"},
        turns=[
            (0.6, 1.4, "네"),
            (5.0, 6.2, "음... 아니요"),
            (10.0, 11.1, "네 알겠습니다"),
        ],
        avti_by_turn={2: 8.0},
        clean_spans=[(0.6, 1.4), (10.0, 11.1)],
        canned=[
            ("짧게라도 대답했어요", "상대방 말에 바로 반응해서 통화가 끊기지 않았어요."),
            ("대답이 흐려졌어요", "상대방이 잘 듣지 못했을 것 같아요. 다음엔 원하는 것을 한 문장으로 말해봐요."),
            ("마무리 대답을 했어요", "알겠다고 답해서 상대방이 안심하고 통화를 끝냈어요."),
        ],
    ),
    Case(
        name="택배 분실 · 속상한 상황",
        scenario={"title": "택배 분실 문의", "callTarget": "택배 대리점",
                  "callPurpose": "배송 완료로 뜨는데 없는 물건 찾기"},
        turns=[
            (0.8, 7.5, "배송 완료라고 뜨는데 물건이 없어서요, 며칠째 이러니까 답답해요"),
            (12.0, 19.0, "그럼 제가 또 기다려야 되는 건가요"),
            (24.0, 28.0, "네 그럼 연락 주세요"),
        ],
        avti_by_turn={1: 2.5, 2: 8.7, 3: 2.0},
        clean_spans=[(1.0, 7.3), (24.2, 27.8)],
        canned=[
            ("상황을 순서대로 설명했어요", "무슨 일이 있었는지 차근차근 말해서 상대방이 바로 확인에 들어갔어요."),
            ("따져 묻는 말이 흔들렸어요", "상대방이 잘 듣지 못했을 것 같아요. 다음엔 한 호흡 쉬고 물어봐요."),
            ("연락 방법을 정하고 끊었어요", "다음에 어떻게 할지 정해두고 끊어서 통화가 깔끔했어요."),
        ],
    ),
    Case(
        name="알바 지원 · 구간이 하나뿐",
        scenario={"title": "아르바이트 지원", "callTarget": "카페",
                  "callPurpose": "구인 공고 보고 지원 문의하기"},
        turns=[
            (2.0, 9.4, "공고 보고 연락드렸는데 아직 사람 구하시나요"),
        ],
        avti_by_turn={1: 1.8},
        clean_spans=[(2.2, 9.2)],
        canned=[
            ("공고를 보고 왔다고 밝혔어요", "어디서 봤는지 먼저 말해서 상대방이 바로 알아들었어요."),
        ],
    ),
    Case(
        name="시나리오 정보 없음 · 일반 통화",
        scenario={"title": "", "callTarget": "", "callPurpose": ""},
        turns=[
            (0.5, 5.0, "여보세요, 잠깐 통화 괜찮으세요"),
            (9.0, 15.5, "다름이 아니라 내일 약속 시간을 좀 늦출 수 있을까 해서요"),
        ],
        avti_by_turn={2: 6.2},
        clean_spans=[(0.7, 4.8)],
        canned=[
            ("통화 되는지 먼저 물었어요", "상대방 사정을 먼저 물어봐서 이야기를 꺼내기 편해졌어요."),
            ("부탁하는 말이 흔들렸어요", "상대방이 잘 듣지 못했을 것 같아요. 다음엔 부탁을 한 문장으로 말해봐요."),
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
