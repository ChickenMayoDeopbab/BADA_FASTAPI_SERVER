from enum import StrEnum

from app.schemas.llm import AiEmotion


class FrameType(StrEnum):
    EMOTION = "emotion"               # 아바타/음성 전환 트리거
    SPEAKING_END = "speaking_end"     # 이번 턴 오디오 끝
    INTERRUPT = "interrupt"           # 클라 재생 버퍼 비우기
    END = "end"                       # 세션 종료
    ERROR = "error"                   # 복구 불가 에러


class EndReason(StrEnum):
    SCENARIO_DONE = "SCENARIO_DONE"   # 시나리오 마지막 step 종료
    USER_END = "USER_END"             # 클라가 end 프레임 전송
    TIMEOUT = "TIMEOUT"               # 최대 시간 초과
    CRISIS = "CRISIS"                 # 위기 신호로 정중히 종료
    END_CALL = "END_CALL"             # LLM이 끝났다고 신호
    NO_AUDIO = "NO_AUDIO"             # STT가 오디오 미수신으로 스트림 종료(정상)
    ERROR = "ERROR"                   # 서버 에러로 종료


def emotion_frame(emotion: AiEmotion) -> dict:
    return {"type": FrameType.EMOTION.value, "value": emotion.value}


def speaking_end_frame() -> dict:
    return {"type": FrameType.SPEAKING_END.value}


def interrupt_frame() -> dict:
    return {"type": FrameType.INTERRUPT.value}


def end_frame(reason: EndReason, feedback: dict | None = None) -> dict:
    frame =  {"type": FrameType.END.value, "reason": reason.value}
    if feedback is not None:
        frame["feedback"] = feedback
    return frame


def error_frame(code: str) -> dict:
    return {"type": FrameType.ERROR.value, "code": code}
