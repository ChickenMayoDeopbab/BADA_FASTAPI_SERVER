from app.services.session import (
    _HISTORY_DROP_CHUNK,
    _HISTORY_MAX_MESSAGES,
    _cap_history,
    build_turn_context,
)


def _history(n: int) -> list[dict]:
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "text": f"m{i}"}
        for i in range(n)
    ]


def test_under_max_is_unchanged() -> None:
    h = _history(_HISTORY_MAX_MESSAGES)
    assert _cap_history(h) == h


def test_over_max_is_capped_and_keeps_latest() -> None:
    h = _history(_HISTORY_MAX_MESSAGES + 1)
    kept = _cap_history(h)
    assert len(kept) <= _HISTORY_MAX_MESSAGES
    assert kept[-1] == h[-1]  # 최신 메시지 보존


def test_cut_point_is_stable_within_chunk() -> None:
    a = _cap_history(_history(_HISTORY_MAX_MESSAGES + 1))
    b = _cap_history(_history(_HISTORY_MAX_MESSAGES + _HISTORY_DROP_CHUNK))
    assert a[0] == b[0]


def test_cut_point_advances_by_chunk() -> None:
    a = _cap_history(_history(_HISTORY_MAX_MESSAGES + 1))
    c = _cap_history(_history(_HISTORY_MAX_MESSAGES + _HISTORY_DROP_CHUNK + 1))
    idx_a = int(a[0]["text"][1:])
    idx_c = int(c[0]["text"][1:])
    assert idx_c - idx_a == _HISTORY_DROP_CHUNK


def test_kept_history_starts_with_user() -> None:
    h = [{"role": "assistant", "text": "a0"}] + _history(_HISTORY_MAX_MESSAGES + 1)
    kept = _cap_history(h)
    assert kept[0]["role"] == "user"


def test_build_turn_context_applies_cap() -> None:
    ctx = build_turn_context(
        {},
        current_step=1,
        history=_history(100),
        user_utterance="여보세요",
    )
    assert len(ctx.history) <= _HISTORY_MAX_MESSAGES
    assert ctx.history[-1]["text"] == "m99"
