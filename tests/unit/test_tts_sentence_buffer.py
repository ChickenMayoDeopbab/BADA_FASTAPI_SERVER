import json

from app.services.tts import TTSSession, _SentenceBuffer


def _feed_all(buf: _SentenceBuffer, chunks: list[str]) -> list[str]:
    out = []
    for c in chunks:
        payload = buf.feed(c)
        if payload:
            out.append(payload)
    return out


def test_midword_split_reassembled_into_sentence() -> None:
    buf = _SentenceBuffer()
    assert buf.feed("안녕하세") is None
    assert buf.feed("요, 무엇을 도와") is None
    payload = buf.feed("드릴까요? 오늘")
    assert payload == "안녕하세요, 무엇을 도와드릴까요? "
    assert buf.flush() == "오늘 "


def test_multiple_sentences_flushed_to_last_boundary() -> None:
    buf = _SentenceBuffer()
    payload = buf.feed("네. 알겠습니다. 확인")
    assert payload == "네. 알겠습니다. "
    assert buf.flush() == "확인 "


def test_no_boundary_over_max_chars_splits_at_last_whitespace() -> None:
    buf = _SentenceBuffer(max_chars=20)
    payload = buf.feed("가나다라 마바사아 자차카타 파하가나다라마바")
    assert payload == "가나다라 마바사아 자차카타 "
    assert buf.flush() == "파하가나다라마바 "


def test_no_whitespace_over_max_chars_flushes_whole_buffer() -> None:
    buf = _SentenceBuffer(max_chars=10)
    payload = buf.feed("가나다라마바사아자차카타파하")
    assert payload == "가나다라마바사아자차카타파하 "
    assert buf.flush() is None


def test_flush_emits_remainder_with_trailing_space() -> None:
    buf = _SentenceBuffer()
    assert buf.feed("잔여 텍스트") is None
    assert buf.flush() == "잔여 텍스트 "
    assert buf.flush() is None


def test_empty_and_whitespace_chunks_ignored() -> None:
    buf = _SentenceBuffer()
    assert buf.feed("") is None
    assert buf.feed("   ") is None
    assert buf.flush() is None


def test_closing_quote_after_boundary_stays_attached() -> None:
    buf = _SentenceBuffer()
    payload = buf.feed('"진짜요?" 라고')
    assert payload == '"진짜요?" '
    assert buf.flush() == "라고 "


def test_boundary_at_end_of_chunk() -> None:
    buf = _SentenceBuffer()
    assert buf.feed("안녕하세요?") == "안녕하세요? "
    assert buf.flush() is None


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))


async def test_send_text_sends_sentence_units_then_eos() -> None:
    ws = _FakeWS()
    session = TTSSession(ws, {"stability": 0.5})

    async def source():
        for c in ("안녕하세", "요. 반갑", "", "습니다."):
            yield c

    await session._send_text(source())
    texts = [m["text"] for m in ws.sent]
    assert texts == ["안녕하세요. ", "반갑습니다. ", ""]
