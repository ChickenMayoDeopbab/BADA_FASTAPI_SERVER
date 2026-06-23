import argparse
import asyncio
import json
import time
import wave
from pathlib import Path
from urllib.parse import urlencode

import websockets

from tests.perf._stats import print_table, summarize

_SAMPLE_RATE = 16000
_SAMPLE_BYTES = 2


def _load_audio(path: str) -> bytes:
    """16kHz/mono/int16 PCM 바이트 로드. wav는 포맷 검증, 그 외는 원시 바이트."""
    p = Path(path)
    if p.suffix.lower() == ".wav":
        with wave.open(str(p), "rb") as wf:
            if wf.getframerate() != _SAMPLE_RATE or wf.getnchannels() != 1 or wf.getsampwidth() != _SAMPLE_BYTES:
                raise SystemExit(
                    f"wav 포맷 불일치: {wf.getframerate()}Hz/{wf.getnchannels()}ch/"
                    f"{wf.getsampwidth() * 8}bit → 16000Hz/mono/16bit 필요 (사전 변환하세요)."
                )
            return wf.readframes(wf.getnframes())
    return p.read_bytes()


def _build_uri(args: argparse.Namespace) -> str:
    base = args.url or f"{args.base_url.rstrip('/')}/ws/voice/{args.session_id}"
    return f"{base}?{urlencode({'token': args.token})}"


def _session_id_of(args: argparse.Namespace) -> str:
    if args.session_id:
        return args.session_id
    return args.url.rstrip("/").split("/")[-1] if args.url else "(unknown)"


async def _run_turn(ws, audio: bytes, chunk_bytes: int, chunk_s: float, timeout: float) -> dict:
    """오디오 1회 스트리밍 → 첫 PCM/ speaking_end 까지 지연 측정."""
    for i in range(0, len(audio), chunk_bytes):
        await ws.send(audio[i : i + chunk_bytes])
        await asyncio.sleep(chunk_s)
    audio_sent = time.perf_counter()

    first_pcm: float | None = None
    turn_end: float | None = None
    terminal: str | None = None
    deadline = audio_sent + timeout
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            terminal = "TIMEOUT"
            break
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=remaining)
        except TimeoutError:
            terminal = "TIMEOUT"
            break
        except websockets.ConnectionClosed:
            terminal = "CLOSED"
            break

        if isinstance(msg, bytes):
            if first_pcm is None:
                first_pcm = time.perf_counter()
            continue
        frame = json.loads(msg)
        ftype = frame.get("type")
        if ftype == "speaking_end":
            turn_end = time.perf_counter()
            break
        if ftype in ("end", "error"):
            turn_end = time.perf_counter()
            terminal = ftype
            break

    return {
        "client_response_ms": (first_pcm - audio_sent) * 1000.0 if first_pcm else None,
        "client_turn_ms": (turn_end - audio_sent) * 1000.0 if turn_end else None,
        "terminal": terminal,
    }


async def _run(args: argparse.Namespace) -> None:
    audio = _load_audio(args.audio)
    if args.tail_silence_ms > 0:
        audio += b"\x00" * int(_SAMPLE_RATE * _SAMPLE_BYTES * args.tail_silence_ms / 1000)
    chunk_bytes = int(_SAMPLE_RATE * _SAMPLE_BYTES * args.chunk_ms / 1000)
    chunk_s = args.chunk_ms / 1000.0
    uri = _build_uri(args)
    sid = _session_id_of(args)

    audio_s = len(audio) / (_SAMPLE_RATE * _SAMPLE_BYTES)
    print(f"session_id={sid} · 오디오 {audio_s:.1f}s(+무음 {args.tail_silence_ms}ms) · {args.turns}턴\n")

    results: list[dict] = []
    async with websockets.connect(uri, max_size=None, open_timeout=args.timeout) as ws:
        for t in range(args.turns):
            res = await _run_turn(ws, audio, chunk_bytes, chunk_s, args.turn_timeout)
            results.append(res)
            print(
                f"  턴 {t + 1}: response={_fmt(res['client_response_ms'])} "
                f"turn={_fmt(res['client_turn_ms'])} ({res['terminal'] or 'ok'})"
            )
            if res["terminal"] in ("end", "error", "CLOSED"):
                print(f"  세션 종료({res['terminal']}) → 중단")
                break

    rows = [
        ("client_response_ms", summarize([r["client_response_ms"] for r in results])),
        ("client_turn_ms", summarize([r["client_turn_ms"] for r in results])),
    ]
    print()
    if args.json:
        print(json.dumps({"session_id": sid, "turns": results}, ensure_ascii=False, indent=2))
    else:
        print_table(rows)
        print(f"\n단계별(STT/LLM/TTS) 분해는 서버 로그에서: grep voice_turn + session_id={sid}")


def _fmt(v: float | None) -> str:
    return f"{v:.1f}ms" if v is not None else "—"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WS 음성 통화 지연 벤치마크",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--url", default=None, help="완전한 ws URL(.../ws/voice/<SID>). base-url/session-id 대신")
    parser.add_argument("--base-url", default="ws://localhost:8000")
    parser.add_argument("--session-id", default=None, help="redis에 존재하는 세션 ID")
    parser.add_argument("--token", default="", help="JWT access token")
    parser.add_argument("--audio", required=True, help="발화 오디오(.wav 16k/mono/16bit 또는 .pcm/.raw)")
    parser.add_argument("--turns", type=int, default=1, help="발화 반복 횟수")
    parser.add_argument("--chunk-ms", type=int, default=100, help="송신 청크 길이(ms)")
    parser.add_argument("--tail-silence-ms", type=int, default=800, help="STT endpointing용 끝 무음(ms)")
    parser.add_argument("--turn-timeout", type=float, default=30.0, help="턴당 응답 대기 한도(s)")
    parser.add_argument("--timeout", type=float, default=10.0, help="연결 타임아웃(s)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not args.url and not args.session_id:
        parser.error("--url 또는 --session-id 중 하나는 필요합니다.")

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
