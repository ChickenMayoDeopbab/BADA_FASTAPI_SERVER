import argparse
import asyncio
import contextlib
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


async def _recv_loop(ws, *, speech_end: float, timeout: float) -> dict:
    """발화 오디오 송신 끝(speech_end) 기준으로 응답 프레임을 받아 시각을 잰다.

    stt_final_ms: 발화 끝 → 첫 `transcript`(role=user) 프레임.
        서버가 STT FINAL 직후 보내므로 endpoint 지연의 클라이언트 측 관측치.
    client_response_ms: 발화 끝 → 첫 PCM. client_turn_ms: 발화 끝 → speaking_end.
    """
    first_pcm: float | None = None
    turn_end: float | None = None
    stt_final: float | None = None
    transcript: str | None = None
    terminal: str | None = None
    deadline = speech_end + timeout
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
        try:
            frame = json.loads(msg)
        except (TypeError, ValueError):
            continue
        if not isinstance(frame, dict):
            continue
        ftype = frame.get("type")
        if ftype == "transcript" and frame.get("role") == "user":
            if stt_final is None:
                stt_final = time.perf_counter()
                transcript = frame.get("text")
            continue
        if ftype == "speaking_end":
            turn_end = time.perf_counter()
            break
        if ftype in ("end", "error"):
            turn_end = time.perf_counter()
            terminal = ftype
            break

    def _ms(t: float | None) -> float | None:
        return (t - speech_end) * 1000.0 if t else None

    return {
        "stt_final_ms": _ms(stt_final),
        "transcript": transcript,
        "client_response_ms": _ms(first_pcm),
        "client_turn_ms": _ms(turn_end),
        "terminal": terminal,
    }


async def _send_chunks(ws, data: bytes, chunk_bytes: int, chunk_s: float) -> None:
    for i in range(0, len(data), chunk_bytes):
        await ws.send(data[i : i + chunk_bytes])
        await asyncio.sleep(chunk_s)


async def _run_turn(
    ws, audio: bytes, silence: bytes, chunk_bytes: int, chunk_s: float, timeout: float, turn_gap: bytes = b""
) -> dict:
    """발화 1회 스트리밍 → 트레일링 무음을 보내는 동안에도 수신을 계속하며 지연 측정.

    turn_gap: 응답이 끝난 뒤 추가로 보낼 무음(사용자가 뜸 들이는 시간). 긴 세션(STT 스트림 재활용) 재현용.
    """
    try:
        await _send_chunks(ws, audio, chunk_bytes, chunk_s)
    except websockets.ConnectionClosed:
        # 서버가 세션을 끝낸 뒤(예: LLM 이 통화 종료 판단) 다음 턴을 보내면 여기서 끊긴다
        return {
            "stt_final_ms": None, "transcript": None, "client_response_ms": None,
            "client_turn_ms": None, "terminal": "CLOSED",
        }
    speech_end = time.perf_counter()

    recv_task = asyncio.create_task(_recv_loop(ws, speech_end=speech_end, timeout=timeout))
    silence_task = asyncio.create_task(_send_chunks(ws, silence, chunk_bytes, chunk_s))
    try:
        await recv_task
    finally:
        if not silence_task.done():
            silence_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, websockets.ConnectionClosed):
            await silence_task
    result = recv_task.result()
    if turn_gap and result["terminal"] is None:
        with contextlib.suppress(websockets.ConnectionClosed):
            await _send_chunks(ws, turn_gap, chunk_bytes, chunk_s)
    return result


async def _run(args: argparse.Namespace) -> None:
    audios = [(Path(a).name, _load_audio(a)) for a in args.audio]
    silence_bytes = int(_SAMPLE_RATE * _SAMPLE_BYTES * max(args.tail_silence_ms, 0) / 1000)
    silence = b"\x00" * silence_bytes
    turn_gap = b"\x00" * int(_SAMPLE_RATE * _SAMPLE_BYTES * max(args.turn_gap_ms, 0) / 1000)
    chunk_bytes = int(_SAMPLE_RATE * _SAMPLE_BYTES * args.chunk_ms / 1000)
    chunk_s = args.chunk_ms / 1000.0
    uri = _build_uri(args)
    sid = _session_id_of(args)

    total_s = sum(len(a) for _, a in audios) / (_SAMPLE_RATE * _SAMPLE_BYTES)
    print(
        f"session_id={sid} · engine={args.engine or '?'} · 오디오 {len(audios)}개({total_s:.1f}s, 턴마다 순환)"
        f" · 무음 {args.tail_silence_ms}ms · {args.turns}턴\n"
    )

    results: list[dict] = []
    async with websockets.connect(uri, max_size=None, open_timeout=args.timeout) as ws:
        for t in range(args.turns):
            name, audio = audios[t % len(audios)]
            res = await _run_turn(ws, audio, silence, chunk_bytes, chunk_s, args.turn_timeout, turn_gap=turn_gap)
            res.update({"turn": t + 1, "audio": name})
            results.append(res)
            print(
                f"  턴 {t + 1} [{name}]: stt_final={_fmt(res['stt_final_ms'])} "
                f"response={_fmt(res['client_response_ms'])} turn={_fmt(res['client_turn_ms'])} "
                f"({res['terminal'] or 'ok'}) {res['transcript']!r}"
            )
            if res["terminal"] in ("end", "error", "CLOSED"):
                print(f"  세션 종료({res['terminal']}) → 중단")
                break

    rows = [
        ("stt_final_ms", summarize([r["stt_final_ms"] for r in results])),
        ("client_response_ms", summarize([r["client_response_ms"] for r in results])),
        ("client_turn_ms", summarize([r["client_turn_ms"] for r in results])),
    ]
    print()
    if args.json:
        payload = {
            "session_id": sid,
            "engine": args.engine,
            "tail_silence_ms": args.tail_silence_ms,
            "turns": results,
        }
        out = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.json_out:
            Path(args.json_out).write_text(out, encoding="utf-8")
            print(f"json → {args.json_out}")
        else:
            print(out)
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
    parser.add_argument(
        "--audio", required=True, nargs="+",
        help="발화 오디오(.wav 16k/mono/16bit 또는 .pcm/.raw). 여러 개면 턴마다 순환",
    )
    parser.add_argument("--engine", default=None, help="결과에 적을 STT 엔진 라벨(서버 STT_ENGINE 과 맞출 것)")
    parser.add_argument("--json-out", default=None, help="--json 결과를 이 파일에 저장")
    parser.add_argument("--turns", type=int, default=1, help="발화 반복 횟수")
    parser.add_argument("--chunk-ms", type=int, default=100, help="송신 청크 길이(ms)")
    parser.add_argument("--tail-silence-ms", type=int, default=800, help="STT endpointing용 끝 무음(ms)")
    parser.add_argument("--turn-timeout", type=float, default=30.0, help="턴당 응답 대기 한도(s, 발화 끝 기준)")
    parser.add_argument(
        "--turn-gap-ms", type=int, default=0, help="응답 뒤 다음 발화까지 보낼 무음(ms). 긴 세션 재현용"
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="연결 타임아웃(s)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not args.url and not args.session_id:
        parser.error("--url 또는 --session-id 중 하나는 필요합니다.")

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
