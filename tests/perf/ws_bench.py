import argparse
import asyncio
import contextlib
import json
import math
import time
import wave
from collections.abc import Callable
from pathlib import Path

import websockets

from tests.perf._stats import print_table, summarize

_SAMPLE_RATE = 16000
_SAMPLE_BYTES = 2


def _load_audio(path: str) -> bytes:
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


class _NoiseLoop:
    """잡음만 클립을 이어 붙여 원하는 길이만큼 꺼낸다(없으면 디지털 무음)"""

    def __init__(self, pcm: bytes) -> None:
        self._pcm = pcm[: len(pcm) // _SAMPLE_BYTES * _SAMPLE_BYTES]
        self._pos = 0

    def take(self, n: int) -> bytes:
        if not self._pcm:
            return b"\x00" * n
        out = bytearray()
        while len(out) < n:
            piece = self._pcm[self._pos : self._pos + n - len(out)]
            out += piece
            self._pos = (self._pos + len(piece)) % len(self._pcm)
        return bytes(out)


def _noise_turn(clip: bytes, utt_end_s: float) -> tuple[bytes, bytes]:
    """잡음 클립을 발화 끝에서 잘라 (앞 잡음+발화, 클립 꼬리)"""
    cut = round(utt_end_s * _SAMPLE_RATE) * _SAMPLE_BYTES
    return clip[:cut], clip[cut:]


class _NoiseCondition:
    """mix_noise manifest 의 한 조건: 발화를 그 조건의 클립으로, 꼬리는 클립 꼬리 뒤 잡음만 클립을 응답까지"""

    def __init__(self, manifest_path: str, noise: str, snr_db: float | None) -> None:
        base = Path(manifest_path).parent
        meta = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

        def same(c: dict) -> bool:
            if c["noise"] != noise:
                return False
            return noise == "clean" or (c["snr_db"] is not None and math.isclose(c["snr_db"], snr_db))

        picked = sorted((c for c in meta["clips"] if same(c)), key=lambda c: c.get("variant") or 0)
        self.noise = noise
        self.snr_db = None if noise == "clean" else snr_db
        self.lead_ms = int(meta.get("lead_ms", 0))
        self._clips: dict[str, list[tuple[int | None, bytes, float]]] = {}
        for c in picked:
            if c["noise_only"]:
                continue
            per = self._clips.setdefault(c["utt"], [])
            if any(v == c.get("variant") for v, _, _ in per):
                raise SystemExit(f"manifest 에 같은 조건·변형의 발화가 둘 이상: {c['file']}")
            per.append((c.get("variant"), _load_audio(str(base / c["file"])), float(c["utt_end_s"])))
        if not self._clips:
            raise SystemExit(f"manifest 에 {noise}/{snr_db} 조건 발화가 없다: {manifest_path}")
        beds = [c for c in picked if c["noise_only"]]
        self.bed = _NoiseLoop(b"".join(_load_audio(str(base / c["file"])) for c in beds))

    def turn(self, name: str, cycle: int = 0) -> tuple[int | None, bytes, bytes]:
        """(변형, 앞 잡음+발화, 클립 꼬리). 발화 순환마다 다음 변형"""
        if name not in self._clips:
            raise SystemExit(f"manifest 조건에 없는 발화: {name}")
        variant, clip, utt_end_s = self._clips[name][cycle % len(self._clips[name])]
        audio, tail = _noise_turn(clip, utt_end_s)
        return variant, audio, tail


def _build_uri(args: argparse.Namespace) -> str:
    return args.url or f"{args.base_url.rstrip('/')}/ws/voice/{args.session_id}"


def _session_id_of(args: argparse.Namespace) -> str:
    if args.session_id:
        return args.session_id
    return args.url.rstrip("/").split("/")[-1] if args.url else "(unknown)"


async def _recv_loop(ws, *, speech_end: float, timeout: float) -> dict:
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


async def _send_tail(ws, data: bytes, chunk_bytes: int, chunk_s: float, more: Callable[[int], bytes] | None) -> None:
    """꼬리를 보내고, more 가 있으면 취소될 때까지 more 로 계속 보낸다(마이크처럼)"""
    await _send_chunks(ws, data, chunk_bytes, chunk_s)
    while more is not None:
        await ws.send(more(chunk_bytes))
        await asyncio.sleep(chunk_s)


async def _run_turn(
    ws,
    audio: bytes,
    silence: bytes,
    chunk_bytes: int,
    chunk_s: float,
    timeout: float,
    turn_gap: bytes = b"",
    tail_more: Callable[[int], bytes] | None = None,
) -> dict:
    try:
        await _send_chunks(ws, audio, chunk_bytes, chunk_s)
    except websockets.ConnectionClosed:
        return {
            "stt_final_ms": None, "transcript": None, "client_response_ms": None,
            "client_turn_ms": None, "terminal": "CLOSED",
        }
    speech_end = time.perf_counter()

    recv_task = asyncio.create_task(_recv_loop(ws, speech_end=speech_end, timeout=timeout))
    silence_task = asyncio.create_task(_send_tail(ws, silence, chunk_bytes, chunk_s, tail_more))
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


def _payload(args: argparse.Namespace, sid: str, results: list[dict], noise: _NoiseCondition | None) -> dict:
    return {
        "session_id": sid,
        "engine": args.engine,
        "tail_silence_ms": None if noise else args.tail_silence_ms,
        "chunk_ms": args.chunk_ms,
        "turn_gap_ms": args.turn_gap_ms,
        "noise": noise.noise if noise else None,
        "snr_db": noise.snr_db if noise else None,
        "lead_ms": noise.lead_ms if noise else 0,
        "turns": results,
    }


async def _run(args: argparse.Namespace) -> None:
    audios = [(Path(a).name, _load_audio(a)) for a in args.audio]
    noise = _NoiseCondition(args.noise_manifest, args.noise, args.snr) if args.noise_manifest else None
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
        + (f" · 잡음 {noise.noise}/{noise.snr_db} (앞 {noise.lead_ms}ms, 꼬리는 응답까지·간격도 잡음)" if noise
           else f" · 무음 {args.tail_silence_ms}ms")
        + f" · {args.turns}턴\n"
    )

    results: list[dict] = []
    async with websockets.connect(uri, max_size=None, open_timeout=args.timeout) as ws:
        await ws.send(json.dumps({"type": "auth", "token": args.token}))
        for t in range(args.turns):
            name, audio = audios[t % len(audios)]
            tail, gap, more, variant = silence, turn_gap, None, None
            if noise:
                variant, audio, tail = noise.turn(name, t // len(audios))
                gap, more = noise.bed.take(len(turn_gap)), noise.bed.take
            res = await _run_turn(
                ws, audio, tail, chunk_bytes, chunk_s, args.turn_timeout, turn_gap=gap, tail_more=more
            )
            res.update({"turn": t + 1, "audio": name})
            if noise:
                res["variant"] = variant
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
        payload = _payload(args, sid, results, noise)
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
    parser.add_argument(
        "--tail-silence-ms", type=int, default=800,
        help="STT endpointing용 끝 무음(ms). 잡음 모드에선 쓰지 않는다(응답이 올 때까지 잡음)",
    )
    parser.add_argument("--turn-timeout", type=float, default=30.0, help="턴당 응답 대기 한도(s, 발화 끝 기준)")
    parser.add_argument(
        "--turn-gap-ms", type=int, default=0, help="응답 뒤 다음 발화까지 보낼 무음(ms). 긴 세션 재현용"
    )
    parser.add_argument(
        "--noise-manifest", default=None,
        help="scripts/stt_noise/mix_noise.py 의 manifest. 주면 --audio 를 그 조건 클립(앞 잡음+발화)으로 바꾸고 "
        "꼬리는 응답이 올 때까지, 턴 간격도 그 잡음으로 채운다. 발화를 한 바퀴 돌 때마다 다음 잡음 변형",
    )
    parser.add_argument("--noise", default=None, help="--noise-manifest 조건: clean·white·pink·babble")
    parser.add_argument("--snr", type=float, default=None, help="--noise-manifest 조건 SNR(dB), clean 이면 생략")
    parser.add_argument("--timeout", type=float, default=10.0, help="연결 타임아웃(s)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not args.url and not args.session_id:
        parser.error("--url 또는 --session-id 중 하나는 필요합니다.")
    if args.noise_manifest and (args.noise is None or (args.noise != "clean" and args.snr is None)):
        parser.error("--noise-manifest 엔 --noise(와 clean 이 아니면 --snr)가 필요합니다.")

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
