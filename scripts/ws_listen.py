"""AI 음성이 프론트에 어떻게 도착하는지 직접 듣기 위한 진단 클라이언트.

서버는 PCM 을 만들어지는 대로 그냥 흘린다(pipeline._run_turn). 실제 재생이 매끄러운지는
"도착 속도가 재생 속도를 앞서는가" 로 결정되므로, 이 스크립트는 두 개의 wav 를 남긴다.

  <out>.raw.wav      받은 PCM 을 그대로 이어붙인 것 — TTS 음질 자체 확인용(끊김 없음)
  <out>.asheard.wav  도착 시각대로 빈 구간을 무음으로 채운 것 — 프론트가 듣는 끊김 재현

두 모드:
  ws   실제 통화 경로 전체 (STT→LLM→TTS). 세션 ID/토큰 필요.
  tts  GPU 워커의 /v1/tts/stream 만 직접. 세션 불필요 — 끊김이 TTS 단계에서 오는지 격리.

예)
  python scripts/ws_listen.py tts --tts-url http://GPU:8080 --text "안녕하세요, 오늘 날씨 좋네요." --play
  python scripts/ws_listen.py ws --base-url ws://localhost:8000 --session-id SID --token JWT \
      --audio tests/output/tts_check_output.wav --play
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
import wave
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlencode

_SR = 16000
_SAMPLE_BYTES = 2
_BYTES_PER_S = _SR * _SAMPLE_BYTES


class Recorder:
    """도착 시각과 함께 PCM 을 모아 두 가지 wav 로 복원한다."""

    def __init__(self, jitter_ms: float, gap_threshold_ms: float) -> None:
        self._jitter = jitter_ms / 1000.0
        self._gap_threshold = gap_threshold_ms / 1000.0
        self._chunks: list[tuple[float, bytes]] = []
        self._t0: float | None = None

    def add(self, pcm: bytes) -> None:
        now = time.perf_counter()
        if self._t0 is None:
            self._t0 = now
        self._chunks.append((now - self._t0, pcm))

    @property
    def empty(self) -> bool:
        return not self._chunks

    def raw(self) -> bytes:
        return b"".join(pcm for _, pcm in self._chunks)

    def as_heard(self) -> tuple[bytes, list[tuple[float, float]]]:
        """재생 헤드가 굶은 만큼 무음을 끼워 넣는다. (오디오, [(재생시각, 무음길이)])"""
        out: list[bytes] = []
        gaps: list[tuple[float, float]] = []
        played = 0.0  # 지금까지 재생한 오디오 길이(s)
        for arrived, pcm in self._chunks:
            playhead = self._jitter + played
            starve = arrived - playhead
            if starve > 0:
                out.append(b"\x00" * (int(starve * _SR) * _SAMPLE_BYTES))
                played += starve
                if starve >= self._gap_threshold:
                    gaps.append((played, starve))
            out.append(pcm)
            played += len(pcm) / _BYTES_PER_S
        return b"".join(out), gaps

    def report(self) -> list[tuple[float, float]]:
        audio_s = len(self.raw()) / _BYTES_PER_S
        wall_s = self._chunks[-1][0] if self._chunks else 0.0
        heard, gaps = self.as_heard()
        heard_s = len(heard) / _BYTES_PER_S
        silence_s = heard_s - audio_s

        print(f"\n청크 {len(self._chunks)}개 · 오디오 {audio_s:.2f}s · 수신에 걸린 시간 {wall_s:.2f}s")
        print(f"선재생 버퍼 {self._jitter * 1000:.0f}ms 가정 → 재생 길이 {heard_s:.2f}s "
              f"(무음 삽입 {silence_s:.2f}s, 실시간 대비 {(heard_s / audio_s if audio_s else 0):.2f}x)")
        if gaps:
            print(f"\n끊김 {len(gaps)}회 (>{self._gap_threshold * 1000:.0f}ms):")
            for at, dur in gaps:
                print(f"  재생 {at:6.2f}s 지점에서 {dur * 1000:7.1f}ms 무음")
        else:
            print(f"\n끊김 없음 (>{self._gap_threshold * 1000:.0f}ms 기준)")
        return gaps

    def write(self, out_prefix: Path) -> tuple[Path, Path]:
        raw_path = out_prefix.with_suffix(".raw.wav")
        heard_path = out_prefix.with_suffix(".asheard.wav")
        _write_wav(raw_path, self.raw())
        _write_wav(heard_path, self.as_heard()[0])
        return raw_path, heard_path


def _write_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(_SAMPLE_BYTES)
        wf.setframerate(_SR)
        wf.writeframes(pcm)


def _load_audio(path: str) -> bytes:
    p = Path(path)
    if p.suffix.lower() == ".wav":
        with wave.open(str(p), "rb") as wf:
            if (wf.getframerate(), wf.getnchannels(), wf.getsampwidth()) != (_SR, 1, _SAMPLE_BYTES):
                raise SystemExit(
                    f"wav 포맷 불일치: {wf.getframerate()}Hz/{wf.getnchannels()}ch/"
                    f"{wf.getsampwidth() * 8}bit → 16000Hz/mono/16bit 로 변환하세요."
                )
            return wf.readframes(wf.getnframes())
    return p.read_bytes()


async def _run_tts(args: argparse.Namespace, rec: Recorder) -> None:
    import httpx

    url = f"{args.tts_url.rstrip('/')}/v1/tts/stream"
    print(f"POST {url}  voice={args.voice}  chars={len(args.text)}")
    started = time.perf_counter()
    async with (
        httpx.AsyncClient(timeout=httpx.Timeout(args.turn_timeout, connect=5.0)) as client,
        client.stream("POST", url, json={"voice": args.voice, "text": args.text}) as resp,
    ):
        resp.raise_for_status()
        async for chunk in resp.aiter_bytes():
            if not chunk:
                continue
            if rec.empty:
                print(f"  첫 청크 {(time.perf_counter() - started) * 1000:.0f}ms")
            rec.add(chunk)


async def _run_ws(args: argparse.Namespace, rec: Recorder) -> None:
    import websockets

    audio = _load_audio(args.audio)
    if args.tail_silence_ms > 0:
        audio += b"\x00" * (int(_SR * args.tail_silence_ms / 1000) * _SAMPLE_BYTES)
    chunk_bytes = int(_BYTES_PER_S * args.chunk_ms / 1000)

    base = args.url or f"{args.base_url.rstrip('/')}/ws/voice/{args.session_id}"
    uri = f"{base}?{urlencode({'token': args.token})}"
    print(f"connect {base}  발화 {len(audio) / _BYTES_PER_S:.1f}s")

    async with websockets.connect(uri, max_size=None, open_timeout=args.timeout) as ws:
        # 송신과 수신을 동시에 돌려야 도착 시각이 실제 값이 된다. 다 보내고 나서 읽으면
        # 응답이 소켓 버퍼에 고여 있다가 한꺼번에 잡혀 끊김이 통째로 사라진다.
        sent_at: dict[str, float] = {}

        async def send_all() -> None:
            for i in range(0, len(audio), chunk_bytes):
                await ws.send(audio[i : i + chunk_bytes])
                await asyncio.sleep(args.chunk_ms / 1000.0)
            sent_at["t"] = time.perf_counter()
            print("  발화 전송 완료 — 응답 대기")

        sender = asyncio.create_task(send_all())

        def since_sent() -> str:
            if "t" not in sent_at:
                return "발화 중"
            return f"발화 종료 +{(time.perf_counter() - sent_at['t']) * 1000.0:.0f}ms"

        started = time.perf_counter()
        deadline = started + args.turn_timeout + len(audio) / _BYTES_PER_S
        try:
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    print("  타임아웃")
                    return
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except (TimeoutError, websockets.ConnectionClosed):
                    print("  연결 종료/타임아웃")
                    return
                if isinstance(msg, bytes):
                    if rec.empty:
                        print(f"  첫 PCM ({since_sent()})")
                    rec.add(msg)
                    continue
                frame = json.loads(msg)
                print(f"  [{since_sent():>16}] {frame}")
                if frame.get("type") in ("speaking_end", "end", "error"):
                    return
        finally:
            sender.cancel()
            with suppress(asyncio.CancelledError):
                await sender


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default="/tmp/ws_listen", help="출력 wav 접두사")
    common.add_argument("--jitter-ms", type=float, default=0.0,
                        help="클라 선재생 버퍼 가정(ms). 0 이면 도착 즉시 재생하는 최악 조건")
    common.add_argument("--gap-ms", type=float, default=30.0, help="끊김으로 볼 무음 하한(ms)")
    common.add_argument("--turn-timeout", type=float, default=60.0)
    common.add_argument("--play", action="store_true", help="끝나고 asheard.wav 를 ffplay 로 재생")

    p_tts = sub.add_parser("tts", parents=[common], help="GPU 워커 /v1/tts/stream 만 직접")
    p_tts.add_argument("--tts-url", required=True, help="예: http://127.0.0.1:8080")
    p_tts.add_argument("--text", required=True)
    p_tts.add_argument("--voice", default="ai")

    p_ws = sub.add_parser("ws", parents=[common], help="통화 경로 전체")
    p_ws.add_argument("--url", default=None, help="완전한 ws URL(.../ws/voice/<SID>)")
    p_ws.add_argument("--base-url", default="ws://localhost:8000")
    p_ws.add_argument("--session-id", default=None)
    p_ws.add_argument("--token", default="")
    p_ws.add_argument("--audio", required=True, help="발화 오디오(.wav 16k/mono/16bit 또는 .pcm)")
    p_ws.add_argument("--chunk-ms", type=int, default=100)
    p_ws.add_argument("--tail-silence-ms", type=int, default=800)
    p_ws.add_argument("--timeout", type=float, default=10.0)

    args = parser.parse_args()
    if args.mode == "ws" and not args.url and not args.session_id:
        parser.error("--url 또는 --session-id 중 하나는 필요합니다.")

    rec = Recorder(args.jitter_ms, args.gap_ms)
    try:
        asyncio.run(_run_tts(args, rec) if args.mode == "tts" else _run_ws(args, rec))
    except KeyboardInterrupt:
        print("\n중단됨 — 지금까지 받은 만큼 저장합니다")

    if rec.empty:
        print("\nPCM 을 한 바이트도 못 받았습니다.", file=sys.stderr)
        raise SystemExit(1)

    rec.report()
    raw_path, heard_path = rec.write(Path(args.out))
    print(f"\n  원본(끊김 없음): {raw_path}")
    print(f"  실제로 들리는 것: {heard_path}")
    if args.play:
        subprocess.run(["ffplay", "-autoexit", "-nodisp", "-loglevel", "error", str(heard_path)])


if __name__ == "__main__":
    main()
