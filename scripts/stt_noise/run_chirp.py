"""mix_noise 클립을 운영 chirp 설정(+ denoiser 조건)으로 실시간 속도로 흘려 모든 FINAL·speech 이벤트를 기록한다.

  .venv/bin/python scripts/stt_noise/run_chirp.py --dry-run     # STT 호출 없이 작업 수·예상 비용
  .venv/bin/python scripts/stt_noise/run_chirp.py               # 실측(돈 든다, --max-usd 넘으면 거절)

운영 설정 = get_settings() + build_stt_client(GoogleSTTClient).
denoiser 조건은 그 config 요청에 DenoiserConfig 만 더한다.
파이프라인처럼 첫 오디오 청크에서 스트림을 열고, FINAL(빈 것 포함)마다 반쯤 닫고 다음 청크에서 다시 연다.
청크는 마이크처럼 그 길이만큼 시간이 지난 뒤 보낸다(기본 100ms). 꼬리 잡음을 보내는 동안에도 계속 받는다.
발화 클립은 덮은 오디오가 말소리 끝 − 0.3초를 넘는 비어 있지 않은 FINAL 에서 멈춘다(score.final_end_s,
--full-tail 이면 끝까지), 잡음만 클립은 끝까지. 반복 r 은 잡음 변형 (r−1) mod K + 1 을 쓴다.
오디오가 다 나가면 EOS 를 넣고 --drain-s 동안 남은 이벤트를 받는다 — 이때 온 이벤트는 after_eos(채점 제외).
워밍업에서 스트림 에러가 나면 본 측정을 시작하지 않는다.
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
import wave
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from google.cloud.speech_v2.types import cloud_speech as cs  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.services.stt import AUDIO_EOS, GoogleSTTClient, STTError, STTEventType, build_stt_client  # noqa: E402
from scripts.stt_noise.mix_noise import BYTES_PER_S, OUT_DIR, SR, cond_name, stt_usd  # noqa: E402
from scripts.stt_noise.score import END_MARGIN_S, failed, final_end_s  # noqa: E402

CLIP_KEYS = (
    "file", "utt", "text", "noise", "snr_db", "variant", "noise_only",
    "lead_s", "speech_start_s", "speech_end_s", "utt_end_s", "dur_s",
)
MAX_STREAM_ERRORS = 3


class DenoiseSTTClient(GoogleSTTClient):
    """운영 chirp 설정에 denoiser 만 켠 클라이언트"""

    def __init__(self, *args, snr_threshold: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.snr_threshold = snr_threshold

    def _build_config_request(self) -> cs.StreamingRecognizeRequest:
        request = super()._build_config_request()
        request.streaming_config.config.denoiser_config = cs.DenoiserConfig(
            denoise_audio=True, snr_threshold=self.snr_threshold
        )
        return request


def make_clients(settings, modes: list[bool], snr_threshold: float) -> dict[bool, GoogleSTTClient]:
    """False = 운영 그대로, True = denoiser"""
    clients: dict[bool, GoogleSTTClient] = {}
    for denoise in modes:
        if not denoise:
            client = build_stt_client(settings)
            if not isinstance(client, GoogleSTTClient):
                raise SystemExit(f"운영 STT 가 chirp 가 아니다(STT_ENGINE={settings.stt_engine})")
            clients[False] = client
        else:
            clients[True] = DenoiseSTTClient(
                project_id=settings.google_project_id,
                location=settings.google_stt_location,
                model=settings.google_stt_model,
                language=settings.google_stt_language,
                snr_threshold=snr_threshold,
            )
    return clients


def config_dict(client: GoogleSTTClient) -> dict:
    return cs.StreamingRecognizeRequest.to_dict(client._build_config_request(), use_integers_for_enums=False)


async def _sleep_until(deadline: float) -> None:
    delay = deadline - time.perf_counter()
    if delay > 0:
        await asyncio.sleep(delay)


class _Mic:
    """청크를 실시간 속도로 지금 큐에 넣는다. FINAL 로 반쯤 닫힌 큐는 새 큐로 바뀐다(파이프라인 _half_close)"""

    def __init__(self, pcm: bytes, chunk_bytes: int) -> None:
        self._chunks = [pcm[i : i + chunk_bytes] for i in range(0, len(pcm), chunk_bytes)]
        self._offset = {id(c): i * chunk_bytes for i, c in enumerate(self._chunks)}
        self.queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.sent = 0
        self.done = False

    def offset_s(self, chunk: bytes) -> float:
        return self._offset[id(chunk)] / BYTES_PER_S

    async def run(self, t0: float, chunk_s: float) -> None:
        for k, chunk in enumerate(self._chunks):
            await _sleep_until(t0 + (k + 1) * chunk_s)
            self.queue.put_nowait(chunk)
            self.sent += len(chunk)
        self.done = True
        self.queue.put_nowait(AUDIO_EOS)

    def half_close(self, queue: "asyncio.Queue[bytes | None]") -> None:
        if self.queue is queue:
            self.queue = asyncio.Queue()
            if self.done:
                self.queue.put_nowait(AUDIO_EOS)
        queue.put_nowait(AUDIO_EOS)


class _Recorder:
    def __init__(self, t0: float) -> None:
        self.t0 = t0
        self.events: list[dict] = []
        self.streams = 0
        self.interims = 0
        self.first_interim_t: float | None = None

    def _t(self) -> float:
        return round(time.perf_counter() - self.t0, 3)

    def add(self, event, stream_no: int, open_s: float, *, after_eos: bool) -> dict | None:
        t = self._t()
        if event.type == STTEventType.INTERIM:
            self.interims += 1
            if self.first_interim_t is None:
                self.first_interim_t = t
            return None
        row = {"t": t, "type": event.type.value, "stream": stream_no, "stream_open_s": round(open_s, 3),
               "after_eos": after_eos}
        if event.type == STTEventType.FINAL:
            row["text"] = event.text
            row["end_offset_s"] = event.end_offset_sec
        self.events.append(row)
        return row

    def error(self, exc: BaseException, stream_no: int) -> None:
        self.events.append({"t": self._t(), "type": "error", "stream": stream_no,
                            "error": f"{type(exc).__name__}: {exc}"})


async def _consume(stt, mic: _Mic, rec: _Recorder, *, stop_after_s: float | None) -> str:
    """스트림을 열고 FINAL 마다 닫고 다음 청크에서 다시 연다. 멈춘 까닭을 돌려준다"""
    errors = 0
    while True:
        queue = mic.queue
        first = await queue.get()
        if first is AUDIO_EOS:
            return "audio_end"
        rec.streams += 1
        stream_no = rec.streams
        open_s = mic.offset_s(first)
        stream = stt.stream(queue, first_chunk=first)
        stopped: str | None = None
        try:
            async for event in stream:
                row = rec.add(event, stream_no, open_s, after_eos=mic.done)
                if event.type != STTEventType.FINAL:
                    continue
                if mic.done:
                    stopped = "audio_end"
                elif event.text.strip() and stop_after_s is not None and final_end_s(row) >= stop_after_s:
                    stopped = "turn"
                break
        except STTError as exc:
            rec.error(exc, stream_no)
            errors += 1
            if errors >= MAX_STREAM_ERRORS:
                stopped = "error"
        finally:
            mic.half_close(queue)
            with suppress(Exception):
                await stream.aclose()
        if stopped:
            return stopped


async def run_clip(stt, pcm: bytes, *, chunk_ms: int, stop_after_s: float | None, drain_s: float) -> dict:
    """한 클립을 흘려 이벤트를 모은다. 끝 위치가 stop_after_s 이상인 비어 있지 않은 FINAL 에서 멈춘다(None=끝까지)"""
    chunk_bytes = max(2, BYTES_PER_S * chunk_ms // 1000 // 2 * 2)
    mic = _Mic(pcm, chunk_bytes)
    t0 = time.perf_counter()
    rec = _Recorder(t0)
    mic_task = asyncio.create_task(mic.run(t0, chunk_ms / 1000))
    consume = asyncio.create_task(_consume(stt, mic, rec, stop_after_s=stop_after_s))
    try:
        while not consume.done():
            if mic_task.done():
                mic_task.result()
                await asyncio.wait({consume}, timeout=drain_s)
                break
            await asyncio.wait({consume, mic_task}, return_when=asyncio.FIRST_COMPLETED)
        stopped = consume.result() if consume.done() else "drain_timeout"
    finally:
        for task in (consume, mic_task):
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
    return {
        "stopped": stopped,
        "audio_sent_s": round(mic.sent / BYTES_PER_S, 3),
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "streams": rec.streams,
        "interims": rec.interims,
        "first_interim_t": rec.first_interim_t,
        "events": rec.events,
    }


@dataclass(frozen=True)
class Job:
    clip: dict
    denoise: bool
    rep: int


def _snr_match(clip: dict, snrs: list[float] | None) -> bool:
    if snrs is None or clip["noise"] == "clean":
        return True
    return any(abs(clip["snr_db"] - s) < 1e-6 for s in snrs)


def plan_jobs(
    manifest: dict,
    *,
    reps: int,
    modes: list[bool],
    noises: list[str] | None = None,
    snrs: list[float] | None = None,
    utts: list[str] | None = None,
    noise_only: bool = True,
) -> list[Job]:
    """반복 → 클립 → denoiser 순으로 섞어 시간대 차이가 조건에 몰리지 않게 한다. 반복마다 다른 잡음 변형"""
    n_var = int(manifest.get("variants") or 1)

    def wanted(c: dict) -> bool:
        if (noises is not None and c["noise"] not in noises) or not _snr_match(c, snrs):
            return False
        if c["noise_only"]:
            return noise_only
        return utts is None or c["utt"] in utts

    def of_rep(c: dict, rep: int) -> bool:
        return c.get("variant") is None or c["variant"] == (rep - 1) % n_var + 1

    clips = [c for c in manifest["clips"] if wanted(c)]
    return [Job(c, d, rep) for rep in range(1, reps + 1) for c in clips if of_rep(c, rep) for d in modes]


def warmup_clip(manifest: dict) -> dict | None:
    utterances = [c for c in manifest["clips"] if not c["noise_only"]]
    clean = [c for c in utterances if c["noise"] == "clean"]
    return (clean or utterances or [None])[0]


def estimate_s(jobs: list[Job], warmup: dict | None, n_warm: int, modes: list[bool]) -> float:
    """끝까지 다 보낼 때의 오디오 초(상한)"""
    warm = warmup["dur_s"] * n_warm * len(modes) if warmup else 0.0
    return sum(j.clip["dur_s"] for j in jobs) + warm


def git_info() -> dict:
    def git(*args: str) -> str:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
        return r.stdout.strip()

    return {"commit": git("rev-parse", "HEAD") or None, "dirty": bool(git("status", "--porcelain"))}


def _clip_row(job: Job, chunk_ms: int, client) -> dict:
    return {
        **{k: job.clip.get(k) for k in CLIP_KEYS}, "denoise": job.denoise, "rep": job.rep, "chunk_ms": chunk_ms,
        "denoise_snr_threshold": getattr(client, "snr_threshold", None) if job.denoise else None,
    }


def _stop_after(clip: dict, full_tail: bool) -> float | None:
    return None if clip["noise_only"] or full_tail else clip["speech_end_s"] - END_MARGIN_S


def save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as f:
        if f.getframerate() != SR or f.getnchannels() != 1 or f.getsampwidth() != 2:
            raise ValueError(f"16kHz mono PCM16 wav 만 받는다: {path}")
        return f.readframes(f.getnframes())


async def run_jobs(
    jobs: list[Job],
    clients: dict,
    audio_dir: Path,
    *,
    chunk_ms: int,
    drain_s: float,
    full_tail: bool = False,
    concurrency: int = 1,
    on_result=None,
) -> list[dict]:
    pcm_cache: dict[str, bytes] = {}

    def pcm_of(clip: dict) -> bytes:
        if clip["file"] not in pcm_cache:
            pcm_cache[clip["file"]] = read_pcm(audio_dir / clip["file"])
        return pcm_cache[clip["file"]]

    gate = asyncio.Semaphore(max(1, concurrency))
    results: list[dict | None] = [None] * len(jobs)

    async def one(i: int, job: Job) -> None:
        async with gate:
            res = await run_clip(
                clients[job.denoise], pcm_of(job.clip), chunk_ms=chunk_ms,
                stop_after_s=_stop_after(job.clip, full_tail), drain_s=drain_s,
            )
        results[i] = {**_clip_row(job, chunk_ms, clients[job.denoise]), **res}
        if on_result is not None:
            on_result(i, results[i])

    await asyncio.gather(*(one(i, j) for i, j in enumerate(jobs)))
    return [r for r in results if r is not None]


async def _run(args: argparse.Namespace, manifest: dict, jobs: list[Job], modes: list[bool]) -> None:
    settings = get_settings()
    clients = make_clients(settings, modes, args.denoise_snr_threshold)
    audio_dir = Path(args.manifest).expanduser().parent
    out = Path(args.out).expanduser()
    payload: dict = {
        "tool": "scripts/stt_noise/run_chirp.py",
        "git": git_info(),
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "finished_at": None,
        "manifest": str(Path(args.manifest).expanduser()),
        "audio": {k: v for k, v in manifest.items() if k != "clips"},
        "stt": {
            "engine": settings.stt_engine,
            "location": settings.google_stt_location,
            "model": settings.google_stt_model,
            "language": settings.google_stt_language,
            "configs": {("denoise" if d else "operational"): config_dict(c) for d, c in clients.items()},
        },
        "chunk_ms": args.chunk_ms,
        "reps": args.reps,
        "drain_s": args.drain_s,
        "full_tail": args.full_tail,
        "concurrency": args.concurrency,
        "warmup": [],
        "clips": [],
        "audio_sent_s": 0.0,
        "stt_audio_s": 0.0,
        "est_usd": 0.0,
    }

    def tally() -> None:
        sent = sum(r["audio_sent_s"] for r in payload["warmup"] + payload["clips"])
        payload["audio_sent_s"] = round(sent, 3)
        payload["stt_audio_s"] = round(sum(c.audio_bytes_sent for c in clients.values()) / BYTES_PER_S, 3)
        payload["est_usd"] = round(stt_usd(payload["stt_audio_s"]), 4)
        save(out, payload)

    warm = warmup_clip(manifest)
    if warm is not None:
        warm_jobs = [Job(warm, d, 0) for _ in range(args.warmup) for d in modes]
        payload["warmup"] = await run_jobs(warm_jobs, clients, audio_dir, chunk_ms=args.chunk_ms, drain_s=args.drain_s)
        tally()
        print(f"워밍업 {len(payload['warmup'])}회: " + ", ".join(r["stopped"] for r in payload["warmup"]))
        bad = [r for r in payload["warmup"] if failed(r)]
        if bad:
            errors = [e["error"] for r in bad for e in r["events"] if e["type"] == "error"]
            raise SystemExit(f"워밍업 스트림 에러 — 본 측정을 시작하지 않는다({out}): {errors[:3]}")

    done = 0

    def on_result(_i: int, row: dict) -> None:
        nonlocal done
        done += 1
        payload["clips"].append(row)
        tally()
        turn = next((e for e in row["events"] if e["type"] == "final" and e["text"].strip()), None)
        cond = ("noise_only/" if row["noise_only"] else "") + cond_name(row["noise"], row["snr_db"])
        print(
            f"[{done}/{len(jobs)}] r{row['rep']} {cond} {row['utt'] or ''} denoise={row['denoise']} "
            f"{row['stopped']} streams={row['streams']} {turn['text'] if turn else '—'!r}"
        )

    await run_jobs(
        jobs, clients, audio_dir, chunk_ms=args.chunk_ms, drain_s=args.drain_s,
        full_tail=args.full_tail, concurrency=args.concurrency, on_result=on_result,
    )
    payload["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    tally()
    print(f"\n보낸 오디오 {payload['stt_audio_s']:.1f}초 ≈ ${payload['est_usd']:.2f} → {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(OUT_DIR / "audio" / "manifest.json"), help="mix_noise.py 의 manifest")
    ap.add_argument("--out", default=None, help="결과 JSON(기본 .harness/stt-noise-2026-10/runs/chirp-<시각>.json)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--chunk-ms", type=int, default=100, help="송신 청크(앱 실측: 웹 256·iOS 64)")
    ap.add_argument("--denoise", choices=("both", "off", "on"), default="both", help="off=운영 그대로, on=denoiser")
    ap.add_argument(
        "--denoise-snr-threshold", type=float, default=0.0, help="DenoiserConfig.snr_threshold, 0=거르지 않음"
    )
    ap.add_argument("--noise", nargs="+", default=None, help="고를 잡음(clean white pink babble), 기본 전부")
    ap.add_argument("--snr", nargs="+", type=float, default=None, help="고를 SNR, 기본 전부")
    ap.add_argument("--utt", nargs="+", default=None, help="고를 발화 파일명, 기본 전부")
    ap.add_argument("--no-noise-only", action="store_true", help="잡음만 클립을 뺀다")
    ap.add_argument("--warmup", type=int, default=1, help="조건마다 먼저 흘려 버리는 clean 클립 수(콜드 대비)")
    ap.add_argument("--drain-s", type=float, default=3.0, help="오디오를 다 보낸 뒤 이벤트 대기")
    ap.add_argument("--full-tail", action="store_true", help="턴 FINAL 뒤에도 클립 끝까지 보낸다")
    ap.add_argument("--concurrency", type=int, default=1, help="동시에 흘릴 클립 수")
    ap.add_argument("--max-usd", type=float, default=5.0, help="예상 비용 상한이 이걸 넘으면 시작하지 않는다")
    ap.add_argument("--dry-run", action="store_true", help="STT 호출 없이 작업 수·예상 비용만")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).expanduser().read_text(encoding="utf-8"))
    modes = {"both": [False, True], "off": [False], "on": [True]}[args.denoise]
    jobs = plan_jobs(
        manifest, reps=args.reps, modes=modes, noises=args.noise, snrs=args.snr, utts=args.utt,
        noise_only=not args.no_noise_only,
    )
    if not jobs:
        raise SystemExit("고른 조건에 클립이 없다")
    upper = estimate_s(jobs, warmup_clip(manifest), args.warmup, modes)
    usd = stt_usd(upper)
    n_only = sum(j.clip["noise_only"] for j in jobs)
    print(
        f"작업 {len(jobs)}개(발화 {len(jobs) - n_only} · 잡음만 {n_only}) · denoiser {args.denoise} · "
        f"{args.reps}회 · 청크 {args.chunk_ms}ms · 잡음 변형 {manifest.get('variants') or 1}개"
    )
    print(f"오디오 상한 {upper:.0f}초({upper / 60:.1f}분) ≈ ${usd:.2f} · 실시간이라 걸리는 시간도 대략 이만큼(÷동시성)")
    if args.dry_run:
        return
    if usd > args.max_usd:
        raise SystemExit(f"예상 ${usd:.2f} > --max-usd {args.max_usd:g}")
    if args.out is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.out = str(OUT_DIR / "runs" / f"chirp-{stamp}.json")
    asyncio.run(_run(args, manifest, jobs, modes))


if __name__ == "__main__":
    main()
