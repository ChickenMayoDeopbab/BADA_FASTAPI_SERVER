"""깨끗한 발화에 합성 잡음(백색·분홍·웅성거림)을 목표 SNR 로 섞어 조건별 오디오와 manifest 를 만든다 (네트워크 없음).

  .venv/bin/python scripts/stt_noise/mix_noise.py
  (기본: .harness/stt-compare-2026-09/audio 8문장 → .harness/stt-noise-2026-10/audio)

SNR = 깨끗한 발화의 말소리 프레임(20ms, −45dBFS 이상) 평균 전력 ÷ 같은 프레임 위 잡음 평균 전력.
발화는 먼저 말소리 레벨 --speech-level-db 로 맞춘다(clean 도 같은 레벨).
피크가 −1dBFS 를 넘으면 클립 전체를 줄인다(SNR 불변).
클립 = 앞 잡음(--lead-ms) + 발화(+잡음) + 뒤 잡음(--tail-ms). clean 은 앞뒤가 디지털 무음.
잡음만 클립(noise_only/)은 (잡음, SNR)마다 그 SNR 의 잡음 레벨로 --noise-only-ms 길이.
웅성거림은 대상 발화를 뺀 나머지 발화들을 --babble-talkers 겹 겹쳐 만든다(잡음만 클립은 전부로).
잡음 실현은 (잡음, 발화, 변형)마다 seed 로 고정이라 같은 변형의 SNR 단계는 크기만 다른 같은 잡음이다.
변형(--variants)끼리는 다른 잡음이고, run_chirp 의 반복 r 은 변형 (r−1) mod K + 1 을 쓴다(clean 은 하나뿐).
"""

import argparse
import json
import math
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.core.pricing import get_price  # noqa: E402

SR = 16000
BYTES_PER_S = SR * 2
FRAME = SR // 50  # 20ms
SOUND_DBFS, ONSET_FRAMES = -45.0, 3  # scripts/tts_bench/prep_ref.py 와 같은 말소리 판정
PEAK_MAX_DB = -1.0
PINK_LOW_HZ = 20.0
NOISES = ("white", "pink", "babble")
DEFAULT_SNRS = (20.0, 10.0, 5.0, 0.0)
SRC_DIR = ROOT / ".harness" / "stt-compare-2026-09" / "audio"
OUT_DIR = ROOT / ".harness" / "stt-noise-2026-10"


def read_pcm16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as f:
        if f.getframerate() != SR or f.getnchannels() != 1 or f.getsampwidth() != 2:
            raise ValueError(f"16kHz mono PCM16 wav 만 받는다: {path}")
        return np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float64) / 32768.0


def write_pcm16(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with wave.open(str(tmp), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(np.round(np.clip(x, -1.0, 32767 / 32768) * 32768).astype("<i2").tobytes())
    tmp.replace(path)


def frame_power(x: np.ndarray) -> np.ndarray:
    k = len(x) // FRAME
    f = np.asarray(x[: k * FRAME], dtype=np.float64).reshape(k, FRAME)
    return np.mean(f * f, axis=1)


def active_frames(x: np.ndarray) -> np.ndarray:
    return 10 * np.log10(frame_power(x) + 1e-20) >= SOUND_DBFS


def speech_span_s(active: np.ndarray) -> tuple[float, float]:
    """(첫 말소리 프레임 시작, 마지막 말소리 프레임 끝) 초 — 60ms 이어져야 시작으로 본다"""
    onset = next(
        (i for i in range(len(active) - ONSET_FRAMES + 1) if active[i : i + ONSET_FRAMES].all()), None
    )
    if onset is None:
        raise ValueError("말소리가 없다")
    end = len(active) - int(np.argmax(active[::-1]))
    return onset * FRAME / SR, end * FRAME / SR


def level_db(x: np.ndarray, active: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(frame_power(x)[active]) + 1e-20))


def rms_db(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-20))


def white(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.standard_normal(n)


def pink(rng: np.random.Generator, n: int) -> np.ndarray:
    """1/f 전력(옥타브마다 같은 에너지), 20Hz 아래는 뺀다"""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    scale = np.zeros_like(f)
    keep = f >= PINK_LOW_HZ
    scale[keep] = 1 / np.sqrt(f[keep])
    x = np.fft.irfft(spec * scale, n)
    return x / math.sqrt(np.mean(x * x))


def babble(rng: np.random.Generator, n: int, sources: list[np.ndarray], talkers: int) -> np.ndarray:
    """말소리 레벨을 맞춘 발화들을 화자마다 순서·시작점을 달리해 이어 붙이고 겹친다"""
    if not sources:
        raise ValueError("웅성거림을 만들 다른 발화가 없다(발화가 2개 이상 필요)")
    out = np.zeros(n)
    for _ in range(talkers):
        seq = np.concatenate([sources[i] for i in rng.permutation(len(sources))])
        off = int(rng.integers(0, len(seq)))
        out += np.tile(seq, (off + n) // len(seq) + 1)[off : off + n]
    return out


def to_level(x: np.ndarray, target_db: float) -> np.ndarray:
    return x * 10 ** ((target_db - level_db(x, active_frames(x))) / 20)


def make_noise(kind: str, rng: np.random.Generator, n: int, sources: list[np.ndarray], talkers: int) -> np.ndarray:
    if kind == "white":
        return white(rng, n)
    if kind == "pink":
        return pink(rng, n)
    if kind == "babble":
        return babble(rng, n, sources, talkers)
    raise ValueError(f"모르는 잡음: {kind}")


def _peak_guard(x: np.ndarray) -> float:
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    limit = 10 ** (PEAK_MAX_DB / 20)
    return limit / peak if peak > limit else 1.0


def mix_clip(
    utt: np.ndarray, noise: np.ndarray | None, snr_db: float | None, *, lead: int, tail: int, target_db: float
) -> tuple[np.ndarray, np.ndarray, dict]:
    """(섞인 클립, 클립 안의 발화 성분, 정보). noise 가 None 이면 clean(앞뒤 디지털 무음)"""
    active = active_frames(utt)
    if not active.any():
        raise ValueError("말소리가 없다")
    start, end = speech_span_s(active)
    gain = 10 ** ((target_db - level_db(utt, active)) / 20)
    track = np.concatenate([np.zeros(lead), utt * gain, np.zeros(tail)])
    mixed = track.copy()
    noise_db = None
    if noise is not None:
        if snr_db is None:
            raise ValueError("잡음 조건엔 SNR 이 필요하다")
        seg = np.asarray(noise[: len(track)], dtype=np.float64)
        if len(seg) < len(track):
            raise ValueError("잡음이 클립보다 짧다")
        under = frame_power(seg[lead : lead + len(utt)])[active].mean()
        noise_gain = math.sqrt(10 ** ((target_db - snr_db) / 10) / under)
        mixed += seg * noise_gain
        noise_db = rms_db(seg * noise_gain)
    k = _peak_guard(mixed)
    info = {
        "lead_s": round(lead / SR, 6),
        "speech_start_s": round(lead / SR + start, 6),
        "speech_end_s": round(lead / SR + end, 6),
        "utt_end_s": round((lead + len(utt)) / SR, 6),
        "dur_s": round(len(mixed) / SR, 6),
        "speech_level_db": round(target_db + 20 * math.log10(k), 2),
        "noise_rms_db": None if noise_db is None else round(noise_db + 20 * math.log10(k), 2),
        "clip_gain_db": round(20 * math.log10(k), 2),
    }
    return mixed * k, track * k, info


def noise_only_clip(noise: np.ndarray, snr_db: float, target_db: float) -> tuple[np.ndarray, dict]:
    """잡음 RMS 를 (말소리 레벨 − SNR) 로"""
    x = np.asarray(noise, dtype=np.float64) * 10 ** ((target_db - snr_db - rms_db(noise)) / 20)
    k = _peak_guard(x)
    info = {
        "dur_s": round(len(x) / SR, 3),
        "noise_rms_db": round(rms_db(x * k), 2),
        "clip_gain_db": round(20 * math.log10(k), 2),
    }
    return x * k, info


def cond_name(noise: str, snr_db: float | None) -> str:
    return "clean" if noise == "clean" else f"{noise}_snr{snr_db:g}"


def build(
    src: Path,
    out: Path,
    *,
    noises: list[str],
    snrs: list[float],
    lead_ms: int,
    tail_ms: int,
    noise_only_ms: int,
    target_db: float,
    talkers: int,
    seed: int,
    variants: int,
) -> dict:
    texts: dict[str, str] = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    names = sorted(texts)
    utts = {name: read_pcm16(src / name) for name in names}
    leveled = {name: to_level(x, target_db) for name, x in utts.items()}
    lead, tail = SR * lead_ms // 1000, SR * tail_ms // 1000
    clips: list[dict] = []

    def add(rel: str, x: np.ndarray, row: dict) -> None:
        write_pcm16(out / rel, x)
        clips.append({"file": rel, **row})

    for name in names:
        mixed, _, info = mix_clip(utts[name], None, None, lead=lead, tail=tail, target_db=target_db)
        add(f"clean/{name}", mixed, {"utt": name, "text": texts[name], "noise": "clean", "snr_db": None,
                                     "variant": None, "noise_only": False, **info})
    for v in range(1, variants + 1):
        for ki, kind in enumerate(noises):
            for ui, name in enumerate(names):
                n = lead + len(utts[name]) + tail
                others = [leveled[o] for o in names if o != name]
                bed = make_noise(kind, np.random.default_rng([seed, ki, ui, v]), n, others, talkers)
                for snr in snrs:
                    mixed, _, info = mix_clip(utts[name], bed, snr, lead=lead, tail=tail, target_db=target_db)
                    add(f"{cond_name(kind, snr)}/v{v}/{name}", mixed, {
                        "utt": name, "text": texts[name], "noise": kind, "snr_db": snr, "variant": v,
                        "noise_only": False, **info,
                    })
        if noise_only_ms > 0:
            n = SR * noise_only_ms // 1000
            for ki, kind in enumerate(noises):
                rng = np.random.default_rng([seed, ki, 10_000, v])
                bed = make_noise(kind, rng, n, list(leveled.values()), talkers)
                for snr in snrs:
                    x, info = noise_only_clip(bed, snr, target_db)
                    add(f"noise_only/v{v}/{cond_name(kind, snr)}.wav", x, {
                        "utt": None, "text": None, "noise": kind, "snr_db": snr, "variant": v,
                        "noise_only": True, **info,
                    })
    manifest = {
        "sr": SR,
        "src": str(src),
        "seed": seed,
        "speech_level_db": target_db,
        "lead_ms": lead_ms,
        "tail_ms": tail_ms,
        "noise_only_ms": noise_only_ms,
        "babble_talkers": talkers,
        "variants": variants,
        "snr_definition": "말소리 프레임(20ms, -45dBFS 이상) 평균 전력 / 같은 프레임 잡음 평균 전력",
        "clips": clips,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


def stt_usd(seconds: float) -> float:
    price = get_price("google_stt", "chirp_3", "stt_sec")
    return price.cost(seconds) if price else float("nan")


def summarize(manifest: dict) -> list[tuple[str, int, float]]:
    rows: dict[str, list[float]] = {}
    for c in manifest["clips"]:
        key = ("noise_only/" if c["noise_only"] else "") + cond_name(c["noise"], c["snr_db"])
        rows.setdefault(key, []).append(c["dur_s"])
    return [(k, len(v), sum(v)) for k, v in rows.items()]


def rep_seconds(manifest: dict) -> float:
    """반복 한 번에 보내는 오디오 초(clean + 변형 하나)"""
    return sum(c["dur_s"] for c in manifest["clips"] if c.get("variant") in (None, 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(SRC_DIR), help="manifest.json(파일명→정답) 과 16k mono wav 가 있는 폴더")
    ap.add_argument("--out", default=str(OUT_DIR / "audio"))
    ap.add_argument("--noise", nargs="+", default=list(NOISES), choices=NOISES)
    ap.add_argument("--snr", nargs="+", type=float, default=list(DEFAULT_SNRS), help="dB")
    ap.add_argument("--lead-ms", type=int, default=1000, help="발화 앞 잡음")
    ap.add_argument("--tail-ms", type=int, default=5000, help="발화 뒤 잡음(끝점 대기)")
    ap.add_argument("--noise-only-ms", type=int, default=10000, help="잡음만 클립 길이, 0 이면 안 만든다")
    ap.add_argument("--speech-level-db", type=float, default=-26.0, help="말소리 레벨(dBFS)")
    ap.add_argument("--babble-talkers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", type=int, default=3, help="잡음 변형 수(반복마다 다른 변형을 쓴다)")
    ap.add_argument("--reps", type=int, default=3, help="예상 비용 계산용 반복 수")
    a = ap.parse_args()

    out = Path(a.out).expanduser()
    manifest = build(
        Path(a.src).expanduser(), out, noises=a.noise, snrs=a.snr, lead_ms=a.lead_ms, tail_ms=a.tail_ms,
        noise_only_ms=a.noise_only_ms, target_db=a.speech_level_db, talkers=a.babble_talkers, seed=a.seed,
        variants=a.variants,
    )
    print(f"{'조건':<24} {'클립':>4} {'초':>8}")
    total = 0.0
    for key, n, secs in summarize(manifest):
        print(f"{key:<24} {n:>4} {secs:>8.1f}")
        total += secs
    upper = rep_seconds(manifest) * a.reps * 2
    print(f"\n클립 {len(manifest['clips'])}개 · {total:.1f}초 → {out}/manifest.json")
    print(
        f"예상 chirp 비용 상한(운영·denoiser 2조건 × {a.reps}회, 조기 종료 없이 끝까지 보낼 때): "
        f"{upper / 60:.1f}분 ≈ ${stt_usd(upper):.2f}"
    )
    print("고른 조건만의 예상은 run_chirp.py --dry-run")


if __name__ == "__main__":
    main()
