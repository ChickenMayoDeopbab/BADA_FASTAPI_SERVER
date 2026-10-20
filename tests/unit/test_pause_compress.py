import numpy as np
import pytest

from app.services.pause_compress import ClientBuffer, PauseCompressor

SR = 16000
LAG = 20


def ms(n: float) -> int:
    return int(SR * n / 1000)


def speech(dur: float, db: float = -20.0) -> np.ndarray:
    t = np.arange(ms(dur)) / SR
    return 10 ** (db / 20) * np.sqrt(2) * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t) ** 2)


def sil(dur: float, db: float | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
    if db is None:
        return np.zeros(ms(dur))
    rng = rng or np.random.default_rng(int(dur * 1000 + abs(db)))
    return rng.normal(0, 10 ** (db / 20), ms(dur))


def pcm(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def run(b: bytes, chunks=None, eos: bool = True, spare=None, **kw):
    pc = PauseCompressor(**kw)
    out, firsts = [], []
    for i, c in enumerate(chunks or [b]):
        o = pc.push(c, spare(i)) if spare else pc.push(c)
        firsts.append(o)
        out.append(o)
    out.append(pc.finish(eos, spare(-1)) if (spare and eos) else pc.finish(eos))
    return b"".join(out), pc, firsts


def split_rand(b: bytes, rng: np.random.Generator, lo: int = 1, hi: int = 6000) -> list[bytes]:
    out, i = [], 0
    while i < len(b):
        n = int(rng.integers(lo, hi))
        out.append(b[i : i + n])
        i += n
    return out


def subseq(out: bytes, inp: bytes) -> bool:
    f = 320
    frames = [inp[i : i + f] for i in range(0, len(inp), f)]
    k = 0
    for frame in (out[i : i + f] for i in range(0, len(out), f)):
        while k < len(frames) and frames[k] != frame:
            k += 1
        if k == len(frames):
            return False
        k += 1
    return True


def loud(b: bytes) -> int:
    a = np.frombuffer(b[: len(b) // 320 * 320], dtype="<i2").reshape(-1, 160).astype(np.float64) / 32768
    return int(((10 * np.log10((a * a).mean(1) + 1e-12)) >= -35).sum())


def _first_chunks(b: bytes) -> list[bytes]:
    return [b[:4480]] + [b[4480 + i : 4480 + i + 5120] for i in range(0, len(b) - 4480, 5120)]


def test_speech_without_pause_is_untouched() -> None:
    x = pcm(np.concatenate([speech(800), speech(600, -25)]))
    assert run(x)[0] == x


def test_first_chunk_and_leading_silence_pass_as_is() -> None:
    b = pcm(np.concatenate([sil(300), speech(500), sil(900)]))
    chunks = _first_chunks(b)
    o, _, firsts = run(b, chunks)
    assert firsts[0] == chunks[0]
    assert o[: ms(300) * 2] == b[: ms(300) * 2]


def test_short_inner_pause_kept_and_end_trimmed_to_keep() -> None:
    o, pc, _ = run(pcm(np.concatenate([sil(100), speech(700), sil(200), speech(500), sil(900)])))
    assert o == pcm(np.concatenate([sil(100), speech(700), sil(200), speech(500), sil(160 + LAG)]))
    assert pc.stats.end_dropped_ms == 720 and pc.stats.outcome == "trimmed_end"


def test_long_inner_pause_capped_to_keep_head_and_tail() -> None:
    s1, s2 = speech(700), speech(500, -18)
    xb = pcm(np.concatenate([s1, sil(900, -70), s2]))
    n1 = len(pcm(s1))
    p_in = xb[n1 : n1 + ms(900) * 2]
    o, pc, _ = run(xb)
    assert o == xb[:n1] + p_in[: ms(160 + LAG) * 2] + p_in[-ms(90) * 2 :] + xb[n1 + ms(900) * 2 :]
    assert pc.stats.pauses_capped == 1 and pc.stats.capped_ms == 630
    assert pc.stats.max_held_ms <= 190


def test_chunking_does_not_change_output() -> None:
    rng = np.random.default_rng(5)
    x = np.concatenate([sil(250), speech(400), sil(120), speech(300), sil(600, -65), speech(900), sil(40),
                        speech(200), sil(1100)])
    xb = pcm(x)
    whole = run(xb)[0]
    for _ in range(30):
        assert run(xb, split_rand(xb, rng))[0] == whole
    assert subseq(whole, xb)


def test_abort_drops_held_tail() -> None:
    pc = PauseCompressor()
    out = pc.push(pcm(np.concatenate([speech(500), sil(800)])))
    assert pc.finish(False) == b""
    assert len(out) == ms(660 + LAG) * 2 and pc.stats.outcome == "aborted"


def test_weak_breath_in_end_silence_dropped_but_short_speech_at_eos_kept() -> None:
    o, _, _ = run(pcm(np.concatenate([speech(600), sil(300), sil(120, -52), sil(400)])))
    assert len(o) == ms(760 + LAG) * 2
    o, _, _ = run(pcm(np.concatenate([speech(600), sil(400), speech(20)])))
    assert o.endswith(pcm(speech(20)))


def test_noise_floor_sets_threshold() -> None:
    x = np.concatenate([sil(200, -55), speech(600), sil(900, -55), speech(400), sil(900, -55)])
    o, _, _ = run(pcm(x))
    assert abs(len(o) / 32 - (200 + 600 + 250 + 400 + 160 + 2 * LAG)) <= 10
    xb = pcm(np.concatenate([sil(200, -40), speech(600), sil(900, -40), speech(400), sil(900, -40)]))
    assert run(xb)[0] == xb


def test_cap_off_keeps_inner_pause() -> None:
    o, _, _ = run(pcm(np.concatenate([speech(500), sil(1000), speech(500), sil(900)])), cap_over_ms=0)
    assert len(o) == ms(500 + 1000 + 500 + 160 + LAG) * 2


def test_keep_250_sets_both_inner_and_end() -> None:
    o, _, _ = run(pcm(np.concatenate([speech(500), sil(900), speech(500), sil(900)])), keep_ms=250)
    assert len(o) == ms(500 + 250 + 500 + 250 + 2 * LAG) * 2


def test_silence_only_passes_as_no_speech() -> None:
    xb = pcm(sil(700, -70))
    o, pc, _ = run(xb)
    assert o == xb and pc.stats.outcome == "no_speech"


def test_unconfirmed_sound_inside_keep_goes_out_with_first_chunk() -> None:
    b = pcm(np.concatenate([speech(50), sil(80), speech(400), sil(900)]))
    chunks = _first_chunks(b)
    _, _, firsts = run(b, chunks)
    assert firsts[0] == chunks[0]


def test_weak_sound_in_pause_head_counts_as_pause() -> None:
    o, _, _ = run(pcm(np.concatenate([speech(500), sil(60), sil(80, -52), sil(200), speech(400), sil(900)])))
    assert len(o) == ms(500 + 340 + 400 + 160 + LAG) * 2


def _random_input(rng: np.random.Generator) -> bytes:
    parts = [sil(int(rng.integers(0, 400)))]
    for _ in range(int(rng.integers(1, 5))):
        parts.append(speech(int(rng.integers(80, 900)), float(rng.uniform(-30, -12))))
        parts.append(sil(int(rng.integers(10, 1300)), float(rng.choice([-90, -70, -60])), rng))
    return pcm(np.concatenate(parts))


def test_random_inputs_keep_speech_and_order_regardless_of_chunking() -> None:
    rng = np.random.default_rng(12)
    for _ in range(200):
        xb = _random_input(rng)
        kw = {"keep_ms": int(rng.choice([160, 200, 250])), "cap_over_ms": int(rng.choice([0, 350, 500]))}
        o, _, _ = run(xb, **kw)
        o2, _, firsts = run(xb, split_rand(xb, rng, 1, 9000), **kw)
        assert o == o2
        assert subseq(o, xb) and loud(o) == loud(xb)
        assert len(o) <= len(xb) and all(len(f) % 2 == 0 for f in firsts)


def test_random_spare_keeps_speech_order_and_even_chunks() -> None:
    rng = np.random.default_rng(13)
    for _ in range(200):
        xb = _random_input(rng)
        seq = [float(rng.choice([0, 0, 50, 200, 1000])) for _ in range(400)]
        o, _, firsts = run(xb, split_rand(xb, rng, 1, 9000), spare=lambda i, seq=seq: seq[i % 400])
        assert subseq(o, xb) and loud(o) == loud(xb)
        assert len(o) <= len(xb) and all(len(f) % 2 == 0 for f in firsts)


def test_spare_keeps_more_pause_and_counts_extra() -> None:
    xb = pcm(np.concatenate([speech(500), sil(900), speech(500), sil(900)]))
    o0, _, _ = run(xb)
    o1, pc1, _ = run(xb, spare=lambda i: 300.0)
    assert len(o1) > len(o0) and pc1.stats.extra_ms > 0 and subseq(o1, xb)


def test_infinite_spare_equals_input() -> None:
    xb = pcm(np.concatenate([speech(500), sil(900), speech(500), sil(900)]))
    o, pc, _ = run(xb, spare=lambda i: 1e9)
    assert o == xb


def test_spare_only_releases_what_was_given_per_push() -> None:
    xb = pcm(np.concatenate([speech(500), sil(900)]))
    chunks = [xb[i : i + 3200] for i in range(0, len(xb), 3200)]
    o, pc, _ = run(xb, chunks, spare=lambda i: 100.0 if i in (7, 8) else 0.0)
    assert len(o) == ms(500 + 160 + LAG + 200) * 2 and pc.stats.extra_ms == 200
    assert subseq(o, xb)


def test_spare_released_in_order_from_held_front() -> None:
    xb = pcm(np.concatenate([speech(500), sil(900)]))
    chunks = [xb[i : i + 3200] for i in range(0, len(xb), 3200)]
    o, _, _ = run(xb, chunks, spare=lambda i: 100.0 if i == 9 else 0.0)
    keep_end = ms(500 + 160 + LAG) * 2
    assert o == xb[: keep_end + ms(100) * 2]


@pytest.mark.parametrize(("eos_spare", "extra"), [(0.0, 0), (50.0, 50), (200.0, 90)])
def test_eos_spare_releases_held_end_up_to_cap(eos_spare, extra) -> None:
    xb = pcm(np.concatenate([speech(500), sil(900)]))
    o, pc, _ = run(xb, spare=lambda i: eos_spare if i == -1 else 0.0)
    assert len(o) == ms(500 + 160 + LAG + extra) * 2 and pc.stats.extra_ms == extra
    assert o == xb[: len(o)]


def test_client_buffer_starts_after_prebuffer_and_counts_starves() -> None:
    cb = ClientBuffer(250)
    cb.on_yield(4480, 0.0)
    assert cb.t0 is None and cb.ahead_ms(10) == 140 and cb.raw_ahead_ms(10) == 140
    cb.on_yield(5120, 70.0)
    assert cb.t0 == 70.0 and cb.ahead_ms(170) == pytest.approx(200)
    cb.on_yield(320, 500.0)
    assert cb.starves == 1 and cb.stall_ms == pytest.approx(130)


def test_client_buffer_raw_ahead_goes_negative_when_dry() -> None:
    cb = ClientBuffer(250)
    cb.on_yield(9600, 0.0)
    assert cb.raw_ahead_ms(100) == pytest.approx(200)
    assert cb.raw_ahead_ms(400) == pytest.approx(-100) and cb.ahead_ms(400) == 0.0
