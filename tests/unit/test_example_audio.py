import importlib.util
from pathlib import Path

import numpy as np
import pytest

from app.services import example_audio as ea

_SR = 16_000
_ROOT = Path(__file__).resolve().parents[2]


def _load_post():
    spec = importlib.util.spec_from_file_location("tts_bench_post_oracle", _ROOT / "scripts" / "tts_bench" / "post.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _silence(sec: float) -> np.ndarray:
    return np.zeros(int(_SR * sec), dtype=np.float32)


def _tone(sec: float, amp: float = 0.3, fade_ms: float = 0.0, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(_SR * sec)) / _SR
    x = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    n = int(_SR * fade_ms / 1000)
    if n:
        x[-n:] *= np.geomspace(1.0, 1e-3, n).astype(np.float32)
    return x


def _flicker(sec: float, spike_dbfs: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(int(_SR * sec)) * 10 ** (-65 / 20)).astype(np.float32)
    for start in range(0, len(x) - 64, _SR // 10):
        x[start : start + 32] = 10 ** (spike_dbfs / 20)
    return x


def _pcm(*parts: np.ndarray) -> bytes:
    return (np.clip(np.concatenate(parts), -1, 1) * 32767).astype("<i2").tobytes()


def _sec(pcm: bytes) -> float:
    return len(pcm) / 2 / _SR


def test_abrupt_end_is_cut_and_faded_end_is_not() -> None:
    assert ea.is_cut(_pcm(_silence(0.3), _tone(1.0), _silence(0.5)))
    assert not ea.is_cut(_pcm(_silence(0.3), _tone(1.0, fade_ms=60), _silence(0.5)))


def test_end_detection_matches_bench_oracle() -> None:
    post = _load_post()
    rng = np.random.default_rng(7)
    for _ in range(40):
        parts = [_silence(rng.uniform(0.0, 0.5))]
        for _ in range(int(rng.integers(1, 4))):
            parts.append(_tone(rng.uniform(0.2, 0.8), amp=rng.uniform(0.05, 0.6), fade_ms=rng.choice([0, 5, 15, 40])))
            parts.append(_silence(rng.uniform(0.05, 0.4)))
        w = np.concatenate(parts)
        ours = [(round(s / _SR, 6), round(d, 6)) for s, _, d in ea._ends(w)]
        theirs = [(round(t, 6), round(d, 6)) for t, d, _ in post.speech_ends(w)]
        assert ours == theirs


def test_trim_keeps_100ms_lead_and_120ms_after_last_speech() -> None:
    pcm = _pcm(_silence(0.8), _tone(1.0, fade_ms=30), _silence(1.0))
    w = ea._samples(pcm)
    onset, last_end = ea._onset(w), ea._ends(w)[-1][0]
    assert ea.trim(pcm) == pcm[(onset - 1600) * 2 : (last_end + 1920) * 2]


def test_onset_ignores_a_blip_shorter_than_three_frames() -> None:
    pcm = _pcm(_silence(0.5), _tone(0.04), _silence(0.5), _tone(1.0, fade_ms=30), _silence(0.3))
    assert ea._onset(ea._samples(pcm)) >= int(1.04 * _SR) - 320


def test_trim_keeps_short_tail_as_is() -> None:
    pcm = _pcm(_silence(0.5), _tone(1.0, fade_ms=30), _silence(0.05))
    out = ea.trim(pcm)
    assert out.endswith(pcm[-int(0.05 * _SR) * 2 :])


def test_trim_keeps_low_level_noise_that_is_loud_relative_to_peak() -> None:
    noisy = _pcm(_silence(0.3), _tone(0.5, amp=0.3, fade_ms=30), _flicker(3.0, -50))
    assert _sec(ea.trim(noisy)) > 3.0


def test_trim_leaves_silence_only_input_alone() -> None:
    pcm = _pcm(_silence(1.0))
    assert ea.trim(pcm) == pcm


@pytest.mark.parametrize(
    ("text", "expected"),
    [("배준하입니다.", 6), ("KTX 3시, 네!", 6), ("  ...?!  ", 0)],
)
def test_syllables_count_hangul_digits_and_latin(text: str, expected: int) -> None:
    assert ea.syllables(text) == expected


def test_too_long_flags_speech_followed_by_noise_flicker() -> None:
    clean = _pcm(_silence(0.2), _tone(0.5, fade_ms=30), _silence(0.5))
    noisy = _pcm(_silence(0.2), _tone(0.5, fade_ms=30), _flicker(5.0, -50))
    assert not ea.too_long(clean, "배준하입니다.")
    assert ea.too_long(noisy, "배준하입니다.")


def test_too_long_has_a_floor_of_one_and_a_half_seconds() -> None:
    assert not ea.too_long(_pcm(_silence(0.2), _tone(1.4, fade_ms=30), _silence(0.3)), "네.")
    assert ea.too_long(_pcm(_silence(0.2), _tone(1.7, fade_ms=30), _silence(0.3)), "네.")


def test_match_level_hits_target_level() -> None:
    out, gain = ea.match_level(_pcm(_silence(0.2), _tone(1.0, amp=0.05, fade_ms=30)), -22.5)
    assert ea.speech_db(out) == pytest.approx(-22.5, abs=0.05)
    assert gain > 0


def test_match_level_caps_peak_at_minus_one_dbfs() -> None:
    spike = _tone(1.0, amp=0.02)
    spike[8000:8010] = 0.9
    out, _ = ea.match_level(_pcm(spike), -10.0)
    peak = np.abs(np.frombuffer(out, dtype="<i2")).max() / 32767
    assert peak <= 10 ** (-1 / 20) + 1e-4


def test_match_level_leaves_silence_unchanged() -> None:
    pcm = _pcm(_silence(1.0))
    out, gain = ea.match_level(pcm, -16.9)
    assert out == pcm and gain == 0.0


def test_match_level_quantizes_like_the_listening_kit() -> None:
    pcm = _pcm(_silence(0.1), _tone(0.8, amp=0.1))
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
    gain = 10 ** ((-16.94749568714168 - ea.speech_db(pcm)) / 20)
    expected = (np.clip(x * gain, -1, 1) * 32767).astype("<i2").tobytes()
    wide = (np.clip(x.astype(np.float64) * gain, -1, 1) * 32767).astype("<i2").tobytes()
    assert wide != expected
    assert ea.match_level(pcm, -16.94749568714168)[0] == expected


def _render(cut: bool, long: bool, decay: float | None, span: float | None, tag: bytes) -> ea.Render:
    return ea.Render(pcm=tag, decay_ms=decay, span_s=span, cut=cut, long=long)


def test_pick_takes_first_acceptable_render() -> None:
    renders = [
        _render(True, False, 5, 1.0, b"a"),
        _render(False, False, 20, 1.0, b"b"),
        _render(False, False, 90, 1.0, b"c"),
    ]
    assert ea.pick(renders).pcm == b"b"


def test_pick_all_cut_takes_longest_decay_among_fitting() -> None:
    renders = [
        _render(True, False, 3, 1.0, b"a"),
        _render(True, True, 9, 9.0, b"b"),
        _render(True, False, 8, 1.0, b"c"),
    ]
    assert ea.pick(renders).pcm == b"c"


def test_pick_tie_keeps_the_earlier_render() -> None:
    assert ea.pick([_render(True, False, 4, 1.0, b"a"), _render(True, False, 4, 1.0, b"b")]).pcm == b"a"


def test_pick_all_long_takes_shortest_span() -> None:
    renders = [
        _render(False, True, 30, 6.0, b"a"),
        _render(True, True, 50, 4.0, b"b"),
        _render(False, True, 30, 5.0, b"c"),
    ]
    assert ea.pick(renders).pcm == b"b"


def test_tail_after_rms_measures_noise_left_after_speech() -> None:
    assert ea.tail_after_rms_s(_pcm(_silence(0.2), _tone(0.5, fade_ms=30), _silence(0.5))) < 0.05
    assert ea.tail_after_rms_s(_pcm(_silence(0.2), _tone(0.5, fade_ms=30), _flicker(2.0, -50))) > 1.5


def test_finish_trims_and_matches_the_chosen_render() -> None:
    raw = _pcm(_silence(0.8), _tone(1.0, amp=0.05, fade_ms=30), _silence(1.0))
    render = ea.evaluate(raw, "안녕하세요")
    pcm, gain, chosen = ea.finish([render], -16.9)
    assert chosen is render
    assert _sec(pcm) < _sec(raw) - 1.0
    assert ea.speech_db(pcm) == pytest.approx(-16.9, abs=0.05)


def test_join_turns_puts_400ms_silence_between() -> None:
    joined = ea.join_turns([b"\x01\x00", b"\x02\x00"])
    assert joined == b"\x01\x00" + b"\x00" * (_SR * 400 // 1000 * 2) + b"\x02\x00"


def test_too_long_threshold_is_point_six_seconds_per_syllable() -> None:
    text = "가나다라마바사아자차"
    assert ea.too_long(_pcm(_silence(0.2), _tone(6.5, fade_ms=30), _silence(0.3)), text)
    assert not ea.too_long(_pcm(_silence(0.2), _tone(5.5, fade_ms=30), _silence(0.3)), text)


def test_level_targets_are_the_listened_medians() -> None:
    assert ea.LEVEL_DB == {"ai": -16.94749568714168, "user": -22.544892951720513}
