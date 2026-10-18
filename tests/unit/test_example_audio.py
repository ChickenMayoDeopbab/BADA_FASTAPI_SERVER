import asyncio

import numpy as np
import pytest

from app.services.example_audio import (
    MAX_ATTEMPTS,
    SR,
    TAIL_PHRASE,
    clean_turn,
    cut_tail_phrase,
    end_decay_ms,
    is_cut,
    trim_lead,
)


def tone(seconds: float, decay_ms: float = 40.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    w = amp * np.sin(2 * np.pi * 220 * t)
    if decay_ms <= 0:
        return w
    rt = np.arange(int(SR * decay_ms * 3 / 1000)) / SR
    tail = amp * np.sin(2 * np.pi * 220 * (seconds + rt)) * np.exp(-rt * np.log(10) / (decay_ms / 1000))
    return np.concatenate([w, tail])


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SR * seconds))


def pcm(*parts: np.ndarray) -> bytes:
    return (np.concatenate(parts) * 32767).astype("<i2").tobytes()


def seconds(data: bytes) -> float:
    return len(data) / 2 / SR


def with_tail(speech_s: float = 1.5, decay_ms: float = 40.0, tail_s: float = 0.3) -> bytes:
    return pcm(tone(speech_s, decay_ms), silence(0.3), tone(tail_s, 30.0), silence(0.1))


def test_natural_decay_is_not_cut_and_hard_stop_is():
    assert not is_cut(pcm(tone(1.0, 40.0), silence(0.2)))
    assert end_decay_ms(pcm(tone(1.0, 40.0), silence(0.2))) == pytest.approx(40.0, abs=4.0)
    assert is_cut(pcm(tone(1.0, 0.0), silence(0.2)))
    assert is_cut(pcm(silence(0.2), tone(1.0, 0.0)))


def test_silence_has_no_end():
    assert end_decay_ms(pcm(silence(0.5))) is None and not is_cut(pcm(silence(0.5)))


def test_trim_lead_keeps_100ms_before_speech():
    trimmed = trim_lead(pcm(silence(2.0), tone(1.0)))
    assert seconds(trimmed) == pytest.approx(seconds(pcm(tone(1.0))) + 0.1, abs=0.03)
    assert trim_lead(pcm(tone(1.0))) == pcm(tone(1.0))
    assert trim_lead(pcm(silence(0.5))) == pcm(silence(0.5))


def test_cut_tail_phrase_keeps_speech_and_120ms():
    got = cut_tail_phrase(with_tail(), "예약하려고 전화드렸어요.")
    assert got is not None
    audio, decay = got
    assert seconds(audio) == pytest.approx(1.5 + 0.09 + 0.12, abs=0.02)
    assert decay == pytest.approx(40.0, abs=4.0) and not is_cut(audio)


@pytest.mark.parametrize(
    ("data", "text"),
    [
        (pcm(tone(1.5), silence(0.1)), "예약하려고 전화드렸어요."),
        (with_tail(tail_s=1.0), "예약하려고 전화드렸어요."),
        (pcm(tone(1.5), silence(0.3), tone(0.004, 0.0), silence(0.1)), "예약하려고 전화드렸어요."),
        (with_tail(), "예약하려고 전화드렸어요 이번 주 금요일 저녁 일곱 시로 네 명 자리 부탁드립니다."),
    ],
    ids=["no-pause", "tail-too-long", "click-after-pause", "tail-longer-than-four-syllables"],
)
def test_cut_tail_phrase_refuses_when_tail_is_not_one_word(data, text):
    assert cut_tail_phrase(data, text) is None


class _Synth:
    def __init__(self, renders: list[bytes], plain: bytes = b"") -> None:
        self.renders = list(renders)
        self.plain = plain or pcm(tone(1.0))
        self.texts: list[str] = []

    async def __call__(self, text: str) -> bytes:
        self.texts.append(text)
        return self.renders.pop(0) if text.endswith(TAIL_PHRASE) else self.plain


def test_clean_turn_retries_only_until_the_end_is_clean():
    synth = _Synth([with_tail(decay_ms=0.0), with_tail()])
    got = asyncio.run(clean_turn(synth, "네, 바다레스토랑입니다."))
    assert synth.texts == ["네, 바다레스토랑입니다." + TAIL_PHRASE] * 2
    assert (got.attempts, got.cut) == (2, False) and not is_cut(got.pcm)


def test_clean_turn_keeps_the_least_abrupt_render_when_every_end_is_cut():
    renders = [with_tail(1.2, 2.0), with_tail(1.3, 8.0), with_tail(1.4, 4.0)]
    renders += [with_tail(1.5, 0.0)] * (MAX_ATTEMPTS - 3)
    got = asyncio.run(clean_turn(_Synth(renders), "네, 바다레스토랑입니다."))
    assert (got.attempts, got.cut) == (MAX_ATTEMPTS, True)
    assert seconds(got.pcm) == pytest.approx(1.3 + 0.018 + 0.12, abs=0.02)


def test_clean_turn_falls_back_to_plain_text_when_tail_never_separates():
    synth = _Synth([pcm(tone(1.5), silence(0.1))] * MAX_ATTEMPTS, plain=pcm(tone(1.0, 0.0)))
    got = asyncio.run(clean_turn(synth, "네, 바다레스토랑입니다."))
    assert synth.texts[-1] == "네, 바다레스토랑입니다." and len(synth.texts) == MAX_ATTEMPTS + 1
    assert (got.attempts, got.cut) == (MAX_ATTEMPTS + 1, True)


def test_clean_turn_skips_tail_phrase_for_text_near_worker_limit():
    synth = _Synth([], plain=pcm(tone(1.0)))
    got = asyncio.run(clean_turn(synth, "가" * 296))
    assert synth.texts == ["가" * 296] and (got.attempts, got.cut) == (1, False)


def test_cut_tail_phrase_absolute_limit_holds_for_one_syllable_text():
    assert cut_tail_phrase(with_tail(tail_s=1.0), "네.") is None


def test_cut_tail_phrase_measures_speech_rate_without_lead_silence():
    data = pcm(silence(3.0), tone(1.0), silence(0.3), tone(0.43, 30.0), silence(0.1))
    assert cut_tail_phrase(data, "바다레스토랑입니다요") is None


def test_clean_turn_trims_lead_silence_on_both_paths():
    lead = pcm(silence(1.0), tone(1.5), silence(0.3), tone(0.3, 30.0), silence(0.1))
    got = asyncio.run(clean_turn(_Synth([lead]), "네, 바다레스토랑입니다."))
    assert (got.attempts, got.cut) == (1, False) and seconds(got.pcm) < 1.5 + 0.09 + 0.12 + 0.12
    plain = asyncio.run(clean_turn(_Synth([], plain=pcm(silence(1.0), tone(1.0))), "가" * 296))
    assert seconds(plain.pcm) == pytest.approx(seconds(pcm(tone(1.0))) + 0.1, abs=0.03)


def test_cut_tail_phrase_picks_the_pause_right_before_the_dummy_in_a_multi_phrase_turn():
    data = pcm(tone(0.4), silence(0.25), tone(1.2), silence(0.3), tone(0.3, 30.0), silence(0.1))
    got = cut_tail_phrase(data, "네, 바다레스토랑입니다.")
    assert got is not None
    assert seconds(got[0]) == pytest.approx(seconds(pcm(tone(0.4), silence(0.25))) + 1.2 + 0.09 + 0.12, abs=0.02)


def test_cut_tail_phrase_refuses_when_a_stray_sound_follows_the_dummy():
    data = pcm(tone(1.5), silence(0.3), tone(0.2, 30.0), silence(0.25), tone(0.1, 0.0), silence(0.1))
    assert cut_tail_phrase(data, "예약하려고 전화드렸어요.") is None


def test_cut_tail_phrase_refuses_a_tail_that_holds_two_words():
    data = pcm(tone(1.5), silence(0.3), tone(0.15, 0.0), silence(0.1), tone(0.2, 30.0), silence(0.1))
    assert cut_tail_phrase(data, "예약하려고 전화드렸어요.") is None


def test_cut_tail_phrase_ignores_a_click_inside_the_pause():
    data = pcm(tone(1.5), silence(0.2), tone(0.002, 0.0), silence(0.2), tone(0.3, 30.0), silence(0.1))
    got = cut_tail_phrase(data, "예약하려고 전화드렸어요.")
    assert got is not None and seconds(got[0]) < 1.5 + 0.09 + 0.12 + 0.03


def test_clean_turn_trims_lead_silence_on_the_least_abrupt_path():
    renders = [pcm(silence(1.5), tone(1.2, 2.0), silence(0.3), tone(0.3, 30.0), silence(0.1))] * MAX_ATTEMPTS
    got = asyncio.run(clean_turn(_Synth(renders), "네, 바다레스토랑입니다."))
    assert got.cut and seconds(got.pcm) == pytest.approx(0.1 + 1.2 + 0.006 + 0.12, abs=0.02)


def test_trim_lead_treats_a_noise_floor_as_silence():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 10 ** (-60 / 20), int(SR * 2.0))
    trimmed = trim_lead(pcm(noise, tone(1.0)))
    assert seconds(trimmed) == pytest.approx(seconds(pcm(tone(1.0))) + 0.1, abs=0.03)


def test_trim_lead_returns_whole_samples():
    assert len(trim_lead(pcm(silence(0.5), tone(0.5)) + b"\x01")) % 2 == 0


def test_cut_tail_phrase_keeps_a_short_real_sound_that_is_not_a_click():
    data = pcm(tone(1.5), silence(0.3), tone(0.04, 0.0), silence(0.3), tone(0.3, 30.0), silence(0.1))
    got = cut_tail_phrase(data, "예약하려고 전화드렸어요.")
    assert got is not None and seconds(got[0]) == pytest.approx(seconds(pcm(tone(1.5))) + 0.3 + 0.04 + 0.12, abs=0.02)


def test_empty_audio_is_handled():
    assert end_decay_ms(b"") is None and not is_cut(b"")
    assert trim_lead(b"") == b"" and cut_tail_phrase(b"", "네.") is None


def test_speech_from_the_first_sample_still_reports_its_end():
    assert end_decay_ms(pcm(tone(1.0, 40.0), silence(0.2))) == pytest.approx(40.0, abs=4.0)
    assert is_cut(pcm(tone(1.0, 0.0), silence(0.2)))


def test_a_quiet_breath_before_the_first_loud_sound_is_not_an_end():
    from app.services.example_audio import _ends, _samples

    breath = tone(0.2, 0.0, amp=0.3 * 10 ** (-30 / 20))
    data = pcm(breath, silence(0.3), tone(1.5), silence(0.3), tone(0.3, 30.0), silence(0.1))
    assert len(_ends(_samples(data))) == 2
    got = cut_tail_phrase(data, "예약하려고 전화드렸어요.")
    assert got is not None and seconds(got[0]) == pytest.approx(0.2 + 0.3 + 1.5 + 0.09 + 0.12, abs=0.02)


def test_cut_tail_phrase_refuses_a_hesitation_kept_before_the_dummy():
    murmur = tone(0.6, 30.0, amp=0.3 * 10 ** (-20 / 20))
    data = pcm(tone(1.5), silence(2.4), murmur, silence(0.3), tone(0.3, 30.0), silence(0.1))
    assert cut_tail_phrase(data, "예약하려고 전화드렸어요.") is None


def test_cut_tail_phrase_keeps_a_normal_sentence_pause():
    data = pcm(tone(1.0), silence(1.3), tone(1.0), silence(0.3), tone(0.3, 30.0), silence(0.1))
    got = cut_tail_phrase(data, "네, 예약하려고 전화드렸어요. 이번 주 금요일이요.")
    expected = seconds(pcm(tone(1.0))) + 1.3 + 1.0 + 0.09 + 0.12
    assert got is not None and seconds(got[0]) == pytest.approx(expected, abs=0.02)


def test_cut_tail_phrase_does_not_count_lead_silence_as_a_pause():
    data = pcm(silence(2.5), tone(1.5), silence(0.3), tone(0.3, 30.0), silence(0.1))
    got = cut_tail_phrase(data, "예약하려고 전화드렸어요.")
    assert got is not None and seconds(got[0]) == pytest.approx(2.5 + 1.5 + 0.09 + 0.12, abs=0.02)


def test_clean_turn_tries_eight_times_before_giving_up():
    assert MAX_ATTEMPTS == 8, "user 참조는 시도당 성공 ~34%(10/19 벤치) — 4번이면 19% 가 잘린 채 남는다"


def test_cut_tail_phrase_refuses_a_dummy_merged_into_the_speech():
    data = pcm(tone(1.5, 0.0), silence(0.12), tone(0.29, 30.0), silence(0.3), tone(0.1, 30.0), silence(0.1))
    assert cut_tail_phrase(data, "일요일 저녁 7시로 바꿀 수 있을까요?") is None


def test_cut_tail_phrase_keeps_a_long_final_phrase_after_a_short_gap():
    data = pcm(tone(1.0, 0.0), silence(0.12), tone(0.8), silence(0.3), tone(0.3, 30.0), silence(0.1))
    got = cut_tail_phrase(data, "네, 예약하려고 전화드렸어요.")
    assert got is not None and seconds(got[0]) == pytest.approx(1.0 + 0.12 + 0.8 + 0.09 + 0.12, abs=0.02)


def test_cut_tail_phrase_judges_the_merged_dummy_by_the_last_gap():
    data = pcm(tone(1.0, 0.0), silence(0.12), tone(0.8, 0.0), silence(0.12), tone(0.29, 30.0), silence(0.3),
               tone(0.1, 30.0), silence(0.1))
    assert cut_tail_phrase(data, "네, 예약하려고 전화드렸어요. 금요일이요.") is None


def test_cut_tail_phrase_refuses_a_short_merged_dummy():
    data = pcm(tone(1.5, 0.0), silence(0.12), tone(0.15, 0.0), silence(0.3), tone(0.1, 30.0), silence(0.1))
    assert cut_tail_phrase(data, "일요일 저녁 7시로 바꿀 수 있을까요?") is None


def test_cut_tail_phrase_does_not_count_lead_silence_as_a_gap_before_a_short_turn():
    data = pcm(silence(0.3), tone(0.3), silence(0.3), tone(0.3, 30.0), silence(0.1))
    got = cut_tail_phrase(data, "네.")
    assert got is not None and seconds(got[0]) == pytest.approx(0.3 + 0.3 + 0.09 + 0.12, abs=0.02)
