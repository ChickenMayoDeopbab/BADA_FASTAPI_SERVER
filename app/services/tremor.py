from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np
from scipy.signal import butter, medfilt, sosfiltfilt


@dataclass
class TremorConfig:
    sample_rate: int = 16000
    pcm_dtype: str = "int16"

    # F0 추출 (하이브리드: 에너지 발성마스크 + yin + 옥타브보정 + median필터)
    fmin: float = 70.0
    fmax: float = 400.0
    contour_fs: float = 100.0
    energy_active_frac: float = 0.10
    energy_ref_percentile: float = 95.0
    octave_ratio: float = 1.5
    median_kernel: int = 5

    # 떨림 대역
    tremor_lo: float = 4.0
    tremor_hi: float = 8.0
    intonation_hp: float = 3.0

    # 지속발성 한정 (짧은 연결발화 모음은 떨림 측정 불가)
    min_sustained_sec: float = 1.2
    # 한국어 파열음이 만드는 짧은 무성 구간은 이어붙인다. 이것보다 짧으면 같은 발성으로 본다.
    sustained_gap_sec: float = 0.20

    # 슬라이딩 윈도우
    win_sec: float = 0.6
    hop_sec: float = 0.1
    min_voiced_frac: float = 0.8

    # 판정 임계값 (★ 실음성 캘리브레이션 대상 — 현재 잠정값)
    amp_thresh_cents: float = 10.0
    concentration_min: float = 0.60

    # 에피소드 묶기
    merge_gap_sec: float = 0.15
    min_episode_sec: float = 0.5

    min_silence_sec: float = 1.5  # 이보다 긴 무음 = '공백' → 발화 끊김


@dataclass
class TremorResult:
    shake_count: int
    episodes: list
    voiced_sec: float
    sustained_sec: float = 0.0
    debug_windows: list = field(default_factory=list)
    good_candidates: list = field(default_factory=list)
    sustained_spans: list = field(default_factory=list)
    voiced_spans: list = field(default_factory=list)


class TremorAnalyzer:
    def __init__(self, config: TremorConfig | None = None):
        self.config = config or TremorConfig()
        self._sos_band = butter(
            4, [self.config.tremor_lo, self.config.tremor_hi],
            btype="bandpass", fs=self.config.contour_fs, output="sos",
        )
        self._sos_hp = butter(
            4, self.config.intonation_hp,
            btype="highpass", fs=self.config.contour_fs, output="sos",
        )

    def _pcm_to_float(self, pcm: bytes) -> np.ndarray:
        if len(pcm) % 2 != 0:
            pcm = pcm[:-1]
        x = np.frombuffer(pcm, dtype=self.config.pcm_dtype).astype(np.float32)
        return x / np.iinfo(self.config.pcm_dtype).max

    def _extract_f0_cents(self, y: np.ndarray):
        config = self.config
        hop = max(1, int(round(config.sample_rate / config.contour_fs)))
        f0 = librosa.yin(y, fmin=config.fmin, fmax=config.fmax,
                         sr=config.sample_rate, hop_length=hop)
        rms = librosa.feature.rms(y=y, frame_length=4 * hop, hop_length=hop)[0]
        m = min(len(f0), len(rms))
        f0, rms = f0[:m], rms[:m]
        if m < 2 or rms.max() <= 0:
            return None, None
        loud = rms[rms > 0]
        ref = float(np.percentile(loud, config.energy_ref_percentile)) if loud.size else 0.0
        active = rms > config.energy_active_frac * ref
        if active.sum() < 2:
            return None, None
        f0 = self._octave_fix(f0)
        if config.median_kernel > 1:
            f0 = medfilt(f0, config.median_kernel)
        ref = float(np.median(f0[active]))
        f0_cents = 1200.0 * np.log2(np.clip(f0, 1e-6, None) / ref)
        return f0_cents.astype(np.float64), active

    def _octave_fix(self, f0: np.ndarray) -> np.ndarray:
        r = self.config.octave_ratio
        out = f0.copy()
        for i in range(len(out)):
            lo = max(0, i - 10)
            ref = np.median(out[lo:i + 1]) if i > 0 else out[i]
            for _ in range(2):
                if out[i] > r * ref:
                    out[i] /= 2
                elif out[i] < ref / r:
                    out[i] *= 2
        return out

    def _sustained_mask(self, voiced: np.ndarray) -> np.ndarray:
        config = self.config
        fs = config.contour_fs
        gap = int(round(config.sustained_gap_sec * fs))
        minlen = int(round(config.min_sustained_sec * fs))
        v = voiced.copy()
        n = len(v)
        i = 0
        while i < n:  # 짧은 끊김 메우기
            if not v[i]:
                j = i
                while j < n and not v[j]:
                    j += 1
                if i > 0 and j < n and (j - i) <= gap:
                    v[i:j] = True
                i = j
            else:
                i += 1
        mask = np.zeros_like(v)
        i = 0
        while i < n:  # 충분히 긴 런만 남기기
            if v[i]:
                j = i
                while j < n and v[j]:
                    j += 1
                if (j - i) >= minlen:
                    mask[i:j] = True
                i = j
            else:
                i += 1
        return mask

    def _speaking_spans(self, voiced: np.ndarray) -> list:
        """발화 구간을 (시작초, 끝초) 리스트로. min_silence 이상 무음에서만 끊는다."""
        config = self.config
        fs = config.contour_fs
        bridge = int(round(config.min_silence_sec * fs))  # 이보다 짧은 무음은 이어붙임
        v = voiced.copy()
        n = len(v)
        i = 0
        while i < n:  # 짧은 무음(< min_silence) 메우기
            if not v[i]:
                j = i
                while j < n and not v[j]:
                    j += 1
                if i > 0 and j < n and (j - i) < bridge:
                    v[i:j] = True
                i = j
            else:
                i += 1
        return self._mask_spans(v, fs)

    @staticmethod
    def _mask_spans(mask, fs: float) -> list:
        """True 런을 (시작초, 끝초) 리스트로."""
        spans = []
        n = len(mask)
        i = 0
        while i < n:
            if mask[i]:
                j = i
                while j < n and mask[j]:
                    j += 1
                spans.append((i / fs, j / fs))
                i = j
            else:
                i += 1
        return spans

    @staticmethod
    def _subtract(spans, holes):
        """spans 각 구간에서 holes(떨림)와 겹치는 부분을 빼낸 나머지."""
        holes = sorted(holes)
        out = []
        for s, e in spans:
            cur = s
            for hs, he in holes:
                if he <= cur or hs >= e:   # hole이 현재 구간 밖
                    continue
                if hs > cur:               # hole 앞에 남는 발화
                    out.append((cur, min(hs, e)))
                cur = max(cur, he)         # 커서를 hole 끝으로
                if cur >= e:
                    break
            if cur < e:                    # 뒤에 남는 발화
                out.append((cur, e))
        return out

    def analyze(self, pcm: bytes) -> TremorResult:
        config = self.config
        y = self._pcm_to_float(pcm)
        f0_cents, voiced = self._extract_f0_cents(y)
        if f0_cents is None:
            return TremorResult(0, [], 0.0)

        fs = config.contour_fs
        sustained = self._sustained_mask(voiced)
        band = sosfiltfilt(self._sos_band, f0_cents)
        hp = sosfiltfilt(self._sos_hp, f0_cents)

        win = int(round(config.win_sec * fs))
        hop = max(1, int(round(config.hop_sec * fs)))
        freqs = np.fft.rfftfreq(win, d=1.0 / fs)
        b_lo, b_hi = config.tremor_lo, config.tremor_hi

        on_flags, centers, dbg = [], [], []
        for start in range(0, len(f0_cents) - win + 1, hop):
            sl = slice(start, start + win)
            sfrac = sustained[sl].mean()
            t_center = (start + win / 2) / fs
            amp = float(np.sqrt(np.mean(band[sl] ** 2)))

            seg = hp[sl] * np.hanning(win)
            psd = np.abs(np.fft.rfft(seg)) ** 2
            in_band = (freqs >= b_lo) & (freqs <= b_hi)
            ref_band = (freqs >= config.intonation_hp) & (freqs <= 20.0)
            denom = psd[ref_band].sum() + 1e-12
            concentration = float(psd[in_band].sum() / denom)
            peak_freq = float(freqs[in_band][np.argmax(psd[in_band])]) if in_band.any() else 0.0

            is_on = (
                sfrac >= config.min_voiced_frac
                and amp >= config.amp_thresh_cents
                and concentration >= config.concentration_min
                and b_lo <= peak_freq <= b_hi
            )
            on_flags.append(is_on)
            centers.append(t_center)
            dbg.append(dict(t=round(t_center, 3), sfrac=round(sfrac, 2),
                            amp=round(amp, 1), conc=round(concentration, 2),
                            fpk=round(peak_freq, 2), on=is_on))

        episodes = self._group_episodes(on_flags, centers, dbg)
        spans = self._speaking_spans(voiced)
        tremor = [(ep[0], ep[1]) for ep in episodes]
        good_candidates = self._subtract(spans, tremor)
        return TremorResult(
            len(episodes), episodes,
            voiced_sec=float(voiced.sum()) / fs,
            sustained_sec=float(sustained.sum()) / fs,
            debug_windows=dbg,
            good_candidates=[(round(s, 2), round(e, 2)) for s, e in good_candidates],
            sustained_spans=[
                (round(s, 3), round(e, 3)) for s, e in self._mask_spans(sustained, fs)
            ],
            voiced_spans=[
                (round(s, 3), round(e, 3))
                for s, e in self._mask_spans(voiced, fs)
            ],
        )

    def _group_episodes(self, on_flags, centers, dbg):
        config = self.config
        episodes, run, last_on_t = [], [], None
        for is_on, t, d in zip(on_flags, centers, dbg, strict=True):
            if is_on:
                if last_on_t is not None and (t - last_on_t) > config.merge_gap_sec:
                    episodes += self._flush(run, config)
                    run = []
                run.append(d)
                last_on_t = t
        episodes += self._flush(run, config)
        return episodes

    @staticmethod
    def _flush(run, config):
        if not run:
            return []
        start = run[0]["t"] - config.win_sec / 2
        end = run[-1]["t"] + config.win_sec / 2
        if (end - start) < config.min_episode_sec:
            return []
        peak = float(np.mean([r["fpk"] for r in run]))
        amp = float(np.mean([r["amp"] for r in run]))
        return [(round(start, 3), round(end, 3), round(peak, 2), round(amp, 1))]
