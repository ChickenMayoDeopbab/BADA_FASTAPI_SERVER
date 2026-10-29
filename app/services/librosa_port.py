# 원본: librosa 0.10.2.post1 의 core/pitch.py·feature/spectral.py·util/utils.py 에서 옮김
#
# ISC License
#
# Copyright (c) 2013--2023, librosa development team.
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import as_strided


def _frame(x: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if x.shape[-1] < frame_length:
        raise ValueError(f"Input is too short (n={x.shape[-1]}) for frame_length={frame_length}")
    xw = as_strided(
        x,
        strides=x.strides + (x.strides[-1],),
        shape=(x.shape[-1] - frame_length + 1, frame_length),
        writeable=False,
    )
    return np.moveaxis(xw, -1, -2)[..., ::hop_length]


def _cumulative_mean_normalized_difference(
    y_frames: np.ndarray, frame_length: int, win_length: int, min_period: int, max_period: int
) -> np.ndarray:
    a = np.fft.rfft(y_frames, frame_length, axis=-2)
    b = np.fft.rfft(y_frames[..., win_length:0:-1, :], frame_length, axis=-2)
    acf_frames = np.fft.irfft(a * b, frame_length, axis=-2)[..., win_length:, :]
    acf_frames[np.abs(acf_frames) < 1e-6] = 0

    energy_frames = np.cumsum(y_frames**2, axis=-2)
    energy_frames = energy_frames[..., win_length:, :] - energy_frames[..., :-win_length, :]
    energy_frames[np.abs(energy_frames) < 1e-6] = 0

    yin_frames = energy_frames[..., :1, :] + energy_frames - 2 * acf_frames

    yin_numerator = yin_frames[..., min_period : max_period + 1, :]
    tau_range = np.arange(1, max_period + 1).reshape(-1, 1)
    cumulative_mean = np.cumsum(yin_frames[..., 1 : max_period + 1, :], axis=-2) / tau_range
    yin_denominator = cumulative_mean[..., min_period - 1 : max_period, :]
    return yin_numerator / (yin_denominator + np.finfo(yin_denominator.dtype).tiny)


def _parabolic_interpolation(x: np.ndarray) -> np.ndarray:
    shifts = np.zeros_like(x)
    prev, cur, nxt = x[:-2], x[1:-1], x[2:]
    a = (nxt + prev).astype(np.float64) - 2 * cur
    b = (nxt - prev).astype(np.float64) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        shifts[1:-1] = np.where(np.abs(b) >= np.abs(a), 0, -b / a)
    return shifts


def _localmin(x: np.ndarray) -> np.ndarray:
    lmin = np.zeros(x.shape, dtype=bool)
    lmin[1:-1] = (x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:])
    lmin[-1] = x[-1] < x[-2]
    return lmin


def yin(
    y: np.ndarray,
    *,
    fmin: float,
    fmax: float,
    sr: float,
    frame_length: int,
    hop_length: int,
    trough_threshold: float = 0.1,
    center: bool = True,
) -> np.ndarray:
    """librosa.yin(win_length=None, pad_mode="constant") 과 같은 값"""
    if not np.issubdtype(y.dtype, np.floating):
        raise ValueError("Audio data must be floating-point")
    if not np.isfinite(y).all():
        raise ValueError("Audio buffer is not finite everywhere")

    win_length = frame_length // 2
    if center:
        y = np.pad(y, (frame_length // 2, frame_length // 2), mode="constant")
    y_frames = _frame(y, frame_length, hop_length)

    min_period = int(np.floor(sr / fmax))
    max_period = min(int(np.ceil(sr / fmin)), frame_length - win_length - 1)
    yin_frames = _cumulative_mean_normalized_difference(y_frames, frame_length, win_length, min_period, max_period)
    parabolic_shifts = _parabolic_interpolation(yin_frames)

    is_trough = _localmin(yin_frames)
    is_trough[0] = yin_frames[0] < yin_frames[1]
    is_threshold_trough = np.logical_and(is_trough, yin_frames < trough_threshold)

    global_min = np.argmin(yin_frames, axis=-2)
    yin_period = np.argmax(is_threshold_trough, axis=-2)
    no_trough_below_threshold = np.all(~is_threshold_trough, axis=-2)
    yin_period[no_trough_below_threshold] = global_min[no_trough_below_threshold]

    shift = np.take_along_axis(parabolic_shifts, yin_period[np.newaxis, :], axis=-2)[0]
    return sr / (min_period + yin_period + shift)


def rms(y: np.ndarray, *, frame_length: int, hop_length: int) -> np.ndarray:
    """librosa.feature.rms(y=y, center=True)[0] 과 같은 값"""
    y = np.pad(y, (frame_length // 2, frame_length // 2), mode="constant")
    x = _frame(y, frame_length, hop_length)
    power = np.mean(np.power(x, 2, dtype=np.float32), axis=-2, keepdims=True)
    return np.sqrt(power)[0]
