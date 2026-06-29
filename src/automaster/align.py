"""Phase 1 — temporal alignment of before/after pairs.

The manual master and its source share program content but may differ in
start offset (edits, handles) and sample rate. We estimate the integer
sample lag by cross-correlation of mono mixdowns, then trim both to their
common region at the *before* sample rate.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve, resample_poly
from math import gcd


def resample_to(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample (samples,) or (samples, ch) from sr_in to sr_out."""
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float64)
    g = gcd(sr_in, sr_out)
    up, down = sr_out // g, sr_in // g
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return resample_poly(x, up, down)
    return np.stack([resample_poly(x[:, c], up, down) for c in range(x.shape[1])], axis=1)


def _mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else np.asarray(x, dtype=np.float64)


def estimate_offset(before: np.ndarray, after: np.ndarray) -> int:
    """Estimate the lag (in samples) by which ``after`` is delayed relative
    to ``before``. Positive means ``after`` starts later.

    Uses normalised cross-correlation via FFT. Both inputs must already be
    at the same sample rate.
    """
    a = _mono(before)
    b = _mono(after)
    a = a - a.mean()
    b = b - b.mean()
    # Cross-correlation = convolution with the reversed kernel.
    corr = fftconvolve(b, a[::-1], mode="full")
    lag = np.argmax(corr) - (len(a) - 1)
    return int(lag)


def align(before, after, sr_before: int, sr_after: int):
    """Align a before/after pair.

    Resamples ``after`` to ``sr_before``, estimates the lag, and trims both
    to the overlapping region. Returns ``(before_aligned, after_aligned, sr)``
    with identical length.
    """
    before = np.asarray(before, dtype=np.float64)
    after = resample_to(after, sr_after, sr_before)
    sr = sr_before

    lag = estimate_offset(before, after)

    # Shift `after` back by `lag` to line up with `before`, then crop both
    # to the common span.
    if lag >= 0:
        a = after[lag:]
        b = before
    else:
        a = after
        b = before[-lag:]
    n = min(len(b), len(a))
    b = b[:n]
    a = a[:n]
    return b, a, sr
