"""Phase 0 — measurement primitives (ITU-R BS.1770 / EBU R128 / Tech 3342).

All public functions accept either mono ``(samples,)`` or multichannel
``(samples, channels)`` float arrays and return scalars in LU / LUFS / dBTP.

Design notes
------------
* Integrated LUFS and the gating come from :mod:`pyloudnorm` (an established
  BS.1770-4 implementation).
* ``pyloudnorm`` provides neither true-peak nor loudness range, so both are
  implemented here: TP via polyphase oversampling, LRA per EBU Tech 3342.
* The K-weighting filter coefficients are pulled straight from a
  ``pyloudnorm.Meter`` so the short-term trajectory and LRA stay consistent
  with the integrated measurement.
"""
from __future__ import annotations

import numpy as np
import pyloudnorm as pyln
from scipy.signal import lfilter, resample_poly

# Channel weighting from BS.1770 (L, R, C, Ls, Rs).
_CHANNEL_GAINS = np.array([1.0, 1.0, 1.0, 1.41, 1.41])
_ABS_GATE = -70.0  # LUFS absolute gate
_OFFSET = -0.691   # BS.1770 loudness offset

# ITU-R BS.1770-4 true-peak meter: the standardised 4x oversampling FIR,
# arranged as 4 polyphase branches of 12 taps each. This is the reference
# filter every compliant meter targets, so we implement it verbatim rather
# than a clean resampler (which would report the lower mathematical
# continuous peak). Different meters still diverge by up to ~0.5 dB on pure
# tones near Nyquist; downstream limiting carries headroom to absorb that.
_TP_PHASES = np.array([
    [0.0017089843750, 0.0109863281250, -0.0196533203125, 0.0332031250000,
     -0.0594482421875, 0.1373291015625, 0.9721679687500, -0.1022949218750,
     0.0476074218750, -0.0266113281250, 0.0148925781250, -0.0083007812500],
    [-0.0291748046875, 0.0292968750000, -0.0517578125000, 0.0891113281250,
     -0.1665039062500, 0.4650878906250, 0.7797851562500, -0.2003173828125,
     0.1015625000000, -0.0582275390625, 0.0330810546875, -0.0189208984375],
    [-0.0189208984375, 0.0330810546875, -0.0582275390625, 0.1015625000000,
     -0.2003173828125, 0.7797851562500, 0.4650878906250, -0.1665039062500,
     0.0891113281250, -0.0517578125000, 0.0292968750000, -0.0291748046875],
    [-0.0083007812500, 0.0148925781250, -0.0266113281250, 0.0476074218750,
     -0.1022949218750, 0.9721679687500, 0.1373291015625, -0.0594482421875,
     0.0332031250000, -0.0196533203125, 0.0109863281250, 0.0017089843750],
])


def _as_2d(x: np.ndarray) -> np.ndarray:
    """Return float64 array shaped (samples, channels)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return x


def _meter(sr: int) -> pyln.Meter:
    return pyln.Meter(sr)


def _kweight(x2d: np.ndarray, sr: int) -> np.ndarray:
    """Apply the BS.1770 K-weighting (high-shelf then high-pass) per channel."""
    m = _meter(sr)
    y = x2d.copy()
    for stage in m._filters.values():
        b, a = stage.b, stage.a
        for ch in range(y.shape[1]):
            y[:, ch] = lfilter(b, a, y[:, ch])
    return y


def _channel_weights(n: int) -> np.ndarray:
    if n <= len(_CHANNEL_GAINS):
        return _CHANNEL_GAINS[:n]
    # Unknown layouts beyond 5ch: treat extras as unweighted.
    w = np.ones(n)
    w[: len(_CHANNEL_GAINS)] = _CHANNEL_GAINS
    return w


def integrated_lufs(x: np.ndarray, sr: int) -> float:
    """Integrated (gated) loudness in LUFS per ITU-R BS.1770-4.

    Returns ``-inf`` for signals with no content above the gate (silence/DC)
    rather than raising or producing NaN.
    """
    x2d = _as_2d(x)
    try:
        val = _meter(sr).integrated_loudness(x2d)
    except Exception:
        return float("-inf")
    if val is None or np.isnan(val):
        return float("-inf")
    return float(val)


def short_term_lufs(
    x: np.ndarray, sr: int, hop: float = 0.1, win: float = 3.0
) -> np.ndarray:
    """Ungated short-term loudness trajectory (LUFS), one value per hop.

    Windows shorter than ``win`` at the signal head are dropped, matching the
    "first full window" convention. Returns a 1-D array; windows whose mean
    square is zero map to ``-inf`` (not NaN).
    """
    x2d = _as_2d(x)
    y = _kweight(x2d, sr)
    z = y ** 2  # per-sample mean-square contribution, (samples, ch)

    win_n = max(1, int(round(win * sr)))
    hop_n = max(1, int(round(hop * sr)))
    n = z.shape[0]
    if n < win_n:
        win_n = n

    # Sliding-window mean via cumulative sum (cheap and exact).
    csum = np.concatenate([np.zeros((1, z.shape[1])), np.cumsum(z, axis=0)], axis=0)
    starts = np.arange(0, n - win_n + 1, hop_n)
    if len(starts) == 0:
        starts = np.array([0])
    weights = _channel_weights(z.shape[1])

    out = np.empty(len(starts), dtype=np.float64)
    for i, s in enumerate(starts):
        block_mean = (csum[s + win_n] - csum[s]) / win_n  # per-channel mean square
        power = float(np.sum(weights * block_mean))
        out[i] = _OFFSET + 10.0 * np.log10(power) if power > 0 else float("-inf")
    return out


def loudness_range(x: np.ndarray, sr: int) -> float:
    """Loudness Range (LU) per EBU Tech 3342.

    3 s short-term windows, 1 s hops, absolute gate at -70 LUFS, relative gate
    at -20 LU below the (power-mean) loudness of the absolutely-gated blocks,
    then the spread between the 95th and 10th percentiles of the survivors.
    Returns 0.0 when there is too little content to form a distribution.
    """
    st = short_term_lufs(x, sr, hop=1.0, win=3.0)
    st = st[np.isfinite(st)]
    above_abs = st[st >= _ABS_GATE]
    if above_abs.size < 2:
        return 0.0

    # Relative threshold from the power mean of absolutely-gated blocks.
    mean_power = np.mean(10.0 ** (above_abs / 10.0))
    rel_thresh = 10.0 * np.log10(mean_power) - 20.0
    gated = above_abs[above_abs >= rel_thresh]
    if gated.size < 2:
        return 0.0

    p10, p95 = np.percentile(gated, [10.0, 95.0])
    return float(p95 - p10)


def true_peak_db(x: np.ndarray, sr: int, oversample: int = 4) -> float:
    """Maximum true (inter-sample) peak in dBTP per ITU-R BS.1770-4.

    Uses the standardised 4x polyphase FIR. The ``oversample`` argument is
    kept for API compatibility; values >4 fall back to a clean polyphase
    resampler (the FIR table only defines 4x). Returns a finite floor
    (~-240 dBTP) for silence rather than ``-inf``.
    """
    x2d = _as_2d(x)
    peak = 0.0
    if oversample == 4:
        for ch in range(x2d.shape[1]):
            xc = x2d[:, ch]
            if xc.size == 0:
                continue
            for phase in _TP_PHASES:
                y = lfilter(phase, [1.0], xc)
                peak = max(peak, float(np.max(np.abs(y))))
    else:
        for ch in range(x2d.shape[1]):
            up = resample_poly(x2d[:, ch], oversample, 1)
            peak = max(peak, float(np.max(np.abs(up))) if up.size else 0.0)
    return 20.0 * np.log10(max(peak, 1e-12))


def measure(x: np.ndarray, sr: int) -> dict:
    """Convenience bundle used by analysis/eval reporting."""
    return {
        "lufs": integrated_lufs(x, sr),
        "lra": loudness_range(x, sr),
        "tp_db": true_peak_db(x, sr),
    }
