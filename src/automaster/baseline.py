"""Phase 2 — deterministic baseline: the honest control the ML must beat.

Pipeline: measure input loudness → broadband gain to the target → *conditional*
true-peak limiter. The limiter is a no-op when the gained signal already sits
under the ceiling, so material that doesn't need taming keeps its dynamics
(ΔLRA ≈ 0). The whole TP stage can be disabled ("replicate" mode) because
Boris's real masters run hot — see data/reports/FINDINGS.md.

Defaults reflect the measured BCCM behaviour: target ≈ -17 LUFS, not -14.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.signal import resample_poly

from automaster import metrics

_FFMPEG = shutil.which("ffmpeg")

# Target loudness measured across the Boris corpus (stereo masters sit -16..-19).
DEFAULT_TARGET_LUFS = -17.0
DEFAULT_CEILING_DB = -1.0


def _as_2d(x):
    x = np.asarray(x, dtype=np.float64)
    return x[:, None] if x.ndim == 1 else x


def _match_shape(y, like):
    return y[:, 0] if (np.ndim(like) == 1 and y.ndim == 2) else y


def true_peak_limit(
    x, sr, ceiling_db=DEFAULT_CEILING_DB, oversample=4,
    lookahead_ms=2.0, release_ms=100.0, margin_db=0.3,
):
    """Brick-wall true-peak limiter.

    Gain reduction is computed on the oversampled, channel-linked peak so
    inter-sample peaks are caught and the stereo image is preserved (same gain
    on both channels). Look-ahead gives an instant, click-free attack; release
    recovers exponentially. A small ``margin_db`` plus a final scalar trim
    guarantee the decimated output never exceeds the ceiling.
    """
    x2d = _as_2d(x)
    n, ch = x2d.shape
    ceiling_lin = 10.0 ** ((ceiling_db - margin_db) / 20.0)

    # Oversampled, channel-linked peak, reduced back to one value per base sample.
    up = np.stack([resample_poly(x2d[:, c], oversample, 1) for c in range(ch)], axis=1)
    peak_os = np.max(np.abs(up), axis=1)
    # max over each base sample's oversampled neighbourhood
    pad = (-len(peak_os)) % oversample
    if pad:
        peak_os = np.concatenate([peak_os, np.zeros(pad)])
    peak_base = peak_os.reshape(-1, oversample).max(axis=1)[:n]

    desired = np.minimum(1.0, ceiling_lin / np.maximum(peak_base, 1e-12))

    # Look-ahead: gain must already be down `la` samples before the peak.
    la = max(1, int(lookahead_ms / 1000.0 * sr))
    g_la = minimum_filter1d(desired, size=2 * la + 1, origin=-la, mode="nearest")

    # Exponential release: instant drop, slow recovery (recursive max-decay on
    # the gain *reduction* r = 1 - g).
    rel_coef = float(np.exp(-1.0 / (release_ms / 1000.0 * sr)))
    r = 1.0 - g_la
    out = np.empty_like(r)
    prev = 0.0
    for i in range(len(r)):
        prev = max(r[i], prev * rel_coef)
        out[i] = prev
    g = 1.0 - out

    y = x2d * g[:, None]

    # Safety: account for any residual decimation overshoot.
    tp = metrics.true_peak_db(y, sr)
    if tp > ceiling_db:
        y *= 10.0 ** ((ceiling_db - tp) / 20.0)
    return _match_shape(y, x)


def process(
    x, sr, target_lufs=DEFAULT_TARGET_LUFS, ceiling_db=DEFAULT_CEILING_DB,
    apply_limiter=True,
):
    """Deterministic master: gain to target, then optional TP limiting."""
    x2d = _as_2d(x)
    lufs = metrics.integrated_lufs(x2d, sr)
    if not np.isfinite(lufs):
        return _match_shape(x2d, x)  # silence: nothing to do
    y = x2d * (10.0 ** ((target_lufs - lufs) / 20.0))
    if apply_limiter:
        y = _as_2d(true_peak_limit(y, sr, ceiling_db=ceiling_db))
    return _match_shape(y, x)


def baseline_ffmpeg(in_path, out_path, target_i=DEFAULT_TARGET_LUFS,
                    target_tp=DEFAULT_CEILING_DB, target_lra=11.0):
    """Second reference: ffmpeg two-pass ``loudnorm``. Returns the output path."""
    if _FFMPEG is None:
        raise RuntimeError("ffmpeg not found on PATH")
    in_path, out_path = str(in_path), str(out_path)
    # Pass 1: measure.
    import json
    meas = subprocess.run(
        [_FFMPEG, "-nostats", "-hide_banner", "-i", in_path, "-af",
         f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}:print_format=json",
         "-f", "null", "-"], capture_output=True, text=True).stderr
    j = json.loads(meas[meas.rfind("{"):meas.rfind("}") + 1])
    # loudnorm upsamples internally to 192 kHz and would emit at that rate;
    # resample back to the source rate so the output is a drop-in master.
    in_sr = int(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate", "-of",
         "default=noprint_wrappers=1:nokey=1", in_path],
        capture_output=True, text=True, check=True).stdout.strip())
    # Pass 2: apply with measured values (linear normalisation).
    af = (f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}:"
          f"measured_I={j['input_i']}:measured_TP={j['input_tp']}:"
          f"measured_LRA={j['input_lra']}:measured_thresh={j['input_thresh']}:"
          f"offset={j['target_offset']}:linear=true,aresample={in_sr}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([_FFMPEG, "-y", "-nostats", "-hide_banner", "-i", in_path,
                    "-af", af, "-ar", str(in_sr), out_path],
                   capture_output=True, check=True)
    return out_path
