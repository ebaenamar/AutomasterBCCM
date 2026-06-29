"""Phase 2 — deterministic baseline (the honest control).

The baseline is gain-to-target with a *conditional* true-peak limiter:
- target loudness defaults to BCCM's measured ≈ -17 LUFS (not -14);
- the ceiling is configurable and the limiter can be disabled entirely
  ("replicate" mode) since Boris's real masters run hot (>0 dBTP).
"""
import shutil

import numpy as np
import pytest
import soundfile as sf

from automaster import baseline, metrics
from conftest import sine, db_to_lin

SR = 48000
FFMPEG = shutil.which("ffmpeg")


def _noisy_program(dur=10.0, seed=0, level_db=-26.0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    x = (0.5 * np.sin(2 * np.pi * 110 * t)
         + 0.3 * np.sin(2 * np.pi * 440 * t)
         + 0.2 * rng.standard_normal(t.shape))
    x = x / np.max(np.abs(x)) * 0.5
    # set it to a known-ish loudness by scaling to a target peak
    return x * db_to_lin(level_db + 6.0)


def test_hits_target_lufs():
    x = _noisy_program(level_db=-30.0)
    y = baseline.process(x, SR, target_lufs=-17.0, ceiling_db=-1.0)
    out = metrics.integrated_lufs(y, SR)
    assert abs(out - (-17.0)) < 0.5, f"got {out:.2f} LUFS"


def test_true_peak_under_ceiling_always():
    """TP must stay under the ceiling for every signal, including nasty
    transients (clicks, a crescendo)."""
    ceiling = -1.0
    signals = {
        "tones": _noisy_program(level_db=-28.0, seed=1),
        "clicks": _clicky(),
        "crescendo": _crescendo(),
        "already_hot": sine(1000.0, 3.0, SR, amp=db_to_lin(-0.2)),
    }
    for name, x in signals.items():
        y = baseline.process(x, SR, target_lufs=-16.0, ceiling_db=ceiling)
        tp = metrics.true_peak_db(y, SR)
        assert tp <= ceiling + 1e-6, f"{name}: TP {tp:.3f} exceeds ceiling {ceiling}"


def test_no_compression_preserves_lra():
    """If the gain-to-target never reaches the ceiling, the limiter is a
    no-op and dynamics (LRA) are preserved (ΔLRA ≈ 0)."""
    quiet = sine(300.0, 5.0, SR, amp=db_to_lin(-40.0))
    loud = sine(300.0, 5.0, SR, amp=db_to_lin(-26.0))
    x = np.concatenate([quiet, loud])
    lra_in = metrics.loudness_range(x, SR)
    # Low target so even after gain we stay well below the ceiling.
    y = baseline.process(x, SR, target_lufs=-30.0, ceiling_db=-1.0)
    lra_out = metrics.loudness_range(y, SR)
    assert abs(lra_out - lra_in) < 1.0, f"LRA changed {lra_in:.2f}->{lra_out:.2f}"


def test_replicate_mode_allows_over_ceiling():
    """With the limiter disabled, a loud push is left hot (matches Boris);
    with it enabled the same push is clamped under the ceiling."""
    x = _noisy_program(level_db=-20.0)
    hot = baseline.process(x, SR, target_lufs=-8.0, ceiling_db=-1.0,
                           apply_limiter=False)
    safe = baseline.process(x, SR, target_lufs=-8.0, ceiling_db=-1.0,
                            apply_limiter=True)
    tp_hot = metrics.true_peak_db(hot, SR)
    tp_safe = metrics.true_peak_db(safe, SR)
    assert tp_hot > -1.0, "replicate mode should permit peaks over the ceiling"
    assert tp_safe <= -1.0 + 1e-6, "limiter mode must respect the ceiling"
    assert tp_hot > tp_safe


def test_silence_passthrough():
    """Silence must not blow up or NaN (gain toward target is undefined)."""
    y = baseline.process(np.zeros(SR), SR, target_lufs=-16.0)
    assert y.shape == (SR,)
    assert not np.any(np.isnan(y))


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")
def test_ffmpeg_baseline_agrees(tmp_path):
    """Our baseline and the ffmpeg two-pass loudnorm reference should land on
    the same target loudness, within a loose tolerance."""
    # Target chosen so the true-peak ceiling is NOT the binding constraint:
    # ffmpeg's linear loudnorm only adjusts gain (no limiter), so it would
    # undershoot whenever TP binds. In the gain-only regime both must agree.
    x = _noisy_program(level_db=-26.0)
    in_wav = tmp_path / "in.wav"
    out_wav = tmp_path / "out.wav"
    sf.write(in_wav, x, SR, subtype="PCM_24")
    target = -23.0

    baseline.baseline_ffmpeg(in_wav, out_wav, target_i=target, target_tp=-1.0)
    yf, sr_out = sf.read(out_wav)
    assert sr_out == SR, "baseline_ffmpeg must preserve the source sample rate"
    lufs_ffmpeg = metrics.integrated_lufs(yf, sr_out)

    yo = baseline.process(x, SR, target_lufs=target, ceiling_db=-1.0)
    lufs_ours = metrics.integrated_lufs(yo, SR)

    assert abs(lufs_ffmpeg - target) < 1.0, f"ffmpeg landed at {lufs_ffmpeg:.2f}"
    assert abs(lufs_ours - lufs_ffmpeg) < 1.0, f"ours {lufs_ours:.2f} vs ffmpeg {lufs_ffmpeg:.2f}"


def _clicky():
    x = np.zeros(int(4.0 * SR))
    x[::SR // 2] = 0.9  # a click every 0.5 s
    x += sine(220.0, 4.0, SR, amp=db_to_lin(-30.0))
    return x


def _crescendo():
    t = np.arange(int(6.0 * SR)) / SR
    env = np.linspace(0.01, 0.95, t.size)
    return env * np.sin(2 * np.pi * 330 * t)
