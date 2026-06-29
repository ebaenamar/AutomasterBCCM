"""Phase 0 — measurement primitives. Oracle-based tests.

The absolute LUFS oracle is a cross-check against ffmpeg's `loudnorm`
(an independent BS.1770 implementation), which is more honest than
hardcoding a K-weighting constant we'd derive from the same theory the
code under test uses. Gain invariance, by contrast, is exact and is the
strongest internal oracle.
"""
import json
import re
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from automaster import metrics
from conftest import sine, db_to_lin


FFMPEG = shutil.which("ffmpeg")


def _ffmpeg_loudnorm_measure(x, sr, tmp_path):
    """Run ffmpeg loudnorm in measurement mode; return its parsed JSON."""
    wav = tmp_path / "probe.wav"
    sf.write(wav, x, sr, subtype="PCM_24")
    out = subprocess.run(
        [FFMPEG, "-nostats", "-hide_banner", "-i", str(wav),
         "-af", "loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True, text=True,
    ).stderr
    # JSON object is the last {...} block in stderr.
    start = out.rfind("{")
    end = out.rfind("}")
    return json.loads(out[start:end + 1])


def _ffmpeg_ebur128_true_peak(x, sr, tmp_path):
    """True peak (dBTP) via ffmpeg's ebur128 filter (libebur128 — a proper
    BS.1770 true-peak meter, unlike loudnorm's overshooting estimator)."""
    wav = tmp_path / "probe.wav"
    sf.write(wav, x, sr, subtype="PCM_24")
    out = subprocess.run(
        [FFMPEG, "-nostats", "-hide_banner", "-i", str(wav),
         "-af", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True, text=True,
    ).stderr
    m = re.search(r"True peak:\s*\n\s*Peak:\s*([-\d.]+)", out)
    return float(m.group(1))


# ----------------------------------------------------------------------------
# integrated LUFS
# ----------------------------------------------------------------------------

def test_gain_invariance_exact(sine_1k_minus20, sr):
    """Scaling by +6.02 dB must raise integrated LUFS by +6.02 ± 0.05 LU."""
    base = metrics.integrated_lufs(sine_1k_minus20, sr)
    louder = metrics.integrated_lufs(sine_1k_minus20 * db_to_lin(6.02), sr)
    assert abs((louder - base) - 6.02) < 0.05


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")
def test_integrated_lufs_matches_ffmpeg(sine_1k_minus20, sr, tmp_path):
    ours = metrics.integrated_lufs(sine_1k_minus20, sr)
    ref = float(_ffmpeg_loudnorm_measure(sine_1k_minus20, sr, tmp_path)["input_i"])
    assert abs(ours - ref) < 0.5


def test_lufs_per_channel_stereo(sr):
    """A stereo signal where both channels are identical must measure the
    same integrated loudness as the mono version (within rounding)."""
    mono = sine(1000.0, 4.0, sr, amp=db_to_lin(-18.0))
    stereo = np.stack([mono, mono], axis=1)
    lm = metrics.integrated_lufs(mono, sr)
    ls = metrics.integrated_lufs(stereo, sr)
    # Summing two identical channels (each gain 1.0) adds +3 LU vs one channel.
    assert abs((ls - lm) - 3.0) < 0.3


# ----------------------------------------------------------------------------
# short-term trajectory
# ----------------------------------------------------------------------------

def test_short_term_trajectory_shape_and_value(sr):
    x = sine(1000.0, 8.0, sr, amp=db_to_lin(-18.0))
    traj = metrics.short_term_lufs(x, sr, hop=0.1, win=3.0)
    assert traj.ndim == 1
    assert len(traj) > 10
    # On a stationary tone the steady-state short-term value tracks integrated.
    integ = metrics.integrated_lufs(x, sr)
    steady = np.median(traj[~np.isinf(traj)])
    assert abs(steady - integ) < 1.0


# ----------------------------------------------------------------------------
# true peak (oversampled)
# ----------------------------------------------------------------------------

def test_true_peak_exceeds_sample_peak(intersample_peak_signal, sr):
    x = intersample_peak_signal
    sample_peak_db = 20 * np.log10(np.max(np.abs(x)))
    tp = metrics.true_peak_db(x, sr, oversample=4)
    assert tp > sample_peak_db + 0.05


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")
def test_true_peak_matches_ffmpeg_midband_tight(sr, tmp_path):
    """At 1 kHz every compliant TP meter agrees: tolerance 0.15 dB."""
    x = sine(997.0, 2.0, sr, amp=db_to_lin(-3.0))
    ours = metrics.true_peak_db(x, sr, oversample=4)
    ref = _ffmpeg_ebur128_true_peak(x, sr, tmp_path)
    assert abs(ours - ref) < 0.15, f"ours={ours} ebur128={ref}"


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")
def test_true_peak_matches_ffmpeg_multi(sr, tmp_path):
    """Cross-check TP against ffmpeg's ebur128 meter on ≥3 distinct signals.

    Near Nyquist, BS.1770 implementations legitimately diverge (the FIR
    overshoot differs between libebur128 and the spec table), so the
    tolerance widens to 0.6 dB there while staying tight mid-band. ebur128
    reports to 0.1 dB resolution, which also eats into the budget.
    """
    cases = [
        (sine(997.0, 2.0, sr, amp=db_to_lin(-3.0)), 0.15),
        (sine(sr / 4 + 30, 2.0, sr, amp=db_to_lin(-6.0), phase=np.pi / 3), 0.6),
        (sine(15000.0, 2.0, sr, amp=db_to_lin(-1.0)), 0.6),
    ]
    for x, tol in cases:
        ours = metrics.true_peak_db(x, sr, oversample=4)
        ref = _ffmpeg_ebur128_true_peak(x, sr, tmp_path)
        assert abs(ours - ref) < tol, f"TP mismatch: ours={ours} ebur128={ref} tol={tol}"


# ----------------------------------------------------------------------------
# loudness range
# ----------------------------------------------------------------------------

@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")
def test_lra_matches_ffmpeg(two_level_signal, sr, tmp_path):
    ours = metrics.loudness_range(two_level_signal, sr)
    ref = float(_ffmpeg_loudnorm_measure(two_level_signal, sr, tmp_path)["input_lra"])
    # EBU Tech 3342 tolerance is loose; allow 1 LU.
    assert abs(ours - ref) < 1.0


def test_lra_nonnegative_on_stationary(sr):
    x = sine(1000.0, 6.0, sr, amp=db_to_lin(-18.0))
    assert metrics.loudness_range(x, sr) >= 0.0


# ----------------------------------------------------------------------------
# robustness: silence / DC / hard clipping must not raise or NaN
# ----------------------------------------------------------------------------

def test_robustness_no_nan(sr):
    cases = {
        "silence": np.zeros(sr * 2),
        "dc": np.ones(sr * 2) * 0.5,
        "clipped": np.clip(sine(1000.0, 2.0, sr, amp=4.0), -1.0, 1.0),
    }
    for name, x in cases.items():
        for fn in (metrics.integrated_lufs, metrics.loudness_range):
            v = fn(x, sr)
            assert not np.isnan(v), f"{fn.__name__} NaN on {name}"
        tp = metrics.true_peak_db(x, sr)
        assert not np.isnan(tp), f"true_peak_db NaN on {name}"
