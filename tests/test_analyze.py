"""Phase 1 — delta computation correctness (the 'Sunday afternoon' analysis)."""
import numpy as np
import pytest

from automaster import analyze
from conftest import sine, db_to_lin

SR = 48000


def test_pure_gain_delta():
    """A pure +5 dB gain change: ΔLUFS ≈ +5, ΔLRA ≈ 0."""
    rng = np.random.default_rng(1)
    t = np.arange(int(6.0 * SR)) / SR
    before = 0.3 * np.sin(2 * np.pi * 300 * t) + 0.05 * rng.standard_normal(t.shape)
    after = before * db_to_lin(5.0)
    d = analyze.pair_delta(before, after, SR, SR)
    assert abs(d["d_lufs"] - 5.0) < 0.3
    assert abs(d["d_lra"]) < 1.0
    assert "tp_after_db" in d


def test_eq_curve_recovers_known_tilt():
    """Apply a known high-shelf boost; the recovered EQ curve must be ~0 dB
    in the lows and clearly positive in the highs (gain-independent)."""
    from scipy.signal import butter, sosfilt
    rng = np.random.default_rng(2)
    before = rng.standard_normal(int(8.0 * SR)) * 0.1
    # First-order high-shelf-ish: boost highs by mixing in a high-passed copy.
    sos = butter(2, 4000, btype="high", fs=SR, output="sos")
    after = before + 1.0 * sosfilt(sos, before)  # ~+6 dB shelf above 4 kHz
    after *= db_to_lin(3.0)  # also a broadband gain — must be removed by the curve
    freqs, curve_db = analyze.eq_curve(before, after, SR)
    lo = curve_db[(freqs > 100) & (freqs < 500)].mean()
    hi = curve_db[(freqs > 8000) & (freqs < 16000)].mean()
    assert hi - lo > 3.0, f"expected HF boost, got lo={lo:.2f} hi={hi:.2f}"
    assert abs(lo) < 1.5, f"low band should be ~0 after gain-match, got {lo:.2f}"


def test_compression_reduces_lra():
    """A loud+quiet program that gets level-compressed shows ΔLRA < 0."""
    quiet = sine(300.0, 4.0, SR, amp=db_to_lin(-30.0))
    loud = sine(300.0, 4.0, SR, amp=db_to_lin(-10.0))
    before = np.concatenate([quiet, loud])
    # crude "compression": bring the quiet part up, leave loud → smaller range
    after = np.concatenate([quiet * db_to_lin(12.0), loud])
    d = analyze.pair_delta(before, after, SR, SR)
    assert d["d_lra"] < -1.0
