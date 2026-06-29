"""Phase 1 — temporal alignment of before/after pairs."""
import numpy as np
import pytest

from automaster import align
from conftest import sine, db_to_lin

SR = 48000


def _program(sr=SR, dur=4.0):
    """A non-periodic-ish test program (sum of tones + noise) so cross-
    correlation has a sharp, unambiguous peak."""
    rng = np.random.default_rng(0)
    t = np.arange(int(dur * sr)) / sr
    x = (0.3 * np.sin(2 * np.pi * 220 * t)
         + 0.2 * np.sin(2 * np.pi * 437 * t)
         + 0.1 * rng.standard_normal(t.shape))
    return x / np.max(np.abs(x)) * 0.7


def test_estimate_offset_recovers_known_shift():
    x = _program()
    N = 1234
    # `after` is `before` delayed by N samples and attenuated.
    after = np.concatenate([np.zeros(N), x]) * db_to_lin(-4.0)
    before = np.concatenate([x, np.zeros(N)])
    est = align.estimate_offset(before, after)
    assert abs(est - N) <= 1, f"estimated {est}, expected {N}"


def test_align_trims_to_common_region():
    x = _program()
    N = 500
    after = np.concatenate([np.zeros(N), x]) * db_to_lin(-4.0)
    before = np.concatenate([x, np.zeros(N)])
    b, a, sr = align.align(before, after, SR, SR)
    assert len(b) == len(a)
    assert sr == SR
    # After alignment the two should be highly correlated (same program).
    corr = np.corrcoef(b, a)[0, 1]
    assert corr > 0.99, f"alignment correlation only {corr}"


def test_align_resamples_after_to_before_sr():
    x = _program(sr=48000)
    # Build a 44.1k "after" by resampling, then align against 48k "before".
    a441 = align.resample_to(x, 48000, 44100)
    b, a, sr = align.align(x, a441, 48000, 44100)
    assert sr == 48000
    assert len(b) == len(a)
    corr = np.corrcoef(b, a)[0, 1]
    assert corr > 0.98, f"cross-rate alignment correlation only {corr}"


def test_align_stereo_keeps_channels():
    x = _program()
    st = np.stack([x, x * 0.9], axis=1)
    b, a, sr = align.align(st, st.copy(), SR, SR)
    assert b.ndim == 2 and b.shape[1] == 2
    assert b.shape == a.shape
