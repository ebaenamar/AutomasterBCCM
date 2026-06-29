"""Phase 4A — per-pair fit. Kept small for speed (the fit is iterative)."""
import numpy as np
import torch
import pytest
from scipy.signal import iirnotch, lfilter

from automaster import dsp_diff, fit

SR = 48000


def _program(dur=3.0):
    rng = np.random.default_rng(0)
    t = np.arange(int(dur * SR)) / SR
    x = (0.3 * np.sin(2 * np.pi * 110 * t) + 0.2 * np.sin(2 * np.pi * 700 * t)
         + 0.12 * rng.standard_normal(t.shape))
    return x / np.max(np.abs(x)) * 0.35


def test_modeled_pair_low_residual():
    """A pair the chain *can* represent fits to a low residual and matches
    loudness — the fit reproduces the target even if θ is degenerate."""
    before = _program()
    star = dsp_diff.physical_to_u(dict(gain_db=5.0, ls_gain_db=6.0,
                                       comp_threshold_db=-22.0, comp_ratio=3.0))
    after = dsp_diff.render_u(torch.tensor(before)[None, None], SR, star).detach().numpy().reshape(-1)
    r = fit.fit_pair(before, after, SR, iters=80, excerpt_s=2.0)
    assert r["residual"] < 0.1, f"residual {r['residual']:.4f}"
    assert r["lufs_err"] < 1.0


def test_unmodeled_raises_residual():
    """A sharp notch the fixed-topology EQ can't make should leave a larger
    residual than the modeled case — the residual is a faithful detector."""
    before = _program()
    star = dsp_diff.physical_to_u(dict(gain_db=4.0, ls_gain_db=4.0))
    after = dsp_diff.render_u(torch.tensor(before)[None, None], SR, star).detach().numpy().reshape(-1)
    r_mod = fit.fit_pair(before, after, SR, iters=80, excerpt_s=2.0)

    bn, an = iirnotch(3000 / (SR / 2), 30)
    after_notch = lfilter(bn, an, after)
    r_unmod = fit.fit_pair(before, after_notch, SR, iters=80, excerpt_s=2.0)

    assert r_unmod["residual"] > r_mod["residual"]
