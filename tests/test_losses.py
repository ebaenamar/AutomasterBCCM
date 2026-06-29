"""Phase 3 — differentiable loudness losses."""
import numpy as np
import torch
import pytest

from automaster import losses, metrics
from conftest import sine, db_to_lin

SR = 48000


def _t(x):
    """numpy mono/stereo -> torch (1, ch, samples)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    else:
        x = x.T
    return torch.tensor(x)[None]


def test_torch_kweight_matches_pyloudnorm_static():
    """Differentiable integrated LUFS must match pyloudnorm on a stationary
    signal within ±0.3 LU (ungated proxy vs gated reference)."""
    x = sine(1000.0, 5.0, SR, amp=db_to_lin(-18.0))
    ref = metrics.integrated_lufs(x, SR)
    got = float(losses.integrated_lufs(_t(x), SR))
    assert abs(got - ref) < 0.3, f"torch={got:.2f} pyln={ref:.2f}"


def test_torch_lufs_gain_invariance():
    x = sine(1000.0, 4.0, SR, amp=db_to_lin(-20.0))
    base = float(losses.integrated_lufs(_t(x), SR))
    louder = float(losses.integrated_lufs(_t(x * db_to_lin(6.0)), SR))
    assert abs((louder - base) - 6.0) < 0.05


def test_short_term_steady_state():
    x = sine(1000.0, 8.0, SR, amp=db_to_lin(-18.0))
    traj = losses.short_term_lufs(_t(x), SR)
    integ = float(losses.integrated_lufs(_t(x), SR))
    assert abs(float(traj.median()) - integ) < 1.0


def test_torch_true_peak_matches_numpy():
    x = sine(SR / 4 + 30, 2.0, SR, amp=db_to_lin(-1.0), phase=np.pi / 4)
    ref = metrics.true_peak_db(x, SR, oversample=4)
    got = float(losses.true_peak_db(_t(x), SR))
    assert abs(got - ref) < 0.1, f"torch={got:.2f} numpy={ref:.2f}"


def test_tp_hinge_penalises_over_ceiling():
    loss = losses.MasteringLoss(SR, w_env=0, w_int=0, w_tp=1.0, w_stft=0,
                                ceiling_db=-1.0)
    hot = _t(sine(1000.0, 1.0, SR, amp=db_to_lin(-0.1)))   # ~0 dBTP, over ceiling
    quiet = _t(sine(1000.0, 1.0, SR, amp=db_to_lin(-12.0)))  # well under
    assert float(loss(hot, hot)) > float(loss(quiet, quiet))
    assert float(loss(quiet, quiet)) < 1e-4


def test_loss_backward_no_nan():
    x = _t(sine(440.0, 1.0, SR, amp=db_to_lin(-18.0))).clone().requires_grad_(True)
    ref = _t(sine(440.0, 1.0, SR, amp=db_to_lin(-14.0)))
    loss = losses.MasteringLoss(SR, w_stft=1.0)(x, ref)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(x.grad).all()
