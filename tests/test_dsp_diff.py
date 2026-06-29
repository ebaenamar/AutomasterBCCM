"""Phase 3 — differentiable chain: parameter recovery + finite gradients.

Recovery is the linchpin test: render y = chain(x, θ*) with known θ*, then fit
θ̂ to match y. Gain and EQ gains are uniquely identifiable and must recover
near-exactly. The compressor's (gain, ratio, makeup) are partly degenerate from
a single program, so there we assert the stronger, meaningful claim — the
fitted chain *reconstructs* y (low residual) and lands on its loudness — which
is exactly the residual-as-detector idea Phase 4 relies on.

Recovery uses a plain waveform loss (sharp minimum at θ*); the loudness-space
MasteringLoss is for the real Boris fit and is tested in test_losses.py.
"""
import numpy as np
import torch
import torch.nn.functional as F
import pytest

from automaster import dsp_diff, losses

SR = 48000
torch.manual_seed(0)


def _program(dur=1.0):
    rng = np.random.default_rng(0)
    t = np.arange(int(dur * SR)) / SR
    x = (0.4 * np.sin(2 * np.pi * 120 * t) + 0.25 * np.sin(2 * np.pi * 900 * t)
         + 0.15 * rng.standard_normal(t.shape))
    x = x / np.max(np.abs(x)) * 0.4
    return torch.tensor(x)[None, None]  # (1, 1, samples) mono — fast


def _fit(x, y_ref, free_names, iters=250, lr=0.1):
    """Fit the free params (started at mid-range for max sigmoid gradient);
    masked params stay at neutral. Plain MSE — sharp minimum at θ*."""
    neutral = dsp_diff.neutral_u().clone()
    mask = torch.tensor([1.0 if n in free_names else 0.0 for n in dsp_diff.PARAM_NAMES],
                        dtype=torch.float64)
    u = torch.zeros_like(neutral).requires_grad_(True)
    opt = torch.optim.Adam([u], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        loss = F.mse_loss(dsp_diff.render_u(x, SR, neutral * (1 - mask) + u * mask), y_ref)
        loss.backward()
        opt.step()
    u_final = (neutral * (1 - mask) + u * mask).detach()
    return dsp_diff.to_physical(u_final), u_final


def test_recover_gain_only():
    x = _program()
    y = dsp_diff.render_u(x, SR, dsp_diff.physical_to_u(dict(gain_db=6.0))).detach()
    phys, _ = _fit(x, y, {"gain_db"}, iters=150)
    assert abs(float(phys["gain_db"]) - 6.0) < 0.5


def test_recover_eq_gains():
    x = _program()
    star = dsp_diff.physical_to_u(dict(ls_gain_db=6.0, hs_gain_db=-4.0))
    y = dsp_diff.render_u(x, SR, star).detach()
    phys, _ = _fit(x, y, {"ls_gain_db", "hs_gain_db"}, iters=300)
    assert abs(float(phys["ls_gain_db"]) - 6.0) < 1.0
    assert abs(float(phys["hs_gain_db"]) - (-4.0)) < 1.0


def test_recover_compressor_reconstruction():
    """(gain, ratio, makeup) are degenerate, so require: the fit reconstructs
    the compressed target (residual < 2% of signal variance), recovers the
    threshold within a few dB, and matches its loudness."""
    x = _program()
    star = dsp_diff.physical_to_u(dict(
        gain_db=4.0, comp_threshold_db=-25.0, comp_ratio=4.0, comp_makeup_db=3.0))
    y = dsp_diff.render_u(x, SR, star).detach()
    phys, u = _fit(x, y, {"gain_db", "comp_threshold_db", "comp_ratio", "comp_makeup_db"},
                   iters=350)
    y_hat = dsp_diff.render_u(x, SR, u).detach()
    nmse = float(F.mse_loss(y_hat, y) / y.var())
    assert nmse < 0.02, f"reconstruction NMSE {nmse:.4f} too high"
    assert abs(float(phys["comp_threshold_db"]) - (-25.0)) < 5.0
    assert abs(float(losses.integrated_lufs(y_hat, SR))
               - float(losses.integrated_lufs(y, SR))) < 0.5


def test_gradients_finite_all_params():
    x = _program()
    y_ref = dsp_diff.render_u(x, SR, dsp_diff.physical_to_u(dict(gain_db=3.0))).detach()
    u = dsp_diff.neutral_u().clone().requires_grad_(True)
    loss = losses.MasteringLoss(SR)(dsp_diff.render_u(x, SR, u), y_ref)
    loss.backward()
    assert torch.isfinite(u.grad).all()
    assert u.grad.abs().sum() > 0
