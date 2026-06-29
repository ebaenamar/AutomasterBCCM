"""Phase 4A — per-pair fit: recover Boris's effective θ from a before/after pair.

For each aligned pair we optimise the differentiable chain's parameters to
reproduce `after` from `before`, using the loudness-space loss (+ multi-res
STFT for EQ/timbre). The fitted θ are the clean labels for the regressor; the
final residual tells us whether the chain captures the editor's process or
whether something (heavier EQ, time-varying moves) is missing.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from automaster import dsp_diff, losses


def _to_mono_tensor(x: np.ndarray) -> torch.Tensor:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return torch.tensor(x)[None, None]  # (1, 1, samples)


def _excerpt(b, a, sr, excerpt_s, offset_frac=0.4):
    if excerpt_s is None:
        return b, a
    n = int(excerpt_s * sr)
    if len(b) <= n:
        return b, a
    start = int(len(b) * offset_frac)
    start = min(start, len(b) - n)
    return b[start:start + n], a[start:start + n]


def fit_pair(before, after, sr, iters=150, lr=0.1, excerpt_s=6.0,
             w_env=0.3, w_int=1.0, w_stft=1.0, seed=0,
             freeze=("comp_makeup_db",)):
    """Fit θ so chain(before) ≈ after. Returns dict with physical θ, the
    residual (final STFT-magnitude relative error), and loudness errors.

    ``freeze`` pins params to their neutral value (kept out of the gradient).
    Freezing ``comp_makeup_db`` removes the gain↔makeup degeneracy so the
    broadband gain — not the compressor — carries loudness, which stops the
    optimiser from over-compressing to hit level.
    """
    torch.manual_seed(seed)
    b_np, a_np = _excerpt(np.asarray(before), np.asarray(after), sr, excerpt_s)
    x = _to_mono_tensor(b_np)
    y = _to_mono_tensor(a_np)

    neutral = dsp_diff.neutral_u().clone()
    free_mask = torch.tensor(
        [0.0 if n in (freeze or ()) else 1.0 for n in dsp_diff.PARAM_NAMES],
        dtype=torch.float64)
    # Free params start at mid-range (max sigmoid gradient); frozen stay neutral.
    u = torch.zeros(dsp_diff.N_PARAMS, dtype=torch.float64).requires_grad_(True)
    loss_fn = losses.MasteringLoss(sr, w_env=w_env, w_int=w_int, w_tp=0.0,
                                   w_stft=w_stft)
    opt = torch.optim.Adam([u], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        u_cur = neutral * (1 - free_mask) + u * free_mask
        loss = loss_fn(dsp_diff.render_u(x, sr, u_cur), y)
        loss.backward()
        opt.step()

    u = (neutral * (1 - free_mask) + u * free_mask).detach()
    y_hat = dsp_diff.render_u(x, sr, u).detach()
    phys = {k: float(v) for k, v in dsp_diff.to_physical(u).items()}

    # Residual: relative error of the log-magnitude spectrum (tonal/dynamic
    # mismatch the chain couldn't absorb), plus loudness errors.
    resid = _spectral_residual(y_hat, y)
    lufs_err = abs(float(losses.integrated_lufs(y_hat, sr))
                   - float(losses.integrated_lufs(y, sr)))
    return {"theta": phys, "residual": resid, "lufs_err": lufs_err,
            "u": u.numpy().tolist()}


def _spectral_residual(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """Relative log-magnitude spectral error (0 = perfect)."""
    Y = torch.stft(y.reshape(-1), n_fft=2048, hop_length=512,
                   return_complex=True, window=torch.hann_window(2048)).abs()
    H = torch.stft(y_hat.reshape(-1), n_fft=2048, hop_length=512,
                   return_complex=True, window=torch.hann_window(2048)).abs()
    num = torch.linalg.norm((torch.log1p(H) - torch.log1p(Y)).reshape(-1))
    den = torch.linalg.norm(torch.log1p(Y).reshape(-1)) + 1e-9
    return float(num / den)
