"""Phase 3 — differentiable mastering chain (dasp-pytorch).

Chain (learnable part): gain → parametric EQ → compressor. The true-peak
limiter is a deterministic render-time stage (see :mod:`automaster.baseline`),
not part of the gradient graph — Boris's masters run hot, so forcing a ceiling
during the fit would prevent matching them, and keeping the limiter identical
in baseline/fit/render is what guarantees "what we learn is what we apply".

Param notes verified against the installed dasp version:
* ``release_ms`` gets NO gradient from dasp's approximate ballistics, so it is
  a fixed hyperparameter, not learnable. EQ band freqs/Qs are also fixed; only
  the band gains (+ shelves), broadband gain, and compressor
  threshold/ratio/makeup are learned.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
import dasp_pytorch.functional as F

# Learnable parameters: name -> (min, max) physical range.
PARAM_RANGES = OrderedDict([
    ("gain_db", (-12.0, 24.0)),
    ("ls_gain_db", (-15.0, 15.0)),
    ("b0_gain_db", (-12.0, 12.0)),
    ("b1_gain_db", (-12.0, 12.0)),
    ("b2_gain_db", (-12.0, 12.0)),
    ("b3_gain_db", (-12.0, 12.0)),
    ("hs_gain_db", (-15.0, 15.0)),
    ("comp_threshold_db", (-50.0, 0.0)),
    ("comp_ratio", (1.0, 12.0)),
    ("comp_makeup_db", (0.0, 15.0)),
])
PARAM_NAMES = list(PARAM_RANGES)
N_PARAMS = len(PARAM_NAMES)

# Neutral (~identity) physical value for each learnable param: 0 dB gains,
# unity compression. Used as the default when a param is unspecified so that
# `physical_to_u({...})` yields an identity chain except for the named params.
NEUTRAL_PHYS = dict(
    gain_db=0.0, ls_gain_db=0.0, b0_gain_db=0.0, b1_gain_db=0.0,
    b2_gain_db=0.0, b3_gain_db=0.0, hs_gain_db=0.0,
    comp_threshold_db=0.0, comp_ratio=1.0, comp_makeup_db=0.0,
)

# Fixed DSP topology (Boris's signature: low shelf for bass + flexible bands).
FIXED = dict(
    ls_cutoff=90.0, ls_q=0.707,
    b0_freq=200.0, b1_freq=700.0, b2_freq=2500.0, b3_freq=7000.0, band_q=1.0,
    hs_cutoff=10000.0, hs_q=0.707,
    comp_attack_ms=15.0, comp_release_ms=120.0, comp_knee_db=6.0,
)


def _lo_hi(device, dtype):
    lo = torch.tensor([PARAM_RANGES[n][0] for n in PARAM_NAMES], device=device, dtype=dtype)
    hi = torch.tensor([PARAM_RANGES[n][1] for n in PARAM_NAMES], device=device, dtype=dtype)
    return lo, hi


def to_physical(u: torch.Tensor) -> dict:
    """Map unconstrained params (N_PARAMS,) -> dict of physical values."""
    lo, hi = _lo_hi(u.device, u.dtype)
    phys = lo + (hi - lo) * torch.sigmoid(u)
    return {n: phys[i] for i, n in enumerate(PARAM_NAMES)}


def physical_to_u(values: dict) -> torch.Tensor:
    """Inverse of :func:`to_physical` for initialising at a known point."""
    lo, hi = _lo_hi("cpu", torch.float64)
    out = torch.zeros(N_PARAMS, dtype=torch.float64)
    for i, n in enumerate(PARAM_NAMES):
        v = float(values.get(n, NEUTRAL_PHYS[n]))
        frac = (v - float(lo[i])) / (float(hi[i]) - float(lo[i]))
        frac = min(max(frac, 1e-4), 1 - 1e-4)
        out[i] = torch.logit(torch.tensor(frac))
    return out


def neutral_u() -> torch.Tensor:
    """Unconstrained params for an ~identity chain (0 dB gains, ratio 1)."""
    return physical_to_u(dict(
        gain_db=0.0, ls_gain_db=0.0, b0_gain_db=0.0, b1_gain_db=0.0,
        b2_gain_db=0.0, b3_gain_db=0.0, hs_gain_db=0.0,
        comp_threshold_db=0.0, comp_ratio=1.0, comp_makeup_db=0.0,
    ))


def _c(v, x):
    """Scalar/tensor -> (1,1) tensor on x's device/dtype."""
    if not torch.is_tensor(v):
        v = torch.tensor(float(v), device=x.device, dtype=x.dtype)
    return v.reshape(1, 1)


def apply_chain(x: torch.Tensor, sr: int, phys: dict, fixed: dict = None) -> torch.Tensor:
    """Apply gain → parametric EQ → compressor. x: (1, ch, samples)."""
    f = {**FIXED, **(fixed or {})}

    y = F.gain(x, sr, _c(phys["gain_db"], x))

    y = F.parametric_eq(
        y, sr,
        low_shelf_gain_db=_c(phys["ls_gain_db"], x),
        low_shelf_cutoff_freq=_c(f["ls_cutoff"], x),
        low_shelf_q_factor=_c(f["ls_q"], x),
        band0_gain_db=_c(phys["b0_gain_db"], x),
        band0_cutoff_freq=_c(f["b0_freq"], x),
        band0_q_factor=_c(f["band_q"], x),
        band1_gain_db=_c(phys["b1_gain_db"], x),
        band1_cutoff_freq=_c(f["b1_freq"], x),
        band1_q_factor=_c(f["band_q"], x),
        band2_gain_db=_c(phys["b2_gain_db"], x),
        band2_cutoff_freq=_c(f["b2_freq"], x),
        band2_q_factor=_c(f["band_q"], x),
        band3_gain_db=_c(phys["b3_gain_db"], x),
        band3_cutoff_freq=_c(f["b3_freq"], x),
        band3_q_factor=_c(f["band_q"], x),
        high_shelf_gain_db=_c(phys["hs_gain_db"], x),
        high_shelf_cutoff_freq=_c(f["hs_cutoff"], x),
        high_shelf_q_factor=_c(f["hs_q"], x),
    )

    y = F.compressor(
        y, sr,
        threshold_db=_c(phys["comp_threshold_db"], x),
        ratio=_c(phys["comp_ratio"], x),
        attack_ms=_c(f["comp_attack_ms"], x),
        release_ms=_c(f["comp_release_ms"], x),
        knee_db=_c(f["comp_knee_db"], x),
        makeup_gain_db=_c(phys["comp_makeup_db"], x),
    )
    return y


def render_u(x: torch.Tensor, sr: int, u: torch.Tensor, fixed: dict = None):
    """Convenience: apply the chain from unconstrained params."""
    return apply_chain(x, sr, to_physical(u), fixed)
