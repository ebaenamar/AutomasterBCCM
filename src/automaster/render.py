"""Phase 5 — production renderer: input → features → θ → chain → audio.

Same differentiable DSP as the fit (so what we learned is what we apply), run
without gradients, plus the deterministic true-peak limiter as the configurable
safety stage. Loudness is driven to the model's target with a one-step gain
correction around the (near level-preserving) EQ/compressor.
"""
from __future__ import annotations

import numpy as np
import torch

from automaster import dsp_diff, metrics, baseline, io


def features(x: np.ndarray, sr: int) -> dict:
    """Input descriptors for the model (and for logging)."""
    m = metrics.measure(x, sr)
    return {"lufs": m["lufs"], "lra": m["lra"], "tp_db": m["tp_db"]}


def render(x, sr, model, apply_limiter=True, ceiling_db=-1.0):
    """Master ``x`` with ``model``. Returns audio shaped like the input."""
    x = np.asarray(x, dtype=np.float64)
    feats = features(x, sr)
    theta = model.predict(feats)

    lufs_in = feats["lufs"]
    if not np.isfinite(lufs_in):
        return x  # silence

    # Initial gain to reach target, then EQ + compressor.
    theta = dict(theta)
    theta["gain_db"] = float(model.target_lufs - lufs_in)

    x2d = x[:, None] if x.ndim == 1 else x
    xt = torch.tensor(x2d.T, dtype=torch.float64)[None]  # (1, ch, samples)
    with torch.no_grad():
        yt = dsp_diff.apply_chain(xt, sr, _physical_tensors(theta))
    y = yt.squeeze(0).numpy().T  # (samples, ch)

    # One-step loudness correction (EQ/comp shift the level slightly).
    lufs_out = metrics.integrated_lufs(y, sr)
    if np.isfinite(lufs_out):
        y *= 10.0 ** ((model.target_lufs - lufs_out) / 20.0)

    if apply_limiter:
        y = baseline.true_peak_limit(y, sr, ceiling_db=ceiling_db)

    y2 = np.asarray(y)
    if x.ndim == 1:
        y2 = y2[:, 0] if y2.ndim == 2 else y2
    return y2


def _physical_tensors(theta: dict) -> dict:
    return {k: torch.tensor(float(v), dtype=torch.float64) for k, v in theta.items()}


def render_file(in_path, out_path, model, apply_limiter=True, ceiling_db=-1.0,
                subtype="PCM_24"):
    """Decode any input (wav/mp4/...), master it, write a WAV."""
    x, sr = io.load_audio(in_path)
    y = render(x, sr, model, apply_limiter=apply_limiter, ceiling_db=ceiling_db)
    io.save_audio(out_path, y, sr, subtype=subtype)
    return out_path
