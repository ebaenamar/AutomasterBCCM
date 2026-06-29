"""Phase 3 — differentiable loudness/dynamics losses.

Loss lives in loudness space, not sample MSE: error of the short-term LUFS
envelope + error of (a differentiable proxy for) integrated LUFS + a hinge on
true-peak over the ceiling, optionally plus a multi-resolution STFT term to
catch EQ/timbre. The K-weighting and short-term loudness are reimplemented in
torch so the *estimate* is differentiable; targets may be constants from the
numpy meter.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import pyloudnorm as pyln

_OFFSET = -0.691
_CHANNEL_GAINS = [1.0, 1.0, 1.0, 1.41, 1.41]


def _kweight_coeffs(sr, device, dtype):
    m = pyln.Meter(sr)
    stages = []
    for st in m._filters.values():
        b = torch.tensor(np.asarray(st.b, dtype=np.float64), device=device, dtype=dtype)
        a = torch.tensor(np.asarray(st.a, dtype=np.float64), device=device, dtype=dtype)
        stages.append((b, a))
    return stages


def kweight(x: torch.Tensor, sr: int) -> torch.Tensor:
    """Apply BS.1770 K-weighting. x: (batch, ch, samples) -> same shape."""
    y = x
    for b, a in _kweight_coeffs(sr, x.device, x.dtype):
        y = torchaudio.functional.lfilter(y, a, b, clamp=False)
    return y


def _channel_power(x: torch.Tensor) -> torch.Tensor:
    """Channel-weighted mean-square contribution, summed over channels.
    x: (batch, ch, samples) -> (batch, samples)."""
    ch = x.shape[1]
    w = torch.tensor(_CHANNEL_GAINS[:ch], device=x.device, dtype=x.dtype).view(1, ch, 1)
    return (w * x ** 2).sum(dim=1)


def integrated_lufs(x: torch.Tensor, sr: int) -> torch.Tensor:
    """Differentiable (ungated) loudness proxy in LUFS. For stationary-ish
    program this tracks the gated integrated value closely. x:(b,ch,s)->(b,)."""
    z = _channel_power(kweight(x, sr))
    ms = z.mean(dim=1).clamp_min(1e-12)
    return _OFFSET + 10.0 * torch.log10(ms)


def short_term_lufs(x: torch.Tensor, sr: int, win: float = 3.0,
                    hop: float = 0.1) -> torch.Tensor:
    """Differentiable short-term LUFS trajectory. x:(b,ch,s)->(b,frames)."""
    z = _channel_power(kweight(x, sr)).unsqueeze(1)  # (b,1,s)
    win_n = max(1, int(round(win * sr)))
    hop_n = max(1, int(round(hop * sr)))
    if z.shape[-1] < win_n:
        win_n = z.shape[-1]
    ms = F.avg_pool1d(z, kernel_size=win_n, stride=hop_n).squeeze(1)
    return _OFFSET + 10.0 * torch.log10(ms.clamp_min(1e-12))


# ITU-R BS.1770-4 4x true-peak FIR (same coefficients as metrics.py), as a
# differentiable conv1d upsampler.
from automaster.metrics import _TP_PHASES as _TP_PHASES_NP


def true_peak_db(x: torch.Tensor, sr: int) -> torch.Tensor:
    """Differentiable true-peak (dBTP). x:(b,ch,s)->(b,)."""
    b, ch, s = x.shape
    phases = torch.tensor(_TP_PHASES_NP, device=x.device, dtype=x.dtype)  # (4,12)
    k = phases.shape[1]
    w = phases.unsqueeze(1)  # (4,1,12) -> treat 4 phases as out-channels
    xr = x.reshape(b * ch, 1, s)
    pad = k - 1
    up = F.conv1d(F.pad(xr, (pad, 0)), w)  # (b*ch, 4, s)
    peak = up.abs().amax(dim=(1, 2)).reshape(b, ch).amax(dim=1).clamp_min(1e-12)
    return 20.0 * torch.log10(peak)


class MasteringLoss(torch.nn.Module):
    """w1·short-term-LUFS-envelope + w2·integrated-LUFS + w3·TP hinge
    + w4·multi-res STFT (timbre/EQ)."""

    def __init__(self, sr, w_env=1.0, w_int=1.0, w_tp=1.0, w_stft=1.0,
                 ceiling_db=-1.0):
        super().__init__()
        self.sr = sr
        self.w_env, self.w_int, self.w_tp, self.w_stft = w_env, w_int, w_tp, w_stft
        self.ceiling_db = ceiling_db
        self._stft = None
        if w_stft > 0:
            import auraloss
            self._stft = auraloss.freq.MultiResolutionSTFTLoss(sample_rate=sr)

    def forward(self, y_hat: torch.Tensor, y_ref: torch.Tensor) -> torch.Tensor:
        env = F.l1_loss(short_term_lufs(y_hat, self.sr),
                        short_term_lufs(y_ref, self.sr))
        integ = F.l1_loss(integrated_lufs(y_hat, self.sr),
                          integrated_lufs(y_ref, self.sr))
        tp = true_peak_db(y_hat, self.sr)
        hinge = torch.relu(tp - self.ceiling_db).mean()
        loss = self.w_env * env + self.w_int * integ + self.w_tp * hinge
        if self._stft is not None:
            loss = loss + self.w_stft * self._stft(y_hat, y_ref)
        return loss
