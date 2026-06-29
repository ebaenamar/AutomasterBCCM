# AutoMaster (BCCM) — implementation plan

Goal: replicate the manual mastering the BCCM editors (Boris, later Kim) do in
DaVinci Resolve (raise loudness for YouTube without obvious distortion, minimal
compression) from a dataset of before/after pairs, then package it as a usable
tool (Resolve script / standalone / VST3).

Method: **TDD, phase by phase**. Each phase writes tests first (with synthetic
oracles of known response), then the implementation; we don't advance until the
phase passes its acceptance criterion. End-to-end debugging on real data happens
only in the final phase.

> Verify the exact API signatures of installed packages before writing calls
> (`dasp_pytorch.functional.*` in particular varies between versions). Don't
> assume signatures from memory.

## Architecture decisions (non-negotiable)

1. **DSP core = `dasp-pytorch`** (Apache-2.0), used both to *fit* parameters
   (differentiable) and to *render* production output, so what is learned is
   exactly what is applied and a future VST can reimplement the same DSP.
2. **`pedalboard` (GPLv3) dev-only**: fast audio IO, non-differentiable
   baseline, VST3 hosting for A/B. Not part of the distributable core.
3. **True peak via the standardised ITU-R BS.1770-4 4× FIR**, cross-checked
   against ffmpeg. `pyloudnorm` provides integrated LUFS + LRA, not true peak.
4. **Loss in loudness/dynamics space**, not waveform MSE: short-term LUFS
   envelope error + integrated LUFS error + true-peak hinge over the ceiling
   (+ optional multi-resolution STFT for timbre/EQ).
5. **One model per editor** (Boris / Kim — the "twins") plus a deterministic
   baseline as control.

## Phases

- **0 — Measurement primitives** (`metrics.py`): integrated LUFS, LRA, true
  peak. Oracle tests + ffmpeg cross-check.
- **1 — Dataset, alignment, deltas** (`io/align/dataset/analyze`): pair
  discovery, cross-correlation alignment, per-pair ΔLUFS/ΔLRA/TP, style scatter,
  and EQ-curve analysis. The scatter + EQ curve decide whether the twins are
  nets or lines.
- **2 — Deterministic baseline** (`baseline.py`): gain-to-target + conditional
  true-peak limiter; configurable ceiling and a replicate mode; an ffmpeg
  two-pass `loudnorm` second reference.
- **3 — Differentiable chain** (`dsp_diff.py`, `losses.py`): gain → parametric
  EQ → compressor. Key test: recover a known θ*. The true-peak limiter is a
  deterministic render-time stage, not part of the gradient graph.
- **4 — Per-pair fit + regressor** (`fit.py`, `model.py`): fit θ per pair to
  reproduce `after` (the residual flags what the chain can't capture); learn a
  light regressor from input loudness features → θ, per editor.
- **5 — Production renderer + evaluation** (`render.py`, `evaluate.py`): input →
  features → θ → chain → WAV; aggregate metrics + blind A/B export.
- **6 — Packaging**: Resolve script, standalone app, VST3 (JUCE).
- **Final — Integration & end-to-end debug** on real clips.

See `data/reports/FINDINGS.md` for the Phase-1 empirical results that reframe
the targets (≈ −17 LUFS not −14; masters run hot over 0 dBTP; Boris applies a
consistent low-shelf bass boost).
