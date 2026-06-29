# AutoMaster (BCCM)

Learn the manual YouTube mastering BCCM editors do in DaVinci Resolve, from
before/after pairs, then package it for the BCCM workflow. Built TDD, phase by
phase, per `plan.md`.

## Status

| Phase | Scope | State |
|-------|-------|-------|
| 0 | Measurement primitives (LUFS, LRA, true-peak) | ✅ done — 10 tests, 85% cov |
| 1 | Dataset, alignment, deltas + EQ analysis | ✅ done — 14 tests; real-data report + EQ signature |
| 2 | Deterministic baseline (target ≈ −17, configurable TP) | ✅ done — 6 tests; validated on a real track |
| 3 | Differentiable chain (gain → EQ → comp) + loudness losses | ✅ done — 10 tests; θ* recovery verified |
| 4 | Per-pair fit + Boris model ("twin") | ✅ done — fits all 5 pairs, residual <0.04 |
| 5 | Production renderer + web app | ✅ done — `render.py`, FastAPI + UI, CLI |
| 6 | Packaging (Resolve script / standalone / VST3) | ⬜ future |

## Use it

```bash
# CLI
automaster master input.wav -o mastered.wav --editor boris
automaster master input.wav -o hot.wav --editor boris --no-limiter   # replicate, hot

# Web app with Docker (bundles ffmpeg, so .mp4 works) — http://localhost:8000
docker compose up --build

# ...or without Docker (needs ffmpeg on PATH for .mp4)
./.venv/bin/uvicorn app.server:app --port 8000

# (Re)train the model from the before/after pairs
./.venv/bin/python scripts/train_boris.py
```

Deploy: see [DEPLOY.md](DEPLOY.md) (Render blueprint included; Vercel for the
static frontend).

## The Boris model

`models/boris.json`, learned from 5 before/after pairs. Robust, consistent
signature: **target ≈ −17.5 LUFS** and **heavy compression (ratio ~7–8)** — the
chain reproduces every master with residual < 0.04. The per-band EQ is weakly
identified (gain↔EQ degeneracy in the fit), so the reliable learned moves are
loudness + compression; the bass-boost the Phase-1 analysis found is best read
from `data/reports/eq_curves.png`. More pairs (and Kim's set) would let the
`predict()` interface graduate from a preset to a real per-clip regressor.

Run the suite: `./.venv/bin/python -m pytest -q`

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -e .
```

`ffmpeg`/`ffprobe` must be on PATH (used for mp4 decode and as the metric
cross-check oracle).

## Layout

```
src/automaster/
  metrics.py   # BS.1770 LUFS, EBU 3342 LRA, BS.1770-4 true-peak FIR
  io.py        # WAV via soundfile; mp4/m4a/etc. decoded via ffmpeg
  align.py     # cross-correlation offset + common-region trim + resample
  dataset.py   # Pair discovery, editor labels, leakage-free split
  analyze.py   # per-pair deltas, deltas.csv + style scatter
data/raw/<editor>/<clip_id>_{before,after}.<ext>   # corpus (symlinks ok)
data/reports/                                       # deltas.csv, scatter
```

## Key implementation notes

- **True peak** uses the standardised ITU-R BS.1770-4 4× polyphase FIR, not a
  clean resampler. Meters legitimately diverge ~0.5 dB on pure tones near
  Nyquist; mid-band agrees with `ffmpeg ebur128` to <0.15 dB. Downstream
  limiting must carry headroom to absorb meter disagreement.
- **TP must be measured on the stereo signal, not a mono mixdown** — averaging
  L+R hides inter-sample peaks. The deltas table uses mono for LUFS/LRA speed;
  TP-for-delivery is measured per-channel on stereo.
- `dasp-pytorch` signatures were verified against the installed version
  (compressor takes `threshold_db, ratio, attack_ms, release_ms, knee_db,
  makeup_gain_db`).

See `data/reports/FINDINGS.md` for the Phase-1 empirical results that reframe
Phase 2's targets.
