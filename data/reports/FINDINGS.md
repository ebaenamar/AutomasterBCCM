# Phase 1 findings — what the manual masters actually do

Measured on the 5 before/after pairs (mono mixdown for LUFS/LRA; stereo for
delivery true-peak). `before` = Resolve source export, `after` = manual master.

## Deltas (mono)

| clip | ΔLUFS | ΔLRA | best-fit gain | residual | corr | flag |
|------|------:|-----:|--------------:|---------:|-----:|------|
| 2025-11-04-n2 | +4.06 | −2.19 | +3.99 | −15.3 dB | 0.99 | clean |
| 2025-11-04-n4 | +6.66 | −4.50 | +6.93 | −14.2 dB | 0.98 | clean |
| 2026-04-04-n1 | +5.29 | −4.05 | +5.45 | −14.1 dB | 0.98 | clean |
| 2026-04-04-n4 | +2.03 | −3.19 | +0.38 | **−3.8 dB** | **0.76** | **edited?** |
| 2026-04-04-n5 | +1.81 | −2.92 | +2.26 | −16.6 dB | 0.99 | clean |

## Delivery levels (stereo)

| clip | master LUFS | master true-peak |
|------|------------:|-----------------:|
| 2025-11-04-n2 | −17.3 | **+1.24 dBTP** |
| 2025-11-04-n4 | −16.1 | **+6.67 dBTP** |
| 2026-04-04-n1 | −19.0 | **+1.97 dBTP** |
| 2026-04-04-n4 | −17.8 | −0.39 dBTP |
| 2026-04-04-n5 | −17.5 | +0.19 dBTP |

## What this means for the plan

1. **Everyone compresses.** All five ΔLRA are negative (−2.2 to −4.5). The
   "pure-gain Kim" hypothesis is not supported by this sample — the twins need
   at least gain + compression, so Phase 3's differentiable chain is justified.
   There's a single coherent trend (more loudness → more compression) rather
   than two visually separate clusters, but editor labels are unknown so styles
   can't be split yet.

2. **The target is ~−16 to −19 LUFS, not −14.** These masters sit *below* the
   streaming norm. Phase 2's default target (−14) is wrong for matching BCCM;
   use ≈ −17 LUFS (or per-editor) instead.

### EQ signature (gain-matched to 80–400 Hz, `eq_curves.png`)

| band | mean move | note |
|------|----------:|------|
| 30–80 Hz | **+2.5 dB** (up to +6.7 at 30 Hz) | consistent low-shelf boost, all 5 pairs |
| 80–4 kHz | ±0.25 dB | essentially flat |
| 4–16 kHz | ~0, ±2 dB per clip | per-clip: n4 darker (−2 dB HF), n5 brighter |

So Boris's tonal move is **a low-shelf bass boost** plus per-clip HF taste; mids
are left alone. The chain needs at least a low shelf; the regressor must be free
to set the HF shelf per clip. `dasp.parametric_eq` (low-shelf + 4 peaks +
high-shelf) covers it.

---

3. **They do NOT respect a true-peak ceiling.** Four of five masters exceed
   0 dBTP, one by +6.7 dB (2025-11-04-n4 — also the most compressed/loudest).
   The plan assumed a −1 dBTP limiter. To *replicate* BCCM we must allow peaks
   over 0; to *improve* on them we'd add limiting. This is a Boris/Kim call.

4. **2026-04-04-n4 is suspect.** After alignment the gain-matched residual is
   only −3.8 dB and correlation 0.76 (vs −14 dB / 0.99 for clean pairs). The
   raw/export differ by more than level + compression: real EQ/edits, a content
   mismatch, or a wrong raw↔export mapping. Flagged `edited` — exclude from
   level-fitting until triaged. (Recall its raw/export file sizes also differ
   more than the other pairs.)
