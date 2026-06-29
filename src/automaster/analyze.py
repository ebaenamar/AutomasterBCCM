"""Phase 1 — per-pair deltas and the style scatter.

This is the decision point of the whole project: the scatter of ΔLUFS vs
ΔLRA, coloured by editor, tells us whether the 'twins' need to be neural
nets or can be straight lines.

    ΔLRA ≈ 0  → pure gain (likely Kim)
    ΔLRA < 0  → compression (likely Boris)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from automaster import align, metrics


def pair_delta(before, after, sr_before, sr_after) -> dict:
    """Compute the loudness/dynamics deltas for a single aligned pair."""
    b, a, sr = align.align(before, after, sr_before, sr_after)

    lufs_b = metrics.integrated_lufs(b, sr)
    lufs_a = metrics.integrated_lufs(a, sr)
    lra_b = metrics.loudness_range(b, sr)
    lra_a = metrics.loudness_range(a, sr)
    tp_b = metrics.true_peak_db(b, sr)
    tp_a = metrics.true_peak_db(a, sr)

    # Residual after best global gain match — a cheap "is this just level?"
    # detector. High residual ⇒ EQ / time-varying moves / edits.
    bm = align._mono(b)
    am = align._mono(a)
    g = np.dot(bm, am) / (np.dot(bm, bm) + 1e-12)  # least-squares scalar gain
    resid = am - g * bm
    resid_db = 10 * np.log10((np.mean(resid ** 2) + 1e-12) / (np.mean(am ** 2) + 1e-12))
    corr = float(np.corrcoef(bm, am)[0, 1])

    return {
        "lufs_before": lufs_b, "lufs_after": lufs_a, "d_lufs": lufs_a - lufs_b,
        "lra_before": lra_b, "lra_after": lra_a, "d_lra": lra_a - lra_b,
        "tp_before_db": tp_b, "tp_after_db": tp_a,
        "fit_gain_db": 20 * np.log10(abs(g) + 1e-12),
        "residual_db": resid_db, "corr": corr,
    }


def eq_curve(before, after, sr_before, sr_after=None, nperseg=8192):
    """Long-term spectral move (dB) the master applies: ``after`` vs ``before``.

    Returns ``(freqs, curve_db)`` where ``curve_db`` is the ratio of average
    power spectra, gain-matched so the broadband level cancels (the curve is
    centred around 0 dB by its low-band mean). This captures the *tonal*
    shaping (EQ) on top of the loudness change. Compression also tilts the
    spectrum, so read this as "net tonal move", not a pure EQ transfer
    function — but it tells us which bands Boris pushes and how much.
    """
    from scipy.signal import welch

    if sr_after is None:
        sr_after = sr_before
    b, a, sr = align.align(before, after, sr_before, sr_after)
    b, a = align._mono(b), align._mono(a)

    f, pb = welch(b, sr, nperseg=nperseg)
    _, pa = welch(a, sr, nperseg=nperseg)
    curve = 10 * np.log10((pa + 1e-12) / (pb + 1e-12))
    # Centre on the low band so broadband gain doesn't dominate the picture.
    lo = curve[(f > 80) & (f < 400)]
    curve = curve - (lo.mean() if lo.size else 0.0)
    return f, curve


def eq_analysis(pairs, out_path, mono=True):
    """Plot every pair's EQ curve plus the mean — Boris's tonal signature."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = []
    fig, ax = plt.subplots(figsize=(8, 5))
    common_f = None
    for p in pairs:
        b, sr_b = p.load_before(mono=mono)
        a, sr_a = p.load_after(mono=mono)
        f, c = eq_curve(b, a, sr_b, sr_a)
        common_f = f
        curves.append(c)
        ax.semilogx(f[1:], c[1:], alpha=0.4, lw=1, label=p.clip_id)
    if curves:
        mean_c = np.mean(np.vstack(curves), axis=0)
        ax.semilogx(common_f[1:], mean_c[1:], color="k", lw=2.5, label="mean")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlim(30, 20000)
    ax.set_ylim(-8, 8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("After − Before (dB, gain-matched)")
    ax.set_title("Boris tonal move (net EQ + compression tilt)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return common_f, curves


def compute_deltas(pairs, sr=None, mono=True) -> pd.DataFrame:
    """Compute deltas for every pair, returning a tidy DataFrame."""
    rows = []
    for p in pairs:
        b, sr_b = p.load_before(mono=mono)
        a, sr_a = p.load_after(mono=mono)
        d = pair_delta(b, a, sr_b, sr_a)
        d.update(clip_id=p.clip_id, editor=p.editor,
                 provenance=p.provenance)
        rows.append(d)
    return pd.DataFrame(rows)


def scatter(df: pd.DataFrame, out_path) -> None:
    """ΔLUFS vs ΔLRA, coloured by editor."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    for editor, sub in df.groupby("editor"):
        ax.scatter(sub["d_lufs"], sub["d_lra"], label=editor, s=80, alpha=0.8)
        for _, r in sub.iterrows():
            ax.annotate(r["clip_id"], (r["d_lufs"], r["d_lra"]),
                        fontsize=7, alpha=0.6)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("ΔLUFS (after − before)")
    ax.set_ylabel("ΔLRA (after − before)")
    ax.set_title("Mastering style: loudness gain vs dynamics change")
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def analyze_corpus(root, out_dir, mono=True):
    """End-to-end: index pairs, compute deltas, write CSV + scatter."""
    from automaster import dataset

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = list(dataset.iter_pairs(root))
    df = compute_deltas(pairs, mono=mono)
    df.to_csv(out_dir / "deltas.csv", index=False)
    scatter(df, out_dir / "deltas_scatter.png")
    return df
