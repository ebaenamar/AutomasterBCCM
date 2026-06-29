"""Train the Boris model: fit θ per before/after pair, aggregate to a preset.

    python scripts/train_boris.py

Writes:
  data/reports/boris_fits.json   per-pair fitted θ + residuals
  models/boris.json              the deployable EditorModel
"""
import json
import time
from pathlib import Path

import numpy as np

from automaster import dataset, align, fit, metrics
from automaster.model import EditorModel

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw"


def main():
    pairs = list(dataset.iter_pairs(DATA, editor="boris"))
    print(f"{len(pairs)} Boris pairs")
    fits, after_lufs = [], []
    for p in pairs:
        t0 = time.time()
        b, sr_b = p.load_before(mono=True)
        a, sr_a = p.load_after(mono=True)
        b_al, a_al, sr = align.align(b, a, sr_b, sr_a)
        # measure the master's loudness on the stereo file (delivery level)
        a_st, sr_st = p.load_after(mono=False)
        after_lufs.append(metrics.integrated_lufs(a_st, sr_st))

        r = fit.fit_pair(b_al, a_al, sr, iters=150, excerpt_s=6.0)
        r["clip_id"] = p.clip_id
        fits.append(r)
        print(f"  {p.clip_id}: residual={r['residual']:.4f} "
              f"lufs_err={r['lufs_err']:.2f} ls={r['theta']['ls_gain_db']:+.1f} "
              f"ratio={r['theta']['comp_ratio']:.1f} [{time.time()-t0:.0f}s]")

    (ROOT / "data" / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "reports" / "boris_fits.json").write_text(json.dumps(fits, indent=2))

    target = float(np.median(after_lufs))
    model = EditorModel.from_fits("boris", fits, target_lufs=target)
    model.notes = (f"Preset from {len(fits)} pairs; target = median master LUFS "
                   f"({target:.1f}); EQ/comp = median of per-pair fits.")
    model.save(ROOT / "models" / "boris.json")
    print(f"\ntarget_lufs={target:.2f}")
    print("preset theta:", json.dumps(model.theta, indent=2))
    print(f"saved -> models/boris.json")


if __name__ == "__main__":
    main()
