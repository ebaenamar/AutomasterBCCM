"""Phase 4B — the editor model ("twin").

With only a handful of Boris pairs (and degenerate gain/makeup/shelf trade-offs
in the fit), a per-clip neural regressor would overfit and isn't credible yet.
The honest, deployable model is a **preset + auto-loudness**: aggregate the
fitted EQ/compression across pairs (median, robust to the EQ-heavy outlier) and
drive loudness to Boris's measured target at render time. The ``predict``
interface takes input features so a real regressor can drop in later (more data
+ Kim's set) without changing callers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

from automaster import dsp_diff

# Params the model carries as a fixed preset. gain_db is NOT among them — it is
# solved at render time to hit ``target_lufs`` regardless of input level.
PRESET_PARAMS = [n for n in dsp_diff.PARAM_NAMES if n != "gain_db"]


@dataclass
class EditorModel:
    editor: str
    target_lufs: float
    theta: dict = field(default_factory=dict)  # physical EQ/comp params
    n_pairs: int = 0
    notes: str = ""

    @staticmethod
    def from_fits(editor, fit_results, target_lufs, drop_high_residual=None):
        """Aggregate per-pair fits into a robust preset (median per param)."""
        results = list(fit_results)
        if drop_high_residual is not None:
            kept = [r for r in results if r["residual"] <= drop_high_residual]
            results = kept or results
        theta = {}
        for p in PRESET_PARAMS:
            theta[p] = float(np.median([r["theta"][p] for r in results]))
        return EditorModel(editor=editor, target_lufs=float(target_lufs),
                           theta=theta, n_pairs=len(results))

    def predict(self, features: dict | None = None) -> dict:
        """Return the physical θ for an input. Preset for now (features
        ignored); the signature is ready for a learned regressor."""
        return dict(self.theta)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))
        return path

    @staticmethod
    def load(path):
        return EditorModel(**json.loads(Path(path).read_text()))
