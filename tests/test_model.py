"""Phase 4B / 5 — editor model aggregation, save/load, and rendering."""
import numpy as np
import pytest

from automaster import model as M
from automaster import render, metrics, dsp_diff
from conftest import sine, db_to_lin

SR = 48000


def _fake_fits():
    base = {n: 0.0 for n in dsp_diff.PARAM_NAMES}
    base.update(comp_ratio=2.0, comp_makeup_db=1.0)
    fits = []
    for i, ls in enumerate([5.0, 6.0, 7.0, 30.0]):  # last is an outlier
        th = dict(base)
        th["ls_gain_db"] = ls
        fits.append({"theta": th, "residual": 0.05 if i < 3 else 0.4, "lufs_err": 0.1})
    return fits


def test_from_fits_median_robust_to_outlier():
    m = M.EditorModel.from_fits("boris", _fake_fits(), target_lufs=-17.0)
    # median of [5,6,7,30] = 6.5 — the 30 outlier doesn't dominate
    assert abs(m.theta["ls_gain_db"] - 6.5) < 1e-6
    assert "gain_db" not in m.theta  # gain is solved at render, not stored


def test_drop_high_residual():
    m = M.EditorModel.from_fits("boris", _fake_fits(), target_lufs=-17.0,
                                drop_high_residual=0.1)
    # the 0.4-residual outlier is dropped -> median of [5,6,7] = 6
    assert abs(m.theta["ls_gain_db"] - 6.0) < 1e-6
    assert m.n_pairs == 3


def test_save_load_roundtrip(tmp_path):
    m = M.EditorModel.from_fits("boris", _fake_fits(), target_lufs=-17.0)
    p = m.save(tmp_path / "boris.json")
    m2 = M.EditorModel.load(p)
    assert m2.editor == "boris" and m2.target_lufs == -17.0
    assert m2.theta == m.theta


def test_render_hits_target_and_ceiling():
    m = M.EditorModel(editor="boris", target_lufs=-16.0,
                      theta={n: 0.0 for n in M.PRESET_PARAMS} | {"comp_ratio": 1.0})
    x = sine(220.0, 4.0, SR, amp=db_to_lin(-30.0))
    y = render.render(x, SR, m, apply_limiter=True, ceiling_db=-1.0)
    assert abs(metrics.integrated_lufs(y, SR) - (-16.0)) < 0.6
    assert metrics.true_peak_db(y, SR) <= -1.0 + 1e-6


def test_render_idempotent_ish():
    """Re-mastering an already-mastered signal must not blow up."""
    m = M.EditorModel(editor="boris", target_lufs=-16.0,
                      theta={n: 0.0 for n in M.PRESET_PARAMS} | {"comp_ratio": 1.0})
    x = sine(220.0, 4.0, SR, amp=db_to_lin(-30.0))
    y1 = render.render(x, SR, m, ceiling_db=-1.0)
    y2 = render.render(y1, SR, m, ceiling_db=-1.0)
    assert metrics.true_peak_db(y2, SR) <= -1.0 + 1e-6
    assert abs(metrics.integrated_lufs(y2, SR) - (-16.0)) < 0.6
    assert np.all(np.isfinite(y2))
