"""Phase 1 — dataset indexing and leakage-free splits."""
import numpy as np
import soundfile as sf
import pytest

from automaster import dataset
from conftest import sine, db_to_lin

SR = 48000


@pytest.fixture
def fake_corpus(tmp_path):
    """Lay out a tiny corpus on disk following the naming convention:
    <root>/<editor>/<id>_before.wav and _after.wav."""
    ids = {
        "boris": ["clipA", "clipB", "clipC"],
        "kim": ["clipD", "clipE"],
    }
    for editor, clip_ids in ids.items():
        d = tmp_path / editor
        d.mkdir()
        for cid in clip_ids:
            x = sine(440.0, 1.0, SR, amp=db_to_lin(-18.0))
            sf.write(d / f"{cid}_before.wav", x, SR)
            sf.write(d / f"{cid}_after.wav", x * db_to_lin(3.0), SR)
    return tmp_path


def test_discovers_all_pairs(fake_corpus):
    pairs = list(dataset.iter_pairs(fake_corpus))
    assert len(pairs) == 5
    assert {p.editor for p in pairs} == {"boris", "kim"}
    for p in pairs:
        assert p.before_path.exists() and p.after_path.exists()


def test_filter_by_editor(fake_corpus):
    boris = list(dataset.iter_pairs(fake_corpus, editor="boris"))
    assert len(boris) == 3
    assert all(p.editor == "boris" for p in boris)


def test_split_no_clip_leakage(fake_corpus):
    pairs = list(dataset.iter_pairs(fake_corpus))
    train, val = dataset.split(pairs, val_frac=0.4, seed=0)
    train_ids = {p.clip_id for p in train}
    val_ids = {p.clip_id for p in val}
    assert train_ids.isdisjoint(val_ids)
    assert len(train) + len(val) == len(pairs)


def test_split_is_deterministic(fake_corpus):
    pairs = list(dataset.iter_pairs(fake_corpus))
    s1 = dataset.split(pairs, val_frac=0.4, seed=7)
    s2 = dataset.split(pairs, val_frac=0.4, seed=7)
    assert [p.clip_id for p in s1[0]] == [p.clip_id for p in s2[0]]


def test_pair_loads_audio(fake_corpus):
    p = next(dataset.iter_pairs(fake_corpus, editor="kim"))
    b, sr_b = p.load_before()
    a, sr_a = p.load_after()
    assert sr_b == SR and sr_a == SR
    assert len(b) > 0 and len(a) > 0
