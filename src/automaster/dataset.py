"""Phase 1 — dataset indexing, editor labelling, leakage-free splits.

Corpus layout convention::

    <root>/<editor>/<clip_id>_before.wav
    <root>/<editor>/<clip_id>_after.wav

``editor`` is the parent directory name (e.g. ``boris`` / ``kim``). Splits
are by clip, seeded, so a clip never lands in two splits and an editor's
clips never leak across the train/val boundary unless intended.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from automaster import io


@dataclass(frozen=True)
class Pair:
    clip_id: str
    editor: str
    before_path: Path
    after_path: Path
    provenance: str = "resolve"  # 'resolve' (trusted) | 'youtube' (excluded)
    edited: bool = False         # True if pair differs by edits, not just level

    def load_before(self, sr=None, mono=False):
        return io.load_audio(self.before_path, sr=sr, mono=mono)

    def load_after(self, sr=None, mono=False):
        return io.load_audio(self.after_path, sr=sr, mono=mono)


def iter_pairs(root, editor: str | None = None):
    """Yield :class:`Pair` for every ``*_before`` with a matching ``*_after``.

    Searches one editor-directory deep. If ``editor`` is given, restricts to
    that subdirectory.
    """
    root = Path(root)
    editor_dirs = [root / editor] if editor else [
        d for d in sorted(root.iterdir()) if d.is_dir()
    ]
    for d in editor_dirs:
        if not d.is_dir():
            continue
        for before in sorted(d.glob("*_before.*")):
            clip_id = before.name.rsplit("_before", 1)[0]
            after = next(iter(d.glob(f"{clip_id}_after.*")), None)
            if after is None:
                continue
            yield Pair(clip_id=clip_id, editor=d.name,
                       before_path=before, after_path=after)


def split(pairs, val_frac: float = 0.2, seed: int = 0):
    """Split pairs into (train, val) by clip, deterministically. No clip
    appears in both splits."""
    pairs = list(pairs)
    rng = random.Random(seed)
    order = pairs[:]
    rng.shuffle(order)
    n_val = max(1, int(round(len(order) * val_frac))) if order else 0
    val = order[:n_val]
    train = order[n_val:]
    return train, val
