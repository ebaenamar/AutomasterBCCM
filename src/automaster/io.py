"""Audio I/O: read/write WAV and decode audio from video containers (mp4).

Convention across automaster: arrays are float64, shape ``(samples,)`` for
mono or ``(samples, channels)`` for multichannel; sample rate carried
separately. ``soundfile`` handles WAV/FLAC; anything it cannot open
(e.g. mp4) is decoded through the system ``ffmpeg`` binary.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

_FFMPEG = shutil.which("ffmpeg")
_DECODE_EXT = {".mp4", ".m4a", ".mov", ".mkv", ".aac", ".mp3"}


def _decode_with_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    """Decode an arbitrary container to float32 PCM via ffmpeg, preserving
    sample rate and channel count. Returns (samples, channels) or (samples,)."""
    if _FFMPEG is None:
        raise RuntimeError("ffmpeg not found on PATH; cannot decode " + str(path))
    # Probe sample rate and channels.
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    sr, ch = int(probe[0]), int(probe[1])
    raw = subprocess.run(
        [_FFMPEG, "-v", "error", "-i", str(path),
         "-f", "f32le", "-acodec", "pcm_f32le", "-ac", str(ch), "-"],
        capture_output=True, check=True,
    ).stdout
    x = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    if ch > 1:
        x = x.reshape(-1, ch)
    return x, sr


def load_audio(path, sr=None, mono=False) -> tuple[np.ndarray, int]:
    """Load audio from WAV/FLAC (soundfile) or video/compressed (ffmpeg).

    If ``sr`` is given, resample to it. If ``mono``, downmix to a single
    channel by averaging.
    """
    from automaster.align import resample_to  # local import: avoid cycle

    path = Path(path)
    if path.suffix.lower() in _DECODE_EXT:
        x, file_sr = _decode_with_ffmpeg(path)
    else:
        x, file_sr = sf.read(str(path), dtype="float64", always_2d=False)

    if mono and x.ndim == 2:
        x = x.mean(axis=1)
    if sr is not None and sr != file_sr:
        x = resample_to(x, file_sr, sr)
        file_sr = sr
    return x, file_sr


def save_audio(path, x: np.ndarray, sr: int, subtype: str = "PCM_24") -> None:
    """Write audio to disk (WAV/FLAC via soundfile)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(x), sr, subtype=subtype)
