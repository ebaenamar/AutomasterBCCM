"""Synthetic signal generators for oracle-based tests.

Everything is float64 with shape (samples,) for mono or (samples, channels)
for multichannel, matching the convention used across automaster.metrics.
"""
import numpy as np
import pytest

SR = 48000


def sine(freq, dur, sr=SR, amp=1.0, phase=0.0):
    """Mono sine of given peak amplitude `amp`."""
    t = np.arange(int(round(dur * sr))) / sr
    return amp * np.sin(2 * np.pi * freq * t + phase)


def db_to_lin(db):
    return 10.0 ** (db / 20.0)


def lin_to_db(lin):
    return 20.0 * np.log10(np.maximum(lin, 1e-12))


@pytest.fixture
def sr():
    return SR


@pytest.fixture
def sine_1k_full(sr):
    """1 kHz sine at 0 dBFS peak, 5 s."""
    return sine(1000.0, 5.0, sr, amp=1.0)


@pytest.fixture
def sine_1k_minus20(sr):
    """1 kHz sine at -20 dBFS peak, 5 s."""
    return sine(1000.0, 5.0, sr, amp=db_to_lin(-20.0))


@pytest.fixture
def intersample_peak_signal(sr):
    """Sine near Nyquist, phase-shifted so that the true peak sits *between*
    samples. Sample peak is well below the inter-sample (true) peak."""
    # f = sr/4 + a bit, with quarter-sample phase offset, classic ISP demo.
    f = sr / 4.0 + 30.0
    x = sine(f, 2.0, sr, amp=db_to_lin(-1.0), phase=np.pi / 4.0)
    return x


@pytest.fixture
def two_level_signal(sr):
    """Two concatenated 1 kHz tones at -30 and -10 LUFS-ish levels.
    Used to exercise loudness range (LRA)."""
    quiet = sine(1000.0, 6.0, sr, amp=db_to_lin(-30.0))
    loud = sine(1000.0, 6.0, sr, amp=db_to_lin(-10.0))
    return np.concatenate([quiet, loud])
