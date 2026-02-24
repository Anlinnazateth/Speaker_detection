"""
Shared pytest fixtures for the Speaker Detection test suite.

All fixtures generate synthetic data using numpy so no real audio files
are required.
"""

import numpy as np
import pytest

from src.config import PipelineConfig


# ──────────────────────────────────────────────────────────────
# Configuration fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def default_config():
    """Return a PipelineConfig with all default values."""
    return PipelineConfig()


@pytest.fixture
def fast_config():
    """
    Return a PipelineConfig tuned for fast test execution.

    Disables denoising and uses relaxed thresholds so that synthetic
    signals pass validation without needing realistic audio content.
    """
    return PipelineConfig(
        denoise_enabled=False,
        denoise_method="none",
        vad_threshold=0.3,
        gcc_energy_floor=1e-8,
        gcc_peak_to_mean_min=1.0,
        gcc_confidence_min=0.01,
    )


# ──────────────────────────────────────────────────────────────
# Audio array fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_rate():
    """Standard sample rate used across tests."""
    return 16000


@pytest.fixture
def stereo_silence(sample_rate):
    """One second of stereo silence as float32."""
    n_samples = sample_rate
    return np.zeros((n_samples, 2), dtype=np.float32)


@pytest.fixture
def stereo_sine(sample_rate):
    """
    One second of stereo audio: 440 Hz sine on the left, 880 Hz sine on
    the right, peak-normalized to 0.9.
    """
    t = np.linspace(0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
    left = 0.9 * np.sin(2 * np.pi * 440 * t)
    right = 0.9 * np.sin(2 * np.pi * 880 * t)
    audio = np.column_stack([left, right]).astype(np.float32)
    return audio


@pytest.fixture
def stereo_noise(sample_rate):
    """
    One second of stereo white noise, peak-normalized to [-1, 1].
    Seed is fixed for reproducibility.
    """
    rng = np.random.RandomState(42)
    audio = rng.randn(sample_rate, 2).astype(np.float32)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    return audio


@pytest.fixture
def mono_signal(sample_rate):
    """One second of mono audio (1-D array) for rejection tests."""
    t = np.linspace(0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture
def stereo_delayed(sample_rate):
    """
    Stereo pair where the right channel is a delayed copy of the left.
    Delay is 5 samples (~0.3125 ms at 16 kHz).  Useful for GCC-PHAT and
    azimuth tests with a known ground-truth TDOA.
    """
    delay_samples = 5
    t = np.linspace(0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
    signal = np.zeros(sample_rate, dtype=np.float32)
    for freq in [500, 1000, 1500, 2000, 2500]:
        signal += np.sin(2 * np.pi * freq * t)
    signal /= np.max(np.abs(signal))

    left = signal.copy()
    right = np.zeros_like(signal)
    right[delay_samples:] = signal[:-delay_samples]

    audio = np.column_stack([left, right]).astype(np.float32)
    return audio, delay_samples
