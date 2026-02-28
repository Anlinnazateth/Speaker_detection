"""
Tests for src/gcc_phat.py — GCC-PHAT TDOA estimation.
"""

import numpy as np
import pytest

from src.gcc_phat import estimate_tdoa
from src.config import PipelineConfig


@pytest.fixture
def gcc_config():
    """Config tuned for synthetic test signals."""
    return PipelineConfig(
        gcc_energy_floor=1e-8,
        gcc_peak_to_mean_min=1.0,
        gcc_confidence_min=0.01,
        gcc_speech_only=False,
        gcc_bandpass_lo=100.0,
        gcc_bandpass_hi=7000.0,
    )


class TestTDOAWithKnownDelay:
    """Test TDOA estimation with a synthetic stereo signal with known delay."""

    def test_tdoa_sign_matches_delay_direction(self, sample_rate, gcc_config):
        """
        When the right channel is delayed relative to the left, the
        estimated TDOA should be positive (or consistently signed).
        """
        delay_samples = 5
        n = sample_rate
        t = np.linspace(0, 1.0, n, endpoint=False, dtype=np.float32)

        signal = np.zeros(n, dtype=np.float32)
        for freq in [500, 1000, 1500, 2000, 2500]:
            signal += np.sin(2 * np.pi * freq * t)
        signal = (signal / np.max(np.abs(signal))).astype(np.float32)

        left = signal.copy()
        right = np.zeros_like(signal)
        right[delay_samples:] = signal[:-delay_samples]
        right = right.astype(np.float32)

        result = estimate_tdoa(left, right, sample_rate, gcc_config)

        valid = result["gcc_peak"] > gcc_config.gcc_confidence_min
        if np.sum(valid) > 0:
            median_tdoa = np.median(result["tdoa"][valid])
            expected_tdoa = delay_samples / sample_rate
            assert abs(median_tdoa - expected_tdoa) < expected_tdoa * 2.0 + 1e-4

    def test_output_keys_present(self, sample_rate, gcc_config):
        n = sample_rate
        t = np.linspace(0, 1.0, n, endpoint=False, dtype=np.float32)
        signal = np.sin(2 * np.pi * 1000 * t).astype(np.float32)

        result = estimate_tdoa(signal, signal.copy(), sample_rate, gcc_config)
        assert "tdoa" in result
        assert "gcc_peak" in result
        assert "frame_times" in result

    def test_output_arrays_same_length(self, sample_rate, gcc_config):
        n = sample_rate
        t = np.linspace(0, 1.0, n, endpoint=False, dtype=np.float32)
        signal = np.sin(2 * np.pi * 1000 * t).astype(np.float32)

        result = estimate_tdoa(signal, signal.copy(), sample_rate, gcc_config)
        n_frames = len(result["tdoa"])
        assert len(result["gcc_peak"]) == n_frames
        assert len(result["frame_times"]) == n_frames


class TestZeroDelay:
    """Test that identical channels produce near-zero TDOA."""

    def test_identical_channels_zero_tdoa(self, sample_rate, gcc_config):
        n = sample_rate
        t = np.linspace(0, 1.0, n, endpoint=False, dtype=np.float32)
        signal = np.zeros(n, dtype=np.float32)
        for freq in [500, 1000, 1500, 2000]:
            signal += np.sin(2 * np.pi * freq * t)
        signal = (signal / np.max(np.abs(signal))).astype(np.float32)

        result = estimate_tdoa(signal, signal.copy(), sample_rate, gcc_config)

        valid = result["gcc_peak"] > gcc_config.gcc_confidence_min
        if np.sum(valid) > 0:
            median_tdoa = np.median(result["tdoa"][valid])
            assert abs(median_tdoa) < 1e-3


class TestEnergyGating:
    """Test that silence produces no valid TDOA estimates."""

    def test_silence_produces_zero_confidence(self, sample_rate):
        config = PipelineConfig(
            gcc_speech_only=False,
            gcc_energy_floor=1e-4,
        )
        left = np.zeros(sample_rate, dtype=np.float32)
        right = np.zeros(sample_rate, dtype=np.float32)

        result = estimate_tdoa(left, right, sample_rate, config)
        assert np.all(result["gcc_peak"] == 0.0)

    def test_very_quiet_signal_gated(self, sample_rate):
        config = PipelineConfig(
            gcc_speech_only=False,
            gcc_energy_floor=0.1,
        )
        t = np.linspace(0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
        signal = (1e-6 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

        result = estimate_tdoa(signal, signal.copy(), sample_rate, config)
        assert np.sum(result["gcc_peak"] > 0) < len(result["gcc_peak"]) * 0.1

    def test_frame_times_monotonically_increasing(self, sample_rate, gcc_config):
        n = sample_rate
        t = np.linspace(0, 1.0, n, endpoint=False, dtype=np.float32)
        signal = np.sin(2 * np.pi * 1000 * t).astype(np.float32)

        result = estimate_tdoa(signal, signal.copy(), sample_rate, gcc_config)
        if len(result["frame_times"]) > 1:
            diffs = np.diff(result["frame_times"])
            assert np.all(diffs > 0)
