"""
Tests for src/preprocessing.py — stereo signal conditioning.
"""

import numpy as np
import pytest

from src.preprocessing import preprocess_stereo
from src.config import PipelineConfig


@pytest.fixture
def no_denoise_config():
    """Config with denoising disabled for isolated unit tests."""
    return PipelineConfig(denoise_enabled=False)


class TestDCOffsetRemoval:
    """Test that DC offset is removed from the signal."""

    def test_dc_offset_removed_from_left(self, sample_rate, no_denoise_config):
        """A constant offset on the left channel should be removed."""
        n = sample_rate
        t = np.linspace(0, 1.0, n, endpoint=False, dtype=np.float32)
        left = 0.3 + 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.5 * np.sin(2 * np.pi * 440 * t)
        audio = np.column_stack([left, right]).astype(np.float32)
        audio /= np.max(np.abs(audio))

        result = preprocess_stereo(audio, sample_rate, no_denoise_config)
        assert abs(np.mean(result["left"])) < 0.05

    def test_dc_offset_removed_from_right(self, sample_rate, no_denoise_config):
        """A constant offset on the right channel should be removed."""
        n = sample_rate
        t = np.linspace(0, 1.0, n, endpoint=False, dtype=np.float32)
        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = -0.2 + 0.5 * np.sin(2 * np.pi * 440 * t)
        audio = np.column_stack([left, right]).astype(np.float32)
        audio /= np.max(np.abs(audio))

        result = preprocess_stereo(audio, sample_rate, no_denoise_config)
        assert abs(np.mean(result["right"])) < 0.05


class TestHighPassFilter:
    """Test high-pass filter application."""

    def test_low_frequency_attenuated(self, sample_rate, no_denoise_config):
        """A 20 Hz signal (below the 80 Hz cutoff) should be attenuated."""
        n = sample_rate
        t = np.linspace(0, 1.0, n, endpoint=False, dtype=np.float32)
        tone_20 = 0.9 * np.sin(2 * np.pi * 20 * t)
        audio = np.column_stack([tone_20, tone_20]).astype(np.float32)

        result = preprocess_stereo(audio, sample_rate, no_denoise_config)
        rms_out = np.sqrt(np.mean(result["left"] ** 2))

        tone_500 = 0.9 * np.sin(2 * np.pi * 500 * t)
        audio_ref = np.column_stack([tone_500, tone_500]).astype(np.float32)
        result_ref = preprocess_stereo(audio_ref, sample_rate, no_denoise_config)
        rms_ref = np.sqrt(np.mean(result_ref["left"] ** 2))

        assert rms_out < rms_ref * 0.5

    def test_high_frequency_passes(self, sample_rate, no_denoise_config):
        """A 1 kHz signal should pass through the 80 Hz HPF with minimal loss."""
        n = sample_rate
        t = np.linspace(0, 1.0, n, endpoint=False, dtype=np.float32)
        tone_1k = 0.9 * np.sin(2 * np.pi * 1000 * t)
        audio = np.column_stack([tone_1k, tone_1k]).astype(np.float32)

        result = preprocess_stereo(audio, sample_rate, no_denoise_config)
        rms_out = np.sqrt(np.mean(result["left"] ** 2))
        assert rms_out > 0.01


class TestRMSNormalization:
    """Test RMS normalization to target dBFS."""

    def test_output_rms_near_target(self, sample_rate, no_denoise_config):
        """The RMS of the output should be close to the configured target."""
        n = sample_rate
        t = np.linspace(0, 1.0, n, endpoint=False, dtype=np.float32)
        tone = 0.5 * np.sin(2 * np.pi * 1000 * t)
        audio = np.column_stack([tone, tone]).astype(np.float32)

        result = preprocess_stereo(audio, sample_rate, no_denoise_config)
        rms_left = np.sqrt(np.mean(result["left"] ** 2))

        target_rms = 10.0 ** (no_denoise_config.target_rms_dbfs / 20.0)
        assert abs(rms_left - target_rms) < target_rms * 0.5

    def test_silence_does_not_explode(self, stereo_silence, sample_rate, no_denoise_config):
        """Silence should not be amplified to infinity."""
        result = preprocess_stereo(stereo_silence, sample_rate, no_denoise_config)
        assert np.all(np.isfinite(result["left"]))
        assert np.all(np.isfinite(result["right"]))


class TestMonoDownmix:
    """Test the mono downmix output."""

    def test_mono_is_1d(self, stereo_sine, sample_rate, no_denoise_config):
        result = preprocess_stereo(stereo_sine, sample_rate, no_denoise_config)
        assert result["mono"].ndim == 1

    def test_mono_length_matches_input(self, stereo_sine, sample_rate, no_denoise_config):
        result = preprocess_stereo(stereo_sine, sample_rate, no_denoise_config)
        assert len(result["mono"]) == stereo_sine.shape[0]

    def test_mono_dtype_float32(self, stereo_sine, sample_rate, no_denoise_config):
        result = preprocess_stereo(stereo_sine, sample_rate, no_denoise_config)
        assert result["mono"].dtype == np.float32

    def test_mono_is_average_of_channels(self, sample_rate, no_denoise_config):
        """With identical channels, mono should be very similar to either channel."""
        n = sample_rate
        t = np.linspace(0, 1.0, n, endpoint=False, dtype=np.float32)
        tone = 0.9 * np.sin(2 * np.pi * 1000 * t)
        audio = np.column_stack([tone, tone]).astype(np.float32)
        result = preprocess_stereo(audio, sample_rate, no_denoise_config)

        correlation = np.corrcoef(result["left"], result["mono"])[0, 1]
        assert correlation > 0.99

    def test_output_has_all_keys(self, stereo_sine, sample_rate, no_denoise_config):
        result = preprocess_stereo(stereo_sine, sample_rate, no_denoise_config)
        assert "left" in result
        assert "right" in result
        assert "mono" in result
        assert "sr" in result
        assert "diagnostics" in result

    def test_diagnostics_has_expected_fields(self, stereo_sine, sample_rate, no_denoise_config):
        result = preprocess_stereo(stereo_sine, sample_rate, no_denoise_config)
        diag = result["diagnostics"]
        assert "max_amplitude_left" in diag
        assert "max_amplitude_right" in diag
        assert "rms_left" in diag
        assert "rms_right" in diag
        assert "inter_channel_correlation" in diag
