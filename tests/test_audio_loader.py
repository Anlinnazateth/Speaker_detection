"""
Tests for src/audio_loader.py — audio validation and normalization.
"""

import numpy as np
import pytest

from src.audio_loader import validate_audio
from src.config import PipelineConfig


class TestValidStereoLoading:
    """Test that valid stereo audio passes through correctly."""

    def test_stereo_sine_returns_dict(self, stereo_sine, sample_rate):
        result = validate_audio(stereo_sine, sample_rate)
        assert isinstance(result, dict)

    def test_stereo_sine_keys_present(self, stereo_sine, sample_rate):
        result = validate_audio(stereo_sine, sample_rate)
        assert "audio" in result
        assert "sr" in result
        assert "duration" in result
        assert "n_samples" in result

    def test_stereo_output_shape(self, stereo_sine, sample_rate):
        result = validate_audio(stereo_sine, sample_rate)
        assert result["audio"].shape == (sample_rate, 2)

    def test_sample_rate_preserved(self, stereo_sine, sample_rate):
        result = validate_audio(stereo_sine, sample_rate)
        assert result["sr"] == sample_rate

    def test_duration_correct(self, stereo_sine, sample_rate):
        result = validate_audio(stereo_sine, sample_rate)
        assert abs(result["duration"] - 1.0) < 1e-6

    def test_n_samples_correct(self, stereo_sine, sample_rate):
        result = validate_audio(stereo_sine, sample_rate)
        assert result["n_samples"] == sample_rate


class TestMonoRejection:
    """Test that mono (1-D) audio is rejected."""

    def test_mono_raises_value_error(self, mono_signal, sample_rate):
        with pytest.raises(ValueError, match="mono"):
            validate_audio(mono_signal, sample_rate)

    def test_3_channel_raises_value_error(self, sample_rate):
        audio_3ch = np.zeros((sample_rate, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="stereo"):
            validate_audio(audio_3ch, sample_rate)

    def test_4d_array_raises(self, sample_rate):
        bad_shape = np.zeros((10, 2, 2), dtype=np.float32)
        with pytest.raises((ValueError, AssertionError)):
            validate_audio(bad_shape, sample_rate)


class TestSampleRateValidation:
    """Test sample rate warnings and acceptance."""

    def test_valid_sample_rate_no_error(self, stereo_sine):
        result = validate_audio(stereo_sine, 16000)
        assert result["sr"] == 16000

    def test_high_sample_rate_accepted(self, stereo_sine):
        result = validate_audio(stereo_sine, 44100)
        assert result["sr"] == 44100

    def test_low_sample_rate_produces_warning(self, stereo_sine, capsys):
        """Below min_sample_rate should print a warning but not raise."""
        result = validate_audio(stereo_sine, 8000)
        assert result["sr"] == 8000
        captured = capsys.readouterr()
        assert "WARNING" in captured.out


class TestPeakNormalization:
    """Test that audio is peak-normalized to [-1, 1]."""

    def test_loud_signal_normalized(self, sample_rate):
        audio = np.ones((sample_rate, 2), dtype=np.float32) * 5.0
        result = validate_audio(audio, sample_rate)
        assert np.max(np.abs(result["audio"])) <= 1.0 + 1e-6

    def test_quiet_signal_normalized(self, sample_rate):
        audio = np.ones((sample_rate, 2), dtype=np.float32) * 0.1
        result = validate_audio(audio, sample_rate)
        assert abs(np.max(np.abs(result["audio"])) - 1.0) < 1e-6

    def test_silence_stays_silent(self, stereo_silence, sample_rate):
        result = validate_audio(stereo_silence, sample_rate)
        assert np.max(np.abs(result["audio"])) == 0.0

    def test_already_normalized_unchanged(self, sample_rate):
        t = np.linspace(0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
        left = np.sin(2 * np.pi * 440 * t)
        left /= np.max(np.abs(left))
        right = left.copy()
        audio = np.column_stack([left, right])
        result = validate_audio(audio, sample_rate)
        assert abs(np.max(np.abs(result["audio"])) - 1.0) < 1e-6


class TestDtypeEnforcement:
    """Test that the output is always float32."""

    def test_float64_converted(self, sample_rate):
        audio = np.random.randn(sample_rate, 2).astype(np.float64)
        result = validate_audio(audio, sample_rate)
        assert result["audio"].dtype == np.float32

    def test_int16_converted(self, sample_rate):
        audio = (np.random.randn(sample_rate, 2) * 1000).astype(np.int16)
        result = validate_audio(audio, sample_rate)
        assert result["audio"].dtype == np.float32

    def test_float32_preserved(self, stereo_sine, sample_rate):
        result = validate_audio(stereo_sine, sample_rate)
        assert result["audio"].dtype == np.float32
