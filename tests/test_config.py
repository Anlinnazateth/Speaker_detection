"""
Tests for src/config.py — PipelineConfig dataclass.
"""

import pytest

from src.config import PipelineConfig


class TestPipelineConfigDefaults:
    """Verify that default values match the documented pipeline settings."""

    def test_default_sample_rate(self, default_config):
        assert default_config.min_sample_rate == 16000

    def test_default_vad_threshold(self, default_config):
        assert default_config.vad_threshold == 0.35

    def test_default_mic_distance(self, default_config):
        assert default_config.mic_distance == 0.20

    def test_default_speed_of_sound(self, default_config):
        assert default_config.speed_of_sound == 343.0

    def test_default_embedding_dim(self, default_config):
        assert default_config.embedding_dim == 192

    def test_default_cluster_max_k(self, default_config):
        assert default_config.cluster_max_k == 10

    def test_default_denoise_enabled(self, default_config):
        assert default_config.denoise_enabled is True

    def test_default_denoise_method(self, default_config):
        assert default_config.denoise_method == "wiener"

    def test_default_highpass_cutoff(self, default_config):
        assert default_config.highpass_cutoff == 80.0

    def test_default_target_rms_dbfs(self, default_config):
        assert default_config.target_rms_dbfs == -20.0

    def test_default_gcc_frame_len(self, default_config):
        assert default_config.gcc_frame_len == 0.04

    def test_default_kalman_q(self, default_config):
        assert default_config.kalman_q == 1.0

    def test_default_movement_threshold(self, default_config):
        assert default_config.movement_threshold == 10.0


class TestPipelineConfigCustom:
    """Verify custom values can be set on construction."""

    def test_custom_sample_rate(self):
        cfg = PipelineConfig(min_sample_rate=8000)
        assert cfg.min_sample_rate == 8000

    def test_custom_vad_threshold(self):
        cfg = PipelineConfig(vad_threshold=0.7)
        assert cfg.vad_threshold == 0.7

    def test_custom_mic_distance(self):
        cfg = PipelineConfig(mic_distance=0.50)
        assert cfg.mic_distance == 0.50

    def test_custom_denoise_method(self):
        cfg = PipelineConfig(denoise_method="spectral")
        assert cfg.denoise_method == "spectral"

    def test_custom_cluster_max_k(self):
        cfg = PipelineConfig(cluster_max_k=5)
        assert cfg.cluster_max_k == 5

    def test_custom_highpass_cutoff(self):
        cfg = PipelineConfig(highpass_cutoff=100.0)
        assert cfg.highpass_cutoff == 100.0

    def test_multiple_custom_values(self):
        cfg = PipelineConfig(
            min_sample_rate=44100,
            vad_threshold=0.5,
            mic_distance=0.10,
            denoise_method="none",
        )
        assert cfg.min_sample_rate == 44100
        assert cfg.vad_threshold == 0.5
        assert cfg.mic_distance == 0.10
        assert cfg.denoise_method == "none"


class TestPipelineConfigValidation:
    """Test parameter bounds and logical constraints."""

    def test_vad_threshold_in_range(self, default_config):
        """VAD threshold should be between 0 and 1."""
        assert 0.0 <= default_config.vad_threshold <= 1.0

    def test_mic_distance_positive(self, default_config):
        """Microphone distance must be positive."""
        assert default_config.mic_distance > 0

    def test_speed_of_sound_reasonable(self, default_config):
        """Speed of sound should be in the realistic range (300-370 m/s)."""
        assert 300.0 <= default_config.speed_of_sound <= 370.0

    def test_highpass_cutoff_positive(self, default_config):
        """High-pass cutoff frequency must be positive."""
        assert default_config.highpass_cutoff > 0

    def test_cluster_min_members_at_least_one(self, default_config):
        """Minimum cluster members should be at least 1."""
        assert default_config.cluster_min_members >= 1

    def test_gcc_frame_len_positive(self, default_config):
        """GCC frame length must be positive."""
        assert default_config.gcc_frame_len > 0

    def test_gcc_hop_len_positive(self, default_config):
        """GCC hop length must be positive."""
        assert default_config.gcc_hop_len > 0

    def test_gcc_hop_len_leq_frame_len(self, default_config):
        """GCC hop length should not exceed frame length."""
        assert default_config.gcc_hop_len <= default_config.gcc_frame_len

    def test_embedding_dim_positive(self, default_config):
        """Embedding dimension must be positive."""
        assert default_config.embedding_dim > 0

    def test_wiener_gain_floor_between_zero_and_one(self, default_config):
        """Wiener gain floor should be in [0, 1]."""
        assert 0.0 <= default_config.wiener_gain_floor <= 1.0

    def test_bandpass_lo_less_than_hi(self, default_config):
        """GCC band-pass lower edge must be below upper edge."""
        assert default_config.gcc_bandpass_lo < default_config.gcc_bandpass_hi

    def test_target_rms_dbfs_negative(self, default_config):
        """RMS normalization target should be negative (below full scale)."""
        assert default_config.target_rms_dbfs < 0
