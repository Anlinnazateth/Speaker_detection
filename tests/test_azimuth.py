"""
Tests for src/azimuth.py — TDOA to azimuth conversion.
"""

import numpy as np
import pytest

from src.azimuth import tdoa_to_azimuth
from src.config import PipelineConfig


@pytest.fixture
def azimuth_config():
    """Standard config for azimuth tests."""
    return PipelineConfig(mic_distance=0.20, speed_of_sound=343.0)


class TestZeroTDOA:
    """Test that zero TDOA gives 0 degrees azimuth."""

    def test_single_zero_tdoa(self, azimuth_config):
        tdoa = np.array([0.0])
        azimuth = tdoa_to_azimuth(tdoa, azimuth_config)
        assert abs(azimuth[0]) < 1e-6

    def test_all_zeros_give_zero_azimuth(self, azimuth_config):
        tdoa = np.zeros(10)
        azimuth = tdoa_to_azimuth(tdoa, azimuth_config)
        assert np.allclose(azimuth, 0.0, atol=1e-6)


class TestPositiveNegativeTDOA:
    """Test that positive/negative TDOA gives correct azimuth direction."""

    def test_positive_tdoa_gives_positive_azimuth(self, azimuth_config):
        d = azimuth_config.mic_distance
        c = azimuth_config.speed_of_sound
        tau = (d / c) * np.sin(np.radians(30.0))
        tdoa = np.array([tau])
        azimuth = tdoa_to_azimuth(tdoa, azimuth_config)
        assert abs(azimuth[0] - 30.0) < 0.1

    def test_negative_tdoa_gives_negative_azimuth(self, azimuth_config):
        d = azimuth_config.mic_distance
        c = azimuth_config.speed_of_sound
        tau = -(d / c) * np.sin(np.radians(45.0))
        tdoa = np.array([tau])
        azimuth = tdoa_to_azimuth(tdoa, azimuth_config)
        assert abs(azimuth[0] - (-45.0)) < 0.1

    def test_symmetric_tdoa_gives_symmetric_azimuth(self, azimuth_config):
        d = azimuth_config.mic_distance
        c = azimuth_config.speed_of_sound
        tau = (d / c) * np.sin(np.radians(20.0))
        tdoa = np.array([tau, -tau])
        azimuth = tdoa_to_azimuth(tdoa, azimuth_config)
        assert abs(azimuth[0] + azimuth[1]) < 0.01

    def test_max_tdoa_gives_90_degrees(self, azimuth_config):
        d = azimuth_config.mic_distance
        c = azimuth_config.speed_of_sound
        tau_max = d / c
        tdoa = np.array([tau_max])
        azimuth = tdoa_to_azimuth(tdoa, azimuth_config)
        assert abs(azimuth[0] - 90.0) < 0.1


class TestClippingExtremeValues:
    """Test that out-of-range TDOA values are clipped, not producing NaN."""

    def test_extreme_positive_tdoa_clipped(self, azimuth_config):
        d = azimuth_config.mic_distance
        c = azimuth_config.speed_of_sound
        tau_extreme = 2.0 * (d / c)
        tdoa = np.array([tau_extreme])
        azimuth = tdoa_to_azimuth(tdoa, azimuth_config)
        assert abs(azimuth[0] - 90.0) < 0.1
        assert np.all(np.isfinite(azimuth))

    def test_extreme_negative_tdoa_clipped(self, azimuth_config):
        d = azimuth_config.mic_distance
        c = azimuth_config.speed_of_sound
        tau_extreme = -2.0 * (d / c)
        tdoa = np.array([tau_extreme])
        azimuth = tdoa_to_azimuth(tdoa, azimuth_config)
        assert abs(azimuth[0] - (-90.0)) < 0.1
        assert np.all(np.isfinite(azimuth))

    def test_no_nan_in_output(self, azimuth_config):
        tdoa = np.array([-1.0, -0.001, 0.0, 0.001, 1.0])
        azimuth = tdoa_to_azimuth(tdoa, azimuth_config)
        assert np.all(np.isfinite(azimuth))

    def test_azimuth_range_bounded(self, azimuth_config):
        rng = np.random.RandomState(42)
        tdoa = rng.uniform(-0.01, 0.01, size=100)
        azimuth = tdoa_to_azimuth(tdoa, azimuth_config)
        assert np.all(azimuth >= -90.0)
        assert np.all(azimuth <= 90.0)

    def test_default_config_works(self):
        tdoa = np.array([0.0, 0.0001, -0.0001])
        azimuth = tdoa_to_azimuth(tdoa)
        assert len(azimuth) == 3
        assert np.all(np.isfinite(azimuth))
