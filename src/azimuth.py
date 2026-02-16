"""
Module 7: TDOA to Azimuth Conversion
Maps time difference of arrival (seconds) to horizontal angle (degrees)
using the far-field plane-wave stereo model.
"""

import numpy as np

from .config import PipelineConfig


def tdoa_to_azimuth(
    tdoa: np.ndarray,
    config: PipelineConfig = None,
) -> np.ndarray:
    """
    Convert TDOA values to azimuth angles.

    Uses the far-field plane-wave assumption:
        tau = (d / c) * sin(theta)
        theta = arcsin((tau * c) / d)

    Parameters
    ----------
    tdoa : np.ndarray
        (F,) TDOA in seconds per frame.
    config : PipelineConfig
        Pipeline configuration.

    Returns
    -------
    azimuth : np.ndarray
        (F,) azimuth in degrees. 0° = center, -90° = left, +90° = right.
    """
    if config is None:
        config = PipelineConfig()

    d = config.mic_distance
    c = config.speed_of_sound

    # ── Compute sin(theta) = (tau * c) / d ──
    sin_theta = (tdoa * c) / d

    # ── Clip to [-1, 1] to handle noise-induced out-of-range values ──
    sin_theta = np.clip(sin_theta, -1.0, 1.0)

    # ── Convert to degrees ──
    azimuth = np.degrees(np.arcsin(sin_theta))

    return azimuth
