"""
Module 6b: Interaural Level Difference (ILD) Estimation
Computes per-frame energy difference between left and right channels.
Complements GCC-PHAT for amplitude-panned stereo recordings where
time delays may be negligible but level differences encode direction.
"""

import numpy as np
from scipy.signal import get_window

from .config import PipelineConfig


def estimate_ild(
    left: np.ndarray,
    right: np.ndarray,
    sr: int,
    config: PipelineConfig = None,
) -> dict:
    """
    Compute per-frame Interaural Level Difference between stereo channels.

    ILD (dB) = 10 * log10(E_right / E_left)
        Positive ILD → source is on the RIGHT
        Negative ILD → source is on the LEFT
        ILD near 0   → source is at CENTER

    Parameters
    ----------
    left : np.ndarray
        (N,) preprocessed left channel.
    right : np.ndarray
        (N,) preprocessed right channel.
    sr : int
        Sample rate in Hz.
    config : PipelineConfig
        Pipeline configuration.

    Returns
    -------
    dict with keys:
        "ild_db"      : np.ndarray (F,) ILD in decibels per frame
        "ild_azimuth" : np.ndarray (F,) estimated azimuth from ILD (-90 to +90)
        "frame_times" : np.ndarray (F,) center time of each frame
    """
    if config is None:
        config = PipelineConfig()

    # ── Input contract assertions ──
    assert left.ndim == 1, f"[ILD] Left must be 1-D, got {left.shape}"
    assert right.ndim == 1, f"[ILD] Right must be 1-D, got {right.shape}"
    assert left.dtype == np.float32, f"[ILD] Left dtype: {left.dtype}"
    assert right.dtype == np.float32, f"[ILD] Right dtype: {right.dtype}"
    assert len(left) == len(right), (
        f"[ILD] Channel length mismatch: L={len(left)}, R={len(right)}"
    )

    frame_samples = int(config.gcc_frame_len * sr)
    hop_samples = int(config.gcc_hop_len * sr)

    window = get_window("hann", frame_samples, fftbins=True).astype(np.float32)

    n_samples = min(len(left), len(right))
    n_frames = max(0, (n_samples - frame_samples) // hop_samples + 1)

    ild_db = np.zeros(n_frames, dtype=np.float64)
    frame_times = np.zeros(n_frames, dtype=np.float64)

    for i in range(n_frames):
        start = i * hop_samples
        end = start + frame_samples

        left_frame = left[start:end] * window
        right_frame = right[start:end] * window

        # Frame energy (RMS squared)
        energy_left = np.mean(left_frame ** 2)
        energy_right = np.mean(right_frame ** 2)

        # ILD in dB (positive = right, negative = left)
        eps = 1e-12
        ild_db[i] = 10.0 * np.log10((energy_right + eps) / (energy_left + eps))

        frame_times[i] = (start + frame_samples / 2) / sr

    # ── Convert ILD to approximate azimuth ──
    # For amplitude-panned stereo, the ILD maps roughly linearly to azimuth.
    # Max ILD (~12-15 dB) corresponds to ±90°.
    # We use a sigmoid-like mapping: azimuth = 90 * tanh(ILD / ild_scale)
    ild_scale = 6.0  # dB at which azimuth reaches ~76% of 90°
    ild_azimuth = 90.0 * np.tanh(ild_db / ild_scale)

    print(f"[ILD] Processed {n_frames} frames, "
          f"ILD range: [{np.min(ild_db):.1f}, {np.max(ild_db):.1f}] dB")

    return {
        "ild_db": ild_db,
        "ild_azimuth": ild_azimuth,
        "frame_times": frame_times,
    }
