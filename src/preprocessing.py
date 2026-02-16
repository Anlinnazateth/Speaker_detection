"""
Module 2: Stereo Preprocessing
DC offset removal, high-pass filtering, RMS normalization, and mono downmix.
"""

import numpy as np
from scipy.signal import butter, sosfilt
import noisereduce as nr

from .config import PipelineConfig


def preprocess_stereo(audio: np.ndarray, sr: int, config: PipelineConfig = None) -> dict:
    """
    Condition the raw stereo signal for downstream analysis.

    Parameters
    ----------
    audio : np.ndarray
        Shape (N, 2), float32 stereo audio.
    sr : int
        Sample rate in Hz.
    config : PipelineConfig
        Pipeline configuration.

    Returns
    -------
    dict with keys:
        "left"  : np.ndarray (N,) — preprocessed left channel
        "right" : np.ndarray (N,) — preprocessed right channel
        "mono"  : np.ndarray (N,) — mono downmix
        "sr"    : int
    """
    if config is None:
        config = PipelineConfig()

    left = audio[:, 0].copy()
    right = audio[:, 1].copy()

    # ── Step 1: DC offset removal (subtract per-channel mean) ──
    left -= np.mean(left)
    right -= np.mean(right)

    # ── Step 1b: Spectral-gating denoising (per-channel, preserves stereo) ──
    if config.denoise_enabled:
        left = nr.reduce_noise(
            y=left, sr=sr,
            stationary=config.denoise_stationary,
            prop_decrease=config.denoise_prop_decrease,
            n_fft=2048, hop_length=512,
        ).astype(np.float32)
        right = nr.reduce_noise(
            y=right, sr=sr,
            stationary=config.denoise_stationary,
            prop_decrease=config.denoise_prop_decrease,
            n_fft=2048, hop_length=512,
        ).astype(np.float32)
        print("[Preprocessing] Spectral-gating denoising applied (per-channel)")

    # ── Step 2: High-pass Butterworth filter at 80 Hz ──
    # Uses second-order sections (SOS) for numerical stability
    sos = butter(
        config.highpass_order,
        config.highpass_cutoff,
        btype="high",
        fs=sr,
        output="sos",
    )
    left = sosfilt(sos, left).astype(np.float32)
    right = sosfilt(sos, right).astype(np.float32)

    # ── Step 3: RMS normalization to target dBFS ──
    target_rms = 10.0 ** (config.target_rms_dbfs / 20.0)  # Convert dBFS to linear
    left = _normalize_rms(left, target_rms)
    right = _normalize_rms(right, target_rms)

    # ── Step 4: Mono downmix (arithmetic mean of L and R) ──
    mono = (left + right) / 2.0

    return {
        "left": left,
        "right": right,
        "mono": mono,
        "sr": sr,
    }


def _normalize_rms(signal: np.ndarray, target_rms: float) -> np.ndarray:
    """
    Normalize a signal so its RMS matches target_rms.
    Skips normalization if the signal is near-silent to avoid amplifying noise.
    """
    rms = np.sqrt(np.mean(signal ** 2))
    if rms < 1e-8:
        # Near-silent channel — skip normalization
        print("[WARNING] Near-silent channel detected, skipping RMS normalization.")
        return signal
    return (signal * (target_rms / rms)).astype(np.float32)
