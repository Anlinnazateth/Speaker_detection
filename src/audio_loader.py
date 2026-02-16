"""
Module 1: Audio Loader
Loads a stereo WAV file, validates format, and extracts metadata.
"""

import numpy as np
import soundfile as sf

from .config import PipelineConfig


def load_audio(file_path: str, config: PipelineConfig = None) -> dict:
    """
    Load a stereo WAV file into memory and validate its format.

    Parameters
    ----------
    file_path : str
        Path to the .wav file.
    config : PipelineConfig
        Pipeline configuration (uses defaults if None).

    Returns
    -------
    dict with keys:
        "audio"     : np.ndarray, shape (N, 2), float32, range [-1, 1]
        "sr"        : int, sample rate
        "duration"  : float, duration in seconds
        "n_samples" : int, total samples per channel
    """
    if config is None:
        config = PipelineConfig()

    # ── Read raw audio ──
    try:
        audio, sr = sf.read(file_path, dtype="float32")
    except sf.SoundFileError as e:
        raise ValueError(f"Cannot read audio file '{file_path}': {e}")

    # ── Validate stereo ──
    if audio.ndim == 1:
        raise ValueError(
            f"Expected stereo (2-channel) audio, got mono. "
            f"File: {file_path}"
        )
    if audio.shape[1] != 2:
        raise ValueError(
            f"Expected stereo (2-channel) audio, got {audio.shape[1]} channels. "
            f"File: {file_path}"
        )

    # ── Validate sample rate ──
    if sr < config.min_sample_rate:
        print(
            f"[WARNING] Sample rate {sr} Hz is below recommended minimum "
            f"{config.min_sample_rate} Hz. Results may be degraded."
        )

    # ── Normalize peak to [-1, 1] ──
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    n_samples = audio.shape[0]
    duration = n_samples / sr

    return {
        "audio": audio,
        "sr": sr,
        "duration": duration,
        "n_samples": n_samples,
    }
