"""
Module 1: Audio Loader
Loads a stereo WAV file, validates format, and extracts metadata.

Provides three entry points for consistent audio handling:
  - load_audio(file_path)       — read from disk (CLI path)
  - load_audio_from_buffer(buf) — read from BytesIO (Streamlit path)
  - validate_audio(array, sr)   — validate an in-memory array directly
All three guarantee the same output contract:
  shape (N, 2), dtype float32, peak-normalized to [-1, 1].
"""

import numpy as np
import soundfile as sf

from .config import PipelineConfig


def validate_audio(audio: np.ndarray, sr: int, config: PipelineConfig = None) -> dict:
    """
    Validate and normalize an in-memory stereo audio array.

    Enforces the pipeline's input contract:
      - Shape must be (N, 2)
      - Dtype is coerced to float32
      - Peak-normalized to [-1, 1]

    Parameters
    ----------
    audio : np.ndarray
        Raw audio array (should be stereo).
    sr : int
        Sample rate in Hz.
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

    # ── Validate stereo ──
    if audio.ndim == 1:
        raise ValueError(
            f"Expected stereo (2-channel) audio, got mono (shape {audio.shape})."
        )
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError(
            f"Expected stereo shape (N, 2), got {audio.shape}."
        )

    # ── Enforce float32 ──
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

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

    # ── Post-validation assertions ──
    assert audio.shape[1] == 2, f"Stereo invariant violated: {audio.shape}"
    assert audio.dtype == np.float32, f"Float32 invariant violated: {audio.dtype}"
    assert np.max(np.abs(audio)) <= 1.0 + 1e-6, (
        f"Range invariant violated: [{np.min(audio):.4f}, {np.max(audio):.4f}]"
    )

    n_samples = audio.shape[0]
    duration = n_samples / sr

    return {
        "audio": audio,
        "sr": sr,
        "duration": duration,
        "n_samples": n_samples,
    }


def load_audio(file_path: str, config: PipelineConfig = None) -> dict:
    """
    Load a stereo WAV file from disk and validate its format.

    Parameters
    ----------
    file_path : str
        Path to the .wav file.
    config : PipelineConfig
        Pipeline configuration (uses defaults if None).

    Returns
    -------
    dict — same contract as validate_audio().
    """
    if config is None:
        config = PipelineConfig()

    try:
        audio, sr = sf.read(file_path, dtype="float32")
    except sf.SoundFileError as e:
        raise ValueError(f"Cannot read audio file '{file_path}': {e}")

    return validate_audio(audio, sr, config)


def load_audio_from_buffer(buffer, config: PipelineConfig = None) -> dict:
    """
    Load a stereo WAV from a file-like object (e.g. io.BytesIO)
    and validate its format.

    This is the Streamlit/in-memory entry point — no temp files needed.

    Parameters
    ----------
    buffer : file-like object
        Readable buffer containing WAV data.
    config : PipelineConfig
        Pipeline configuration (uses defaults if None).

    Returns
    -------
    dict — same contract as validate_audio().
    """
    if config is None:
        config = PipelineConfig()

    try:
        audio, sr = sf.read(buffer, dtype="float32")
    except sf.SoundFileError as e:
        raise ValueError(f"Cannot read audio from buffer: {e}")

    return validate_audio(audio, sr, config)
