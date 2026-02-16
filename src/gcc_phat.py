"""
Module 6: GCC-PHAT TDOA Estimation
Estimates the Time Difference of Arrival between left and right channels
using Generalized Cross-Correlation with Phase Transform.
"""

import numpy as np
from scipy.signal import get_window

from .config import PipelineConfig


def estimate_tdoa(
    left: np.ndarray,
    right: np.ndarray,
    sr: int,
    config: PipelineConfig = None,
    speech_segments: list[tuple[float, float]] = None,
) -> dict:
    """
    Compute per-frame TDOA between stereo channels using GCC-PHAT.

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
    speech_segments : list of (start_sec, end_sec) or None
        If provided and config.gcc_speech_only is True, only compute
        GCC-PHAT on frames overlapping with speech segments.

    Returns
    -------
    dict with keys:
        "tdoa"        : np.ndarray (F,) TDOA in seconds per frame
        "gcc_peak"    : np.ndarray (F,) peak correlation value (confidence)
        "frame_times" : np.ndarray (F,) center time of each frame in seconds
    """
    if config is None:
        config = PipelineConfig()

    frame_samples = int(config.gcc_frame_len * sr)
    hop_samples = int(config.gcc_hop_len * sr)

    # Maximum allowable delay based on mic spacing
    tau_max_samples = int((config.mic_distance / config.speed_of_sound) * sr) + 1

    # FFT size — next power of 2 for efficiency
    nfft = 1 << (2 * frame_samples - 1).bit_length()

    # Hann window for each frame
    window = get_window("hann", frame_samples, fftbins=True).astype(np.float32)

    n_samples = min(len(left), len(right))
    n_frames = max(0, (n_samples - frame_samples) // hop_samples + 1)

    tdoa_array = np.zeros(n_frames, dtype=np.float64)
    gcc_peak_array = np.zeros(n_frames, dtype=np.float64)
    frame_times = np.zeros(n_frames, dtype=np.float64)

    # Pre-compute frame center times
    for i in range(n_frames):
        start = i * hop_samples
        frame_times[i] = (start + frame_samples / 2) / sr

    # Build speech-active mask if requested
    speech_mask = np.ones(n_frames, dtype=bool)
    if config.gcc_speech_only and speech_segments:
        speech_mask[:] = False
        for seg_start, seg_end in speech_segments:
            speech_mask |= (frame_times >= seg_start) & (frame_times <= seg_end)

    n_active = 0
    for i in range(n_frames):
        if not speech_mask[i]:
            # Non-speech frame: zero TDOA and zero confidence
            tdoa_array[i] = 0.0
            gcc_peak_array[i] = 0.0
            continue

        start = i * hop_samples
        end = start + frame_samples

        # ── Window the frame ──
        left_frame = left[start:end] * window
        right_frame = right[start:end] * window

        # ── GCC-PHAT computation ──
        tdoa_sec, peak_val = _gcc_phat_single_frame(
            left_frame, right_frame, nfft, sr, tau_max_samples
        )

        tdoa_array[i] = tdoa_sec
        gcc_peak_array[i] = peak_val
        n_active += 1

    valid_count = int(np.sum(gcc_peak_array >= config.gcc_confidence_min))
    skipped = n_frames - int(np.sum(speech_mask))
    print(f"[GCC-PHAT] Processed {n_active}/{n_frames} speech-active frames "
          f"(skipped {skipped} non-speech), valid TDOA in {valid_count} frames")

    return {
        "tdoa": tdoa_array,
        "gcc_peak": gcc_peak_array,
        "frame_times": frame_times,
    }


def _gcc_phat_single_frame(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    nfft: int,
    sr: int,
    tau_max_samples: int,
) -> tuple[float, float]:
    """
    Compute GCC-PHAT for a single frame pair.

    Returns (tdoa_seconds, peak_value).
    """
    # ── Step 1: FFTs of both channels ──
    L = np.fft.rfft(left_frame, n=nfft)
    R = np.fft.rfft(right_frame, n=nfft)

    # ── Step 2: Cross-power spectrum ──
    cross_spectrum = L * np.conj(R)

    # ── Step 3: PHAT weighting (spectral whitening) ──
    magnitude = np.abs(cross_spectrum)
    magnitude = np.maximum(magnitude, 1e-10)  # Avoid division by zero
    phat_weighted = cross_spectrum / magnitude

    # ── Step 4: Inverse FFT to get GCC-PHAT correlation ──
    gcc = np.fft.irfft(phat_weighted, n=nfft)

    # ── Step 5: Restrict search to valid delay range ──
    # The correlation is circular — rearrange so negative delays are accessible
    gcc = np.concatenate([gcc[-tau_max_samples:], gcc[:tau_max_samples + 1]])
    lag_indices = np.arange(-tau_max_samples, tau_max_samples + 1)

    # ── Step 6: Find the peak ──
    peak_idx = np.argmax(np.abs(gcc))
    peak_value = np.abs(gcc[peak_idx])
    peak_lag = lag_indices[peak_idx]

    # ── Step 7: Sub-sample parabolic interpolation for precision ──
    refined_lag = _parabolic_interpolation(gcc, peak_idx)

    # Convert lag from samples to seconds
    tdoa_seconds = refined_lag / sr

    return tdoa_seconds, peak_value


def _parabolic_interpolation(gcc: np.ndarray, peak_idx: int) -> float:
    """
    Refine the peak location using parabolic (quadratic) interpolation
    around the peak for sub-sample precision.

    tau_refined = peak_idx + 0.5 * (R[i-1] - R[i+1]) / (R[i-1] - 2*R[i] + R[i+1])
    """
    n = len(gcc)

    if peak_idx <= 0 or peak_idx >= n - 1:
        # Can't interpolate at boundaries — return integer index
        lag_range = n // 2
        return float(peak_idx - lag_range)

    r_prev = np.abs(gcc[peak_idx - 1])
    r_peak = np.abs(gcc[peak_idx])
    r_next = np.abs(gcc[peak_idx + 1])

    denominator = r_prev - 2.0 * r_peak + r_next
    if abs(denominator) < 1e-12:
        # Flat region — no interpolation possible
        lag_range = n // 2
        return float(peak_idx - lag_range)

    offset = 0.5 * (r_prev - r_next) / denominator
    lag_range = n // 2
    return float(peak_idx - lag_range) + offset
