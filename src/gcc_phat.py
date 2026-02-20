"""
Module 6: GCC-PHAT TDOA Estimation
Estimates the Time Difference of Arrival between left and right channels
using Generalized Cross-Correlation with Phase Transform.

Noise-robustness features:
  - Band-pass filtering (300–3400 Hz) before cross-correlation
  - Energy gating: skip frames below RMS floor
  - Peak-to-mean ratio validation after correlation
  - Median filtering on raw TDOA delays
  - Outlier rejection for unrealistic frame-to-frame jumps
"""

import numpy as np
from scipy.signal import get_window, butter, sosfilt
from scipy.ndimage import median_filter

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

    # ── Input contract assertions ──
    assert left.ndim == 1, f"[GCC-PHAT] Left must be 1-D, got {left.shape}"
    assert right.ndim == 1, f"[GCC-PHAT] Right must be 1-D, got {right.shape}"
    assert left.dtype == np.float32, f"[GCC-PHAT] Left dtype: {left.dtype}"
    assert right.dtype == np.float32, f"[GCC-PHAT] Right dtype: {right.dtype}"
    assert len(left) == len(right), (
        f"[GCC-PHAT] Channel length mismatch: L={len(left)}, R={len(right)}"
    )

    # ── 1. Band-pass filter both channels (300–3400 Hz) ──
    left_bp, right_bp = _bandpass_stereo(left, right, sr, config)

    frame_samples = int(config.gcc_frame_len * sr)
    hop_samples = int(config.gcc_hop_len * sr)

    # Maximum allowable delay based on mic spacing
    tau_max_samples = int((config.mic_distance / config.speed_of_sound) * sr) + 1

    # FFT size — next power of 2 for efficiency
    nfft = 1 << (2 * frame_samples - 1).bit_length()

    # Hann window for each frame
    window = get_window("hann", frame_samples, fftbins=True).astype(np.float32)

    n_samples = min(len(left_bp), len(right_bp))
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
    n_energy_skipped = 0
    n_peak_rejected = 0

    for i in range(n_frames):
        if not speech_mask[i]:
            # Non-speech frame: zero TDOA and zero confidence
            tdoa_array[i] = 0.0
            gcc_peak_array[i] = 0.0
            continue

        start = i * hop_samples
        end = start + frame_samples

        # ── Window the frame ──
        left_frame = left_bp[start:end] * window
        right_frame = right_bp[start:end] * window

        # ── 2. Energy gating: skip low-energy frames ──
        rms_l = np.sqrt(np.mean(left_frame ** 2))
        rms_r = np.sqrt(np.mean(right_frame ** 2))
        if max(rms_l, rms_r) < config.gcc_energy_floor:
            tdoa_array[i] = 0.0
            gcc_peak_array[i] = 0.0
            n_energy_skipped += 1
            continue

        # ── GCC-PHAT computation ──
        tdoa_sec, peak_val, peak_to_mean = _gcc_phat_single_frame(
            left_frame, right_frame, nfft, sr, tau_max_samples
        )

        # ── 3. Peak-to-mean ratio validation ──
        if peak_to_mean < config.gcc_peak_to_mean_min:
            tdoa_array[i] = 0.0
            gcc_peak_array[i] = 0.0
            n_peak_rejected += 1
            continue

        tdoa_array[i] = tdoa_sec
        gcc_peak_array[i] = peak_val
        n_active += 1

    # ── 4. Median filter on raw TDOA delays ──
    tdoa_array, gcc_peak_array = _median_smooth_tdoa(
        tdoa_array, gcc_peak_array, config.gcc_tdoa_median_window
    )

    # ── 5. Outlier rejection: remove extreme delay jumps ──
    tau_max_sec = tau_max_samples / sr
    tdoa_array, gcc_peak_array = _reject_tdoa_outliers(
        tdoa_array, gcc_peak_array, tau_max_sec, config.gcc_tdoa_jump_max
    )

    valid_count = int(np.sum(gcc_peak_array >= config.gcc_confidence_min))
    skipped = n_frames - int(np.sum(speech_mask))
    print(f"[GCC-PHAT] Processed {n_active}/{n_frames} speech-active frames "
          f"(skipped {skipped} non-speech, {n_energy_skipped} low-energy, "
          f"{n_peak_rejected} weak-peak), valid TDOA in {valid_count} frames")

    return {
        "tdoa": tdoa_array,
        "gcc_peak": gcc_peak_array,
        "frame_times": frame_times,
    }


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _bandpass_stereo(
    left: np.ndarray, right: np.ndarray, sr: int, config: PipelineConfig
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Butterworth band-pass filter (300–3400 Hz) to both channels.
    Focuses GCC-PHAT on the speech frequency range, rejecting low-frequency
    rumble and high-frequency noise that corrupt cross-correlation.
    """
    nyq = sr / 2.0
    lo = config.gcc_bandpass_lo / nyq
    hi = config.gcc_bandpass_hi / nyq

    # Clamp to valid range for the Nyquist
    lo = max(lo, 0.001)
    hi = min(hi, 0.999)

    if lo >= hi:
        # Invalid range for this sample rate — skip filtering
        return left.copy(), right.copy()

    sos = butter(config.gcc_bandpass_order, [lo, hi], btype="band", output="sos")
    left_bp = sosfilt(sos, left).astype(np.float32)
    right_bp = sosfilt(sos, right).astype(np.float32)

    return left_bp, right_bp


def _gcc_phat_single_frame(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    nfft: int,
    sr: int,
    tau_max_samples: int,
) -> tuple[float, float, float]:
    """
    Compute GCC-PHAT for a single frame pair.

    Returns (tdoa_seconds, peak_value, peak_to_mean_ratio).
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
    gcc_abs = np.abs(gcc)
    peak_idx = np.argmax(gcc_abs)
    peak_value = gcc_abs[peak_idx]
    peak_lag = lag_indices[peak_idx]

    # ── Step 6b: Peak-to-mean ratio (confidence metric) ──
    mean_value = np.mean(gcc_abs)
    peak_to_mean = peak_value / max(mean_value, 1e-12)

    # ── Step 7: Sub-sample parabolic interpolation for precision ──
    refined_lag = _parabolic_interpolation(gcc, peak_idx)

    # Convert lag from samples to seconds
    tdoa_seconds = refined_lag / sr

    return tdoa_seconds, peak_value, peak_to_mean


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


def _median_smooth_tdoa(
    tdoa: np.ndarray, gcc_peak: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply median filter to raw TDOA values (only on frames with non-zero
    confidence). This smooths out sporadic noisy delay estimates while
    preserving real transitions between speakers.
    """
    if window < 2 or len(tdoa) < window:
        return tdoa, gcc_peak

    # Only smooth frames that have valid confidence
    valid = gcc_peak > 0
    if np.sum(valid) < window:
        return tdoa, gcc_peak

    smoothed = tdoa.copy()
    valid_tdoa = tdoa[valid]
    valid_tdoa_smoothed = median_filter(valid_tdoa, size=window, mode="nearest")

    # Write back smoothed values only to valid positions
    smoothed[valid] = valid_tdoa_smoothed

    return smoothed, gcc_peak


def _reject_tdoa_outliers(
    tdoa: np.ndarray, gcc_peak: np.ndarray,
    tau_max_sec: float, jump_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reject TDOA frames with unrealistic values or jumps:
      - Absolute TDOA exceeding tau_max (physically impossible)
      - Frame-to-frame TDOA jump exceeding jump_threshold * tau_max
    Rejected frames get their confidence zeroed.
    """
    tdoa_out = tdoa.copy()
    gp_out = gcc_peak.copy()

    max_jump = jump_threshold * tau_max_sec
    prev_valid_tdoa = None

    for i in range(len(tdoa_out)):
        if gp_out[i] <= 0:
            continue

        # Reject physically impossible delays
        if abs(tdoa_out[i]) > tau_max_sec * 1.2:
            tdoa_out[i] = 0.0
            gp_out[i] = 0.0
            continue

        # Reject large frame-to-frame jumps
        if prev_valid_tdoa is not None:
            if abs(tdoa_out[i] - prev_valid_tdoa) > max_jump:
                tdoa_out[i] = 0.0
                gp_out[i] = 0.0
                continue

        prev_valid_tdoa = tdoa_out[i]

    return tdoa_out, gp_out
