"""
Module 2: Stereo Preprocessing
DC offset removal, modular stereo-safe denoising, high-pass filtering,
RMS normalization, mono downmix, and diagnostics.

Denoising methods (all preserve inter-channel phase):
  - "wiener"   : Joint stereo Wiener filter (per-bin percentile noise floor)
  - "spectral" : Spectral subtraction with over-subtraction and spectral floor
  - "none"     : Skip denoising entirely
"""

import numpy as np
from scipy.signal import butter, sosfilt, stft, istft

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
        "left"        : np.ndarray (N,) — preprocessed left channel
        "right"       : np.ndarray (N,) — preprocessed right channel
        "mono"        : np.ndarray (N,) — mono downmix
        "sr"          : int
        "diagnostics" : dict — preprocessing diagnostics
    """
    if config is None:
        config = PipelineConfig()

    # ── Input contract assertions ──
    assert audio.ndim == 2 and audio.shape[1] == 2, (
        f"[Preprocessing] Expected stereo (N, 2), got {audio.shape}"
    )
    assert audio.dtype == np.float32, (
        f"[Preprocessing] Expected float32, got {audio.dtype}"
    )
    assert np.max(np.abs(audio)) <= 1.0 + 1e-6, (
        f"[Preprocessing] Expected peak within [-1, 1], "
        f"got [{np.min(audio):.4f}, {np.max(audio):.4f}]"
    )

    left = audio[:, 0].copy()
    right = audio[:, 1].copy()

    # ── Step 1: DC offset removal (subtract per-channel mean) ──
    left -= np.mean(left)
    right -= np.mean(right)

    # ── Step 2: Stereo-safe denoising (modular dispatch) ──
    if config.denoise_enabled:
        left, right = _denoise_stereo(left, right, sr, config)

    # ── Step 3: High-pass Butterworth filter at 80 Hz ──
    sos = butter(
        config.highpass_order,
        config.highpass_cutoff,
        btype="high",
        fs=sr,
        output="sos",
    )
    left = sosfilt(sos, left).astype(np.float32)
    right = sosfilt(sos, right).astype(np.float32)

    # ── Step 4: RMS normalization to target dBFS (NaN-safe) ──
    target_rms = 10.0 ** (config.target_rms_dbfs / 20.0)
    left = _normalize_rms(left, target_rms)
    right = _normalize_rms(right, target_rms)

    # ── Step 5: Mono downmix (arithmetic mean of L and R) ──
    mono = ((left + right) * np.float32(0.5)).astype(np.float32)

    # ── Step 6: Diagnostics ──
    diagnostics = _compute_diagnostics(left, right, mono, config)

    # ── Output contract: all channels are 1-D float32 ──
    assert left.dtype == np.float32, f"Left channel dtype: {left.dtype}"
    assert right.dtype == np.float32, f"Right channel dtype: {right.dtype}"
    assert mono.dtype == np.float32, f"Mono channel dtype: {mono.dtype}"

    return {
        "left": left,
        "right": right,
        "mono": mono,
        "sr": sr,
        "diagnostics": diagnostics,
    }


# ─────────────────────────────────────────────────────────────
# Denoising dispatcher
# ─────────────────────────────────────────────────────────────

def _denoise_stereo(
    left: np.ndarray, right: np.ndarray, sr: int, config: PipelineConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the selected denoising method."""
    method = config.denoise_method.lower()

    if method == "wiener":
        left, right = _wiener_denoise_stereo(left, right, sr, config)
        print("[Preprocessing] Wiener denoising applied (shared spectral mask)")
    elif method == "spectral":
        left, right = _spectral_subtract_stereo(left, right, sr, config)
        print("[Preprocessing] Spectral subtraction applied (shared spectral mask)")
    elif method == "none":
        print("[Preprocessing] Denoising skipped (method=none)")
    else:
        raise ValueError(
            f"[Preprocessing] Unknown denoise_method '{config.denoise_method}'. "
            f"Choose from: 'wiener', 'spectral', 'none'"
        )

    return left, right


# ─────────────────────────────────────────────────────────────
# Noise estimation (shared by both methods)
# ─────────────────────────────────────────────────────────────

def _estimate_noise_floor(
    power_l: np.ndarray,
    power_r: np.ndarray,
    percentile: float,
    overestimate: float,
) -> np.ndarray:
    """
    Estimate per-frequency-bin noise floor from both channels jointly.

    For each frequency bin, takes the Nth percentile of power values across
    all time frames.  This works because at any given frequency, noise is
    present in most frames but speech is present in only some.  The low
    percentile naturally captures the noise-only component at each bin.

    The joint L+R average ensures the same noise profile is used for both
    channels.

    Parameters
    ----------
    power_l, power_r : np.ndarray
        (n_freq, n_frames) power spectra of left and right channels.
    percentile : float
        Percentile to use (e.g. 15 = 15th percentile per bin).
    overestimate : float
        Multiply noise estimate by this factor for more aggressive removal.

    Returns
    -------
    noise_power : np.ndarray
        (n_freq,) estimated noise power per frequency bin.
    """
    # Per-bin percentile across all frames, joint across channels
    noise_l = np.percentile(power_l, percentile, axis=1)
    noise_r = np.percentile(power_r, percentile, axis=1)
    noise_power = (noise_l + noise_r) * 0.5

    noise_power *= overestimate

    return noise_power


# ─────────────────────────────────────────────────────────────
# Wiener filter
# ─────────────────────────────────────────────────────────────

def _wiener_denoise_stereo(
    left: np.ndarray,
    right: np.ndarray,
    sr: int,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Joint stereo Wiener filter denoising.

    Estimates a SINGLE noise power spectrum via per-frequency-bin percentile
    across all frames of both channels combined, then applies the SAME
    spectral gain mask to both L and R channels.  This preserves
    inter-channel phase coherence required for GCC-PHAT TDOA estimation.

    Wiener gain:  G(f,t) = max(S(f,t) - N(f), 0) / max(S(f,t), eps)
    Floored at gain_floor to prevent musical noise artifacts.
    """
    n_fft = config.wiener_n_fft
    hop = config.wiener_hop
    n_overlap = n_fft - hop
    original_len = len(left)

    # ── STFT of both channels ──
    _, _, Zl = stft(left, fs=sr, nperseg=n_fft, noverlap=n_overlap,
                    boundary="zeros", padded=True)
    _, _, Zr = stft(right, fs=sr, nperseg=n_fft, noverlap=n_overlap,
                    boundary="zeros", padded=True)

    # Power spectra: (n_freq, n_frames)
    power_l = np.abs(Zl) ** 2
    power_r = np.abs(Zr) ** 2

    # ── Noise floor estimation: per-frequency-bin percentile ──
    noise_power = _estimate_noise_floor(
        power_l, power_r,
        config.wiener_noise_percentile,
        config.wiener_noise_overestimate,
    )

    # ── Compute Wiener gain — vectorized ──
    signal_power = (power_l + power_r) * 0.5
    gain = np.maximum(signal_power - noise_power[:, np.newaxis], 0.0) / \
        np.maximum(signal_power, 1e-10)
    gain = np.maximum(gain, config.wiener_gain_floor)

    # Apply the SAME gain to both channels (preserves phase difference)
    Zl *= gain
    Zr *= gain

    # ── Inverse STFT ──
    _, left_out = istft(Zl, fs=sr, nperseg=n_fft, noverlap=n_overlap,
                        boundary=True)
    _, right_out = istft(Zr, fs=sr, nperseg=n_fft, noverlap=n_overlap,
                         boundary=True)

    left_out = left_out[:original_len].astype(np.float32)
    right_out = right_out[:original_len].astype(np.float32)

    return left_out, right_out


# ─────────────────────────────────────────────────────────────
# Spectral subtraction
# ─────────────────────────────────────────────────────────────

def _spectral_subtract_stereo(
    left: np.ndarray,
    right: np.ndarray,
    sr: int,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Joint stereo spectral subtraction denoising.

    Classic spectral subtraction with over-subtraction factor (alpha) and
    spectral floor (beta):

        |S_clean(f)|^2 = max(|X(f)|^2 - alpha * |N(f)|^2,  beta * |N(f)|^2)

    The gain is derived from this cleaned magnitude and applied identically
    to both L and R channels, preserving inter-channel phase.

    More aggressive than Wiener at removing stationary noise, but can
    introduce more musical noise artifacts (mitigated by spectral floor).
    """
    n_fft = config.spectral_n_fft
    hop = config.spectral_hop
    n_overlap = n_fft - hop
    original_len = len(left)
    alpha = config.spectral_alpha
    beta = config.spectral_beta

    # ── STFT of both channels ──
    _, _, Zl = stft(left, fs=sr, nperseg=n_fft, noverlap=n_overlap,
                    boundary="zeros", padded=True)
    _, _, Zr = stft(right, fs=sr, nperseg=n_fft, noverlap=n_overlap,
                    boundary="zeros", padded=True)

    power_l = np.abs(Zl) ** 2
    power_r = np.abs(Zr) ** 2

    # ── Noise floor estimation: per-frequency-bin percentile ──
    noise_power = _estimate_noise_floor(
        power_l, power_r,
        config.spectral_noise_percentile,
        1.0,  # no overestimate — alpha handles aggression
    )

    # ── Spectral subtraction gain — vectorized ──
    signal_power = (power_l + power_r) * 0.5

    # |S_clean|^2 = max(|X|^2 - alpha*|N|^2,  beta*|N|^2)
    subtracted = np.maximum(
        signal_power - alpha * noise_power[:, np.newaxis],
        beta * noise_power[:, np.newaxis],
    )

    # Gain = sqrt(|S_clean|^2 / |X|^2)
    gain = np.sqrt(subtracted / np.maximum(signal_power, 1e-10))
    gain = np.clip(gain, 0.0, 1.0)

    # Apply the SAME gain to both channels (preserves phase difference)
    Zl *= gain
    Zr *= gain

    # ── Inverse STFT ──
    _, left_out = istft(Zl, fs=sr, nperseg=n_fft, noverlap=n_overlap,
                        boundary=True)
    _, right_out = istft(Zr, fs=sr, nperseg=n_fft, noverlap=n_overlap,
                         boundary=True)

    left_out = left_out[:original_len].astype(np.float32)
    right_out = right_out[:original_len].astype(np.float32)

    return left_out, right_out


# ─────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────

def _compute_diagnostics(
    left: np.ndarray, right: np.ndarray, mono: np.ndarray,
    config: PipelineConfig,
) -> dict:
    """
    Compute and print preprocessing diagnostics.

    Checks:
      - Max amplitude per channel (should be < 1.0)
      - RMS energy per channel
      - Inter-channel Pearson correlation (> 0.95 = warning)
    """
    max_l = float(np.max(np.abs(left)))
    max_r = float(np.max(np.abs(right)))
    rms_l = float(np.sqrt(np.mean(left ** 2)))
    rms_r = float(np.sqrt(np.mean(right ** 2)))

    # Inter-channel correlation (Pearson)
    denom = max(np.std(left) * np.std(right), 1e-10)
    correlation = float(np.mean((left - np.mean(left)) * (right - np.mean(right))) / denom)

    print(f"[Diagnostics] Peak L={max_l:.4f}  R={max_r:.4f}  |  "
          f"RMS L={rms_l:.4f}  R={rms_r:.4f}  |  "
          f"Correlation={correlation:.4f}")

    if correlation > config.correlation_warn_threshold:
        print(f"[WARNING] Inter-channel correlation {correlation:.3f} > "
              f"{config.correlation_warn_threshold:.2f} — "
              f"channels are nearly identical, spatial separation may not exist")

    diag = {
        "max_amplitude_left": round(max_l, 5),
        "max_amplitude_right": round(max_r, 5),
        "rms_left": round(rms_l, 5),
        "rms_right": round(rms_r, 5),
        "inter_channel_correlation": round(correlation, 5),
    }

    return diag


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _normalize_rms(signal: np.ndarray, target_rms: float) -> np.ndarray:
    """
    Normalize a signal so its RMS matches target_rms.
    Skips normalization if the signal is near-silent to avoid amplifying noise.
    Guards against NaN propagation.
    """
    rms = np.sqrt(np.mean(signal ** 2))
    if rms < 1e-8 or not np.isfinite(rms):
        print("[WARNING] Near-silent or invalid channel, skipping RMS normalization.")
        return signal
    result = (signal * (target_rms / rms)).astype(np.float32)
    # Guard against NaN/Inf from edge cases
    if not np.all(np.isfinite(result)):
        print("[WARNING] NaN/Inf detected after normalization, using original signal.")
        return signal
    return result
