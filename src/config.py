"""
Configuration parameters for the multi-speaker localization and tracking pipeline.
All tunable constants are centralized here for easy adjustment.
"""

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """All tunable parameters for the processing pipeline."""

    # --- Audio Loader ---
    min_sample_rate: int = 16000

    # --- Preprocessing / Denoising ---
    denoise_enabled: bool = True            # Enable denoising
    denoise_method: str = "wiener"          # "wiener" | "spectral" | "none"
    # Wiener filter parameters
    wiener_n_fft: int = 2048               # STFT window size for Wiener filter
    wiener_hop: int = 512                  # STFT hop length for Wiener filter
    wiener_gain_floor: float = 0.08        # Minimum spectral gain (prevents musical noise)
    wiener_noise_percentile: float = 15.0  # Per-frequency-bin percentile for noise floor
    wiener_noise_overestimate: float = 1.5 # Over-estimate noise power for aggressive removal
    # Spectral subtraction parameters
    spectral_n_fft: int = 2048             # STFT window size for spectral subtraction
    spectral_hop: int = 512                # STFT hop length
    spectral_alpha: float = 2.0            # Over-subtraction factor (1.0–4.0)
    spectral_beta: float = 0.02            # Spectral floor factor (prevents negative power)
    spectral_noise_percentile: float = 15.0  # Per-frequency-bin percentile for noise floor
    # General preprocessing
    highpass_cutoff: float = 80.0           # Hz, removes rumble/mechanical noise
    highpass_order: int = 2                 # Butterworth filter order
    target_rms_dbfs: float = -20.0          # RMS normalization target in dBFS
    correlation_warn_threshold: float = 0.95  # Warn if inter-channel correlation exceeds this

    # --- Voice Activity Detection ---
    vad_threshold: float = 0.35             # Speech probability threshold [0, 1] (lowered for noisy audio)
    vad_min_duration: float = 0.30          # Minimum speech segment duration (seconds, 300ms)
    vad_merge_gap: float = 0.40             # Maximum gap to merge adjacent segments (seconds)

    # --- Speaker Embeddings ---
    embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    embedding_dim: int = 192               # ECAPA-TDNN output dimension
    embedding_segment_max: float = 10.0    # Max segment length before splitting (seconds)
    embedding_sub_len: float = 3.0         # Sub-segment length for long segments (seconds)
    embedding_sub_overlap: float = 1.5     # Sub-segment overlap (seconds)
    embedding_min_duration: float = 0.5    # Minimum segment duration for reliable embedding

    # --- Clustering ---
    cluster_max_k: int = 10                # Maximum number of speakers to consider
    cluster_cosine_threshold: float = 0.4  # AHC cosine distance threshold (raised for noisy)
    cluster_min_members: int = 2           # Minimum segments per valid cluster
    cluster_min_duration: float = 1.0      # Minimum total speech duration (sec) per cluster
    cluster_eigengap_min: float = 0.02     # Minimum eigen-gap to split (lowered for robustness)
    cluster_force_split_dist: float = 0.25 # Force k=2 if max pairwise cosine dist exceeds this

    # --- GCC-PHAT ---
    gcc_frame_len: float = 0.04            # Frame length for TDOA (seconds, 40ms)
    gcc_hop_len: float = 0.02              # Hop length for TDOA (seconds, 20ms)
    gcc_confidence_min: float = 0.15       # Minimum GCC peak for valid TDOA
    gcc_speech_only: bool = True           # Only compute GCC-PHAT on speech-active frames
    gcc_bandpass_lo: float = 300.0         # Band-pass lower edge Hz (speech fundamental)
    gcc_bandpass_hi: float = 3400.0        # Band-pass upper edge Hz (speech formants)
    gcc_bandpass_order: int = 4            # Butterworth band-pass filter order
    gcc_energy_floor: float = 1e-4         # RMS energy floor — skip frame if below
    gcc_peak_to_mean_min: float = 3.0      # Minimum peak-to-mean ratio for valid GCC
    gcc_tdoa_median_window: int = 5        # Median filter window applied to raw TDOA
    gcc_tdoa_jump_max: float = 0.5         # Max frame-to-frame TDOA jump (in tau_max units)
    azimuth_median_window: int = 5         # Median filter window for azimuth smoothing
    azimuth_outlier_sigma: float = 2.5     # Reject azimuths > N sigma from local median

    # --- Azimuth ---
    mic_distance: float = 0.20             # Assumed microphone spacing (meters)
    speed_of_sound: float = 343.0          # Speed of sound (m/s)

    # --- Kalman Filter ---
    kalman_q: float = 1.0                  # Process noise intensity (deg^2/s^3)
    kalman_r_base: float = 25.0            # Base measurement noise (deg^2)
    movement_threshold: float = 10.0       # Sustained azimuth shift to flag movement (degrees)
    movement_min_duration: float = 1.0     # Minimum sustained shift duration (seconds)
    movement_median_window: int = 11       # Median filter window for movement trajectory smoothing
    movement_noise_scale: float = 2.5      # Scale movement threshold by N × observed azimuth std
    kalman_reinit_gap: float = 5.0         # Re-initialize filter if gap exceeds this (seconds)
    kalman_gate_sigma: float = 3.0         # Innovation gating threshold (sigma)

    # --- Speaker-Location Association ---
    azimuth_left_threshold: float = -5.0   # Degrees below which speaker is "left"
    azimuth_right_threshold: float = 5.0   # Degrees above which speaker is "right"
    min_segments_for_localization: int = 3  # Minimum segments for valid localization
