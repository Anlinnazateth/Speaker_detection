"""
Module 8: Speaker-Location Association
Links speaker identity labels (from clustering) with spatial trajectories
by fusing TDOA-based and ILD-based azimuth estimates.
"""

import numpy as np
from scipy.ndimage import median_filter

from .config import PipelineConfig


def associate_speakers_with_locations(
    segments: list[tuple[float, float]],
    labels: np.ndarray,
    frame_times: np.ndarray,
    azimuth: np.ndarray,
    gcc_peak: np.ndarray,
    config: PipelineConfig = None,
    ild_azimuth: np.ndarray = None,
) -> dict:
    """
    Map each speaker cluster to its spatial trajectory by aligning
    VAD segments with TDOA/azimuth frames in time.

    Uses both TDOA-based azimuth and ILD-based azimuth when available.
    When TDOA is near zero but ILD shows clear direction, ILD dominates.

    Parameters
    ----------
    segments : list of (start_sec, end_sec)
        VAD segment times (one per embedding/label).
    labels : np.ndarray
        (S,) cluster labels from speaker clustering.
    frame_times : np.ndarray
        (F,) center time of each GCC-PHAT frame.
    azimuth : np.ndarray
        (F,) TDOA-based azimuth in degrees per frame.
    gcc_peak : np.ndarray
        (F,) GCC-PHAT peak confidence per frame.
    config : PipelineConfig
        Pipeline configuration.
    ild_azimuth : np.ndarray or None
        (F,) ILD-based azimuth in degrees per frame.

    Returns
    -------
    dict with keys:
        "speaker_tracks"   : dict mapping speaker_id -> list of (time, azimuth)
        "speaker_dominant" : dict mapping speaker_id -> dominant azimuth (degrees)
        "speaker_position" : dict mapping speaker_id -> "left" / "center" / "right"
    """
    if config is None:
        config = PipelineConfig()

    # ── Fuse TDOA and ILD azimuth estimates ──
    fused_azimuth = _fuse_azimuth(azimuth, gcc_peak, ild_azimuth)

    # ── Smooth fused azimuth with median filter ──
    fused_azimuth = _median_smooth(fused_azimuth, config.azimuth_median_window)

    # ── Reject outlier azimuth values ──
    fused_azimuth, gcc_peak = _reject_azimuth_outliers(
        fused_azimuth, gcc_peak, config.azimuth_outlier_sigma
    )

    num_speakers = len(np.unique(labels))
    speaker_tracks = {k: [] for k in range(num_speakers)}

    # ── For each segment, find overlapping frames ──
    for seg_idx, (seg_start, seg_end) in enumerate(segments):
        speaker_id = labels[seg_idx]

        frame_mask = (frame_times >= seg_start) & (frame_times <= seg_end)

        # Accept frames where either GCC confidence is high or ILD is available
        confidence_mask = gcc_peak >= config.gcc_confidence_min
        has_ild = ild_azimuth is not None
        valid_mask = frame_mask & (confidence_mask | has_ild)

        valid_times = frame_times[valid_mask]
        valid_azimuths = fused_azimuth[valid_mask]

        for t, az in zip(valid_times, valid_azimuths):
            speaker_tracks[speaker_id].append((float(t), float(az)))

    # ── Compute dominant position per speaker ──
    speaker_dominant = {}
    speaker_position = {}

    for speaker_id in range(num_speakers):
        observations = speaker_tracks[speaker_id]

        if len(observations) < config.min_segments_for_localization:
            speaker_dominant[speaker_id] = 0.0
            speaker_position[speaker_id] = "unknown"
            continue

        azimuths = np.array([az for _, az in observations])
        dominant = float(np.median(azimuths))
        speaker_dominant[speaker_id] = dominant

        if dominant < config.azimuth_left_threshold:
            speaker_position[speaker_id] = "left"
        elif dominant > config.azimuth_right_threshold:
            speaker_position[speaker_id] = "right"
        else:
            speaker_position[speaker_id] = "center"

    print(f"[Association] Mapped {num_speakers} speakers to spatial positions: "
          f"{speaker_position}")

    return {
        "speaker_tracks": speaker_tracks,
        "speaker_dominant": speaker_dominant,
        "speaker_position": speaker_position,
    }


def _fuse_azimuth(
    tdoa_azimuth: np.ndarray,
    gcc_peak: np.ndarray,
    ild_azimuth: np.ndarray | None,
) -> np.ndarray:
    """
    Fuse TDOA and ILD azimuth estimates with confidence-based weighting.

    When GCC-PHAT confidence is high and shows clear direction -> use TDOA.
    When TDOA is near zero but ILD shows direction -> use ILD.
    Otherwise blend proportionally.
    """
    if ild_azimuth is None:
        return tdoa_azimuth.copy()

    fused = np.zeros_like(tdoa_azimuth)

    for i in range(len(tdoa_azimuth)):
        tdoa_val = abs(tdoa_azimuth[i])
        confidence = gcc_peak[i]

        # TDOA weight scales with both its confidence and magnitude
        tdoa_weight = confidence * (tdoa_val / max(tdoa_val + 1.0, 1.0))
        ild_weight = 1.0  # ILD is always informative for panned sources

        total = tdoa_weight + ild_weight
        if total > 0:
            fused[i] = (tdoa_weight * tdoa_azimuth[i] + ild_weight * ild_azimuth[i]) / total
        else:
            fused[i] = ild_azimuth[i]

    return fused


def _median_smooth(azimuth: np.ndarray, window: int) -> np.ndarray:
    """Apply a 1D median filter to smooth noisy azimuth trajectory."""
    if window < 2 or len(azimuth) < window:
        return azimuth.copy()
    return median_filter(azimuth, size=window, mode="nearest")


def _reject_azimuth_outliers(
    azimuth: np.ndarray, gcc_peak: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reject outlier azimuth values that deviate > sigma standard deviations
    from the local median. Outlier frames get their confidence zeroed out
    so downstream modules ignore them.
    """
    az = azimuth.copy()
    gp = gcc_peak.copy()

    if len(az) < 5:
        return az, gp

    # Use a rolling window (31 frames ~0.6s at 20ms hop) for local statistics
    half_win = 15
    for i in range(len(az)):
        lo = max(0, i - half_win)
        hi = min(len(az), i + half_win + 1)
        local = az[lo:hi]
        local_median = np.median(local)
        local_std = np.std(local)
        if local_std < 1.0:
            local_std = 1.0  # floor to avoid over-rejection in quiet regions
        if abs(az[i] - local_median) > sigma * local_std:
            az[i] = local_median  # replace outlier with local median
            gp[i] = 0.0          # zero confidence so Kalman filter ignores it

    return az, gp
