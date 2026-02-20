"""
Module 9: Kalman Filter Tracking
Smooths noisy per-frame azimuth observations into clean trajectories,
detects speaker movement vs. stationarity, and identifies spatial crossing events.

State model per speaker:
    x = [theta, theta_dot]^T   (azimuth in degrees, azimuth velocity in deg/s)
    Constant-velocity prediction with adaptive measurement noise.
"""

import numpy as np
from scipy.ndimage import median_filter

from .config import PipelineConfig


class KalmanFilter1D:
    """
    1D Kalman filter for azimuth tracking.

    State: [theta, theta_dot] — position and velocity in degrees.
    Observation: theta (direct azimuth measurement).
    """

    def __init__(self, initial_theta: float, config: PipelineConfig):
        # ── State vector: [azimuth, azimuth_rate] ──
        self.x = np.array([initial_theta, 0.0])

        # ── State covariance ──
        self.P = np.array([
            [10.0, 0.0],
            [0.0, 1.0],
        ])

        self.config = config

    def predict(self, dt: float) -> None:
        """
        Prediction step using constant-velocity model.

        F = [[1, dt],
             [0,  1]]

        Q = q * [[dt^3/3, dt^2/2],
                 [dt^2/2, dt    ]]
        """
        F = np.array([
            [1.0, dt],
            [0.0, 1.0],
        ])

        q = self.config.kalman_q
        Q = q * np.array([
            [dt**3 / 3.0, dt**2 / 2.0],
            [dt**2 / 2.0, dt],
        ])

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z: float, gcc_confidence: float = 1.0) -> float:
        """
        Update step with a direct azimuth observation.

        Measurement noise R is adaptive:
            R = R_base / max(gcc_confidence, 0.01)
        High GCC confidence → low R → trust the observation more.

        Returns the innovation (residual) for gating checks.
        """
        H = np.array([[1.0, 0.0]])

        # Adaptive measurement noise based on GCC-PHAT confidence
        R = self.config.kalman_r_base / max(gcc_confidence, 0.01)

        # Innovation (residual)
        innovation = z - H @ self.x

        # Innovation covariance
        S = H @ self.P @ H.T + R

        # Kalman gain
        K = (self.P @ H.T) / S

        # State update
        self.x = self.x + K.flatten() * innovation

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(2) - K @ H
        self.P = I_KH @ self.P

        return float(innovation)

    @property
    def theta(self) -> float:
        return float(self.x[0])

    @property
    def theta_dot(self) -> float:
        return float(self.x[1])


def track_speakers(
    speaker_tracks: dict[int, list[tuple[float, float]]],
    gcc_peak_by_time: dict[float, float] | None = None,
    config: PipelineConfig = None,
) -> dict:
    """
    Apply Kalman filtering to each speaker's azimuth observations.
    Detects movement and spatial crossing events.

    Parameters
    ----------
    speaker_tracks : dict
        {speaker_id: [(time, azimuth), ...]} from association module.
    gcc_peak_by_time : dict or None
        {time: gcc_confidence} lookup. If None, uniform confidence is used.
    config : PipelineConfig
        Pipeline configuration.

    Returns
    -------
    dict mapping speaker_id → {
        "trajectory"         : list of {"time": t, "azimuth": theta},
        "movement_detected"  : bool,
        "azimuth_range"      : float (degrees),
        "smoothed_positions" : list of floats,
        "crossings"          : list of {"time": t, "with_speaker": id}
    }
    """
    if config is None:
        config = PipelineConfig()

    if gcc_peak_by_time is None:
        gcc_peak_by_time = {}

    results = {}

    for speaker_id, observations in speaker_tracks.items():
        if len(observations) == 0:
            results[speaker_id] = _empty_track_result()
            continue

        # ── Sort observations by time ──
        observations = sorted(observations, key=lambda x: x[0])

        # ── Initialize Kalman filter at first observation ──
        t0, theta0 = observations[0]
        kf = KalmanFilter1D(theta0, config)

        trajectory = [{"time": t0, "azimuth": theta0}]
        smoothed_positions = [theta0]
        prev_time = t0

        for t, z in observations[1:]:
            dt = t - prev_time

            # Re-initialize if gap is too large (speaker was silent)
            if dt > config.kalman_reinit_gap:
                kf = KalmanFilter1D(z, config)
                trajectory.append({"time": t, "azimuth": z})
                smoothed_positions.append(z)
                prev_time = t
                continue

            # ── Predict to current time ──
            kf.predict(dt)

            # ── Innovation gating: reject extreme outliers ──
            confidence = gcc_peak_by_time.get(t, 0.5)
            predicted_theta = kf.theta
            innovation = z - predicted_theta
            innovation_std = np.sqrt(kf.P[0, 0] + config.kalman_r_base / max(confidence, 0.01))

            if abs(innovation) > config.kalman_gate_sigma * innovation_std:
                # Outlier — skip update, use prediction only
                trajectory.append({"time": t, "azimuth": kf.theta})
                smoothed_positions.append(kf.theta)
            else:
                # ── Update with observation ──
                kf.update(z, confidence)
                trajectory.append({"time": t, "azimuth": kf.theta})
                smoothed_positions.append(kf.theta)

            prev_time = t

        # ── Movement detection (sustained shift, not just range) ──
        times_arr = np.array([p["time"] for p in trajectory])
        smoothed_arr = np.array(smoothed_positions)
        azimuth_range = float(np.max(smoothed_arr) - np.min(smoothed_arr))
        movement_detected = _detect_sustained_movement(
            times_arr, smoothed_arr, config
        )

        results[speaker_id] = {
            "trajectory": trajectory,
            "movement_detected": movement_detected,
            "azimuth_range": round(azimuth_range, 2),
            "smoothed_positions": smoothed_positions,
            "crossings": [],  # Populated in the crossing detection pass
        }

    # ── Detect spatial crossing events between speaker pairs ──
    _detect_crossings(results, config)

    return results


def _detect_sustained_movement(
    times: np.ndarray, positions: np.ndarray, config: PipelineConfig
) -> bool:
    """
    Detect movement by checking for a sustained azimuth shift, not just
    momentary spikes. This prevents noise-induced false positives.

    The movement threshold is noise-adaptive: if the observed azimuth
    standard deviation (from noise jitter) is large, the effective
    threshold scales up to avoid false positives.

    Algorithm:
      1. Median-filter the trajectory to remove short spikes
      2. Compute noise-adaptive threshold: max(base_threshold, scale * observed_std)
      3. Half-split check: require shift between halves to exceed threshold
      4. Sustained shift check: sliding window verification
    """
    if len(positions) < 4:
        return False

    # 1. Median filter to suppress spike noise
    win = min(config.movement_median_window, len(positions))
    if win < 3:
        win = 3
    if win % 2 == 0:
        win += 1  # median_filter needs odd window
    filtered = median_filter(positions, size=win, mode="nearest")

    # 2. Noise-adaptive threshold
    # At low SNR, azimuth jitter inflates std → threshold scales up automatically
    observed_std = float(np.std(filtered))
    effective_threshold = max(
        config.movement_threshold,
        config.movement_noise_scale * observed_std,
    )

    # 3. Half-split check: average azimuth in first half vs second half
    mid = len(filtered) // 2
    first_half_mean = np.mean(filtered[:mid])
    second_half_mean = np.mean(filtered[mid:])
    half_shift = abs(second_half_mean - first_half_mean)

    if half_shift < effective_threshold:
        return False

    # 4. Sustained shift check: sliding window of movement_min_duration
    if len(times) < 2:
        return True  # passed half-check with very few points

    total_duration = times[-1] - times[0]
    if total_duration < config.movement_min_duration:
        return False  # not enough time span to verify

    # Find window size in samples
    avg_dt = total_duration / max(len(times) - 1, 1)
    window_samples = max(int(config.movement_min_duration / avg_dt), 2)

    if window_samples >= len(filtered):
        # Window covers entire trajectory — half-check is sufficient
        return True

    # Scan: check if any window shows a sustained shift from the start
    start_mean = np.mean(filtered[:window_samples])
    for i in range(window_samples, len(filtered) - window_samples + 1):
        local_mean = np.mean(filtered[i:i + window_samples])
        if abs(local_mean - start_mean) > effective_threshold:
            return True

    return False


def _detect_crossings(results: dict, config: PipelineConfig) -> None:
    """
    Detect spatial crossing events — moments when two speakers'
    smoothed azimuth trajectories cross each other.

    A crossing occurs when the sign of (theta_A - theta_B) flips
    between consecutive time steps.
    """
    speaker_ids = list(results.keys())
    if len(speaker_ids) < 2:
        return

    for i in range(len(speaker_ids)):
        for j in range(i + 1, len(speaker_ids)):
            id_a = speaker_ids[i]
            id_b = speaker_ids[j]

            traj_a = results[id_a]["trajectory"]
            traj_b = results[id_b]["trajectory"]

            if len(traj_a) < 2 or len(traj_b) < 2:
                continue

            # Build time-aligned series by interpolation on a common time grid
            times_a = np.array([p["time"] for p in traj_a])
            theta_a = np.array([p["azimuth"] for p in traj_a])
            times_b = np.array([p["time"] for p in traj_b])
            theta_b = np.array([p["azimuth"] for p in traj_b])

            # Common time range
            t_start = max(times_a[0], times_b[0])
            t_end = min(times_a[-1], times_b[-1])

            if t_end <= t_start:
                continue

            # Interpolate both trajectories onto a uniform grid (10 Hz)
            common_times = np.arange(t_start, t_end, 0.1)
            if len(common_times) < 2:
                continue

            interp_a = np.interp(common_times, times_a, theta_a)
            interp_b = np.interp(common_times, times_b, theta_b)

            # Detect sign changes in the difference
            diff = interp_a - interp_b
            sign_changes = np.where(np.diff(np.sign(diff)))[0]

            for idx in sign_changes:
                crossing_time = float(common_times[idx])
                results[id_a]["crossings"].append({
                    "time": round(crossing_time, 3),
                    "with_speaker": id_b,
                })
                results[id_b]["crossings"].append({
                    "time": round(crossing_time, 3),
                    "with_speaker": id_a,
                })


def _empty_track_result() -> dict:
    """Return an empty tracking result for a speaker with no observations."""
    return {
        "trajectory": [],
        "movement_detected": False,
        "azimuth_range": 0.0,
        "smoothed_positions": [],
        "crossings": [],
    }
