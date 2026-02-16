"""
Module 3: Voice Activity Detection (VAD)
Identifies temporal regions containing speech using Silero VAD.
"""

import numpy as np
import torch
import torchaudio

from .config import PipelineConfig


def detect_speech_segments(
    mono: np.ndarray, sr: int, config: PipelineConfig = None
) -> list[tuple[float, float]]:
    """
    Run Silero VAD on a mono signal to identify speech segments.

    Parameters
    ----------
    mono : np.ndarray
        (N,) mono audio signal, float32.
    sr : int
        Sample rate in Hz.
    config : PipelineConfig
        Pipeline configuration.

    Returns
    -------
    list of (start_sec, end_sec) tuples for each speech segment.
    """
    if config is None:
        config = PipelineConfig()

    # ── Load Silero VAD model ──
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    (get_speech_timestamps, _, _, _, _) = utils

    # ── Resample to 16 kHz if needed (Silero requirement) ──
    waveform = torch.from_numpy(mono).unsqueeze(0)  # shape (1, N)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
        working_sr = 16000
    else:
        working_sr = sr

    # ── Run VAD — returns list of dicts with 'start' and 'end' sample indices ──
    speech_timestamps = get_speech_timestamps(
        waveform.squeeze(0),
        model,
        threshold=config.vad_threshold,
        sampling_rate=working_sr,
        min_speech_duration_ms=int(config.vad_min_duration * 1000),
        min_silence_duration_ms=int(config.vad_merge_gap * 1000),
    )

    # ── Convert sample indices to seconds ──
    segments = [
        (ts["start"] / working_sr, ts["end"] / working_sr)
        for ts in speech_timestamps
    ]

    # ── Merge segments separated by less than vad_merge_gap ──
    segments = _merge_close_segments(segments, config.vad_merge_gap)

    # ── Drop segments shorter than vad_min_duration ──
    segments = [
        (s, e) for s, e in segments if (e - s) >= config.vad_min_duration
    ]

    # ── Reject noise bursts: drop segments with very low RMS energy ──
    segments = _reject_low_energy(mono, sr, segments)

    print(f"[VAD] Detected {len(segments)} speech segments "
          f"(total speech: {sum(e - s for s, e in segments):.1f}s)")

    return segments


def _merge_close_segments(
    segments: list[tuple[float, float]], max_gap: float
) -> list[tuple[float, float]]:
    """Merge adjacent segments separated by less than max_gap seconds."""
    if not segments:
        return []

    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap:
            # Merge with previous segment
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def _reject_low_energy(
    mono: np.ndarray, sr: int, segments: list[tuple[float, float]],
    energy_floor_ratio: float = 0.05,
) -> list[tuple[float, float]]:
    """
    Drop segments whose RMS energy is far below the median segment energy.
    This filters out noise bursts that VAD may have mis-detected as speech.
    """
    if len(segments) <= 1:
        return segments

    energies = []
    for s, e in segments:
        s_idx, e_idx = int(s * sr), int(e * sr)
        chunk = mono[s_idx:e_idx]
        rms = np.sqrt(np.mean(chunk ** 2)) if len(chunk) > 0 else 0.0
        energies.append(rms)

    energies = np.array(energies)
    median_energy = np.median(energies)

    if median_energy < 1e-8:
        return segments

    kept = [
        seg for seg, eng in zip(segments, energies)
        if eng >= median_energy * energy_floor_ratio
    ]
    n_dropped = len(segments) - len(kept)
    if n_dropped > 0:
        print(f"[VAD] Dropped {n_dropped} low-energy noise burst(s)")
    return kept
