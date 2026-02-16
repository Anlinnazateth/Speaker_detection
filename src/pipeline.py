"""
Main Pipeline Orchestrator
Runs the full end-to-end multi-speaker localization and tracking pipeline.

Usage:
    python -m src.pipeline path/to/stereo.wav [--output results.json] [--stage 1|2|3]
"""

import argparse
import json
import sys
import numpy as np
import torchaudio

# Compatibility patch: torchaudio 2.10+ removed list_audio_backends()
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["default"]

from .config import PipelineConfig
from .audio_loader import load_audio
from .preprocessing import preprocess_stereo
from .vad import detect_speech_segments
from .embeddings import extract_embeddings
from .clustering import cluster_speakers
from .gcc_phat import estimate_tdoa
from .azimuth import tdoa_to_azimuth
from .ild import estimate_ild
from .association import associate_speakers_with_locations
from .kalman_tracker import track_speakers
from .output_formatter import format_output, save_json, print_summary
from .visualize import visualize_results


def run_pipeline(
    audio_path: str,
    config: PipelineConfig = None,
    max_stage: int = 3,
    output_path: str = None,
) -> dict:
    """
    Execute the full multi-speaker localization and tracking pipeline.

    Parameters
    ----------
    audio_path : str
        Path to the stereo .wav file.
    config : PipelineConfig
        Pipeline configuration (uses defaults if None).
    max_stage : int
        Maximum stage to run (1, 2, or 3).
            1 = VAD + embeddings + clustering (speaker count & diarization)
            2 = + GCC-PHAT + TDOA + azimuth + speaker-location association
            3 = + Kalman tracking + movement + crossing detection
    output_path : str or None
        If provided, save JSON results to this file.

    Returns
    -------
    dict — structured pipeline output.
    """
    if config is None:
        config = PipelineConfig()

    print(f"\n{'='*60}")
    print(f"  MULTI-SPEAKER LOCALIZATION PIPELINE (Stage 1-{max_stage})")
    print(f"  Input: {audio_path}")
    print(f"{'='*60}\n")

    # ════════════════════════════════════════════════════════════
    # STAGE 1: Load → Preprocess → VAD → Embeddings → Clustering
    # ════════════════════════════════════════════════════════════

    print("[Stage 1] Audio loading and preprocessing...")
    audio_data = load_audio(audio_path, config)
    preprocessed = preprocess_stereo(audio_data["audio"], audio_data["sr"], config)

    print("[Stage 1] Voice Activity Detection...")
    segments = detect_speech_segments(
        preprocessed["mono"], preprocessed["sr"], config
    )

    if len(segments) == 0:
        print("[Stage 1] No speech detected in audio. Exiting.")
        result = format_output(
            num_speakers=0,
            segments=[],
            labels=None,
            speaker_dominant=None,
            speaker_position=None,
            tracking_results=None,
            duration=audio_data["duration"],
            sr=audio_data["sr"],
        )
        if output_path:
            save_json(result, output_path)
        return result

    print("[Stage 1] Extracting speaker embeddings...")
    embeddings, segment_times = extract_embeddings(
        preprocessed["mono"], preprocessed["sr"], segments, config
    )

    print("[Stage 1] Clustering speakers...")
    cluster_result = cluster_speakers(embeddings, config)
    labels = cluster_result["labels"]
    num_speakers = cluster_result["num_speakers"]

    # For stage 1, use the segment_times that correspond to embeddings
    # (which may differ from original VAD segments if long segments were split)
    working_segments = segment_times

    # Stage 1 outputs
    speaker_dominant = None
    speaker_position = None
    tracking_results = None

    if max_stage >= 2:
        # ════════════════════════════════════════════════════════
        # STAGE 2: GCC-PHAT → TDOA → Azimuth → Association
        # ════════════════════════════════════════════════════════

        print("\n[Stage 2] Computing GCC-PHAT TDOA...")
        tdoa_result = estimate_tdoa(
            preprocessed["left"], preprocessed["right"],
            preprocessed["sr"], config,
            speech_segments=segments,
        )

        print("[Stage 2] Converting TDOA to azimuth...")
        azimuth_array = tdoa_to_azimuth(tdoa_result["tdoa"], config)

        print("[Stage 2] Computing ILD (Interaural Level Difference)...")
        ild_result = estimate_ild(
            preprocessed["left"], preprocessed["right"],
            preprocessed["sr"], config
        )

        print("[Stage 2] Associating speakers with locations...")
        assoc_result = associate_speakers_with_locations(
            working_segments, labels,
            tdoa_result["frame_times"], azimuth_array,
            tdoa_result["gcc_peak"], config,
            ild_azimuth=ild_result["ild_azimuth"],
        )
        speaker_dominant = assoc_result["speaker_dominant"]
        speaker_position = assoc_result["speaker_position"]

    if max_stage >= 3:
        # ════════════════════════════════════════════════════════
        # STAGE 3: Kalman Tracking → Movement → Crossings
        # ════════════════════════════════════════════════════════

        print("\n[Stage 3] Kalman filter tracking...")
        # Build a time → confidence lookup for adaptive measurement noise
        gcc_peak_by_time = {
            float(t): float(p)
            for t, p in zip(tdoa_result["frame_times"], tdoa_result["gcc_peak"])
        }

        tracking_results = track_speakers(
            assoc_result["speaker_tracks"],
            gcc_peak_by_time, config
        )

        # Log crossing events
        for spk_id, track in tracking_results.items():
            if track["crossings"]:
                print(f"[Stage 3] Speaker {spk_id}: "
                      f"{len(track['crossings'])} crossing event(s) detected")
            movement = "MOVING" if track["movement_detected"] else "STATIONARY"
            print(f"[Stage 3] Speaker {spk_id}: {movement} "
                  f"(range: {track['azimuth_range']}°)")

    # ════════════════════════════════════════════════════════════
    # FORMAT OUTPUT
    # ════════════════════════════════════════════════════════════

    print("\n[Output] Formatting results...")
    result = format_output(
        num_speakers=num_speakers,
        segments=working_segments,
        labels=labels,
        speaker_dominant=speaker_dominant,
        speaker_position=speaker_position,
        tracking_results=tracking_results,
        duration=audio_data["duration"],
        sr=audio_data["sr"],
    )

    print_summary(result)

    if output_path:
        save_json(result, output_path)

    # ════════════════════════════════════════════════════════════
    # VISUALIZATION
    # ════════════════════════════════════════════════════════════

    print("\n[Visualize] Generating plots...")
    output_base = output_path.rsplit(".", 1)[0] if output_path else "output"
    visualize_results(
        results=result,
        audio=audio_data["audio"],
        sr=audio_data["sr"],
        output_prefix=output_base,
    )

    return result


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Speaker Localization & Tracking Pipeline"
    )
    parser.add_argument(
        "audio", type=str,
        help="Path to stereo .wav file"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="results.json",
        help="Output JSON file path (default: results.json)"
    )
    parser.add_argument(
        "--stage", "-s", type=int, default=3, choices=[1, 2, 3],
        help="Maximum pipeline stage to run (1=VAD+clustering, 2=+localization, 3=+tracking)"
    )
    parser.add_argument(
        "--mic-distance", type=float, default=None,
        help="Microphone spacing in meters (default: 0.20)"
    )
    parser.add_argument(
        "--vad-threshold", type=float, default=None,
        help="VAD speech probability threshold (default: 0.5)"
    )

    args = parser.parse_args()

    config = PipelineConfig()
    if args.mic_distance is not None:
        config.mic_distance = args.mic_distance
    if args.vad_threshold is not None:
        config.vad_threshold = args.vad_threshold

    result = run_pipeline(
        audio_path=args.audio,
        config=config,
        max_stage=args.stage,
        output_path=args.output,
    )

    # Also print JSON to stdout
    print("\n" + json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
