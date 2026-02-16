"""
Module 10: Output Formatter
Assembles all pipeline results into a structured JSON report.
"""

import json
from datetime import datetime


def format_output(
    num_speakers: int,
    segments: list[tuple[float, float]],
    labels: list | None,
    speaker_dominant: dict | None,
    speaker_position: dict | None,
    tracking_results: dict | None,
    duration: float,
    sr: int,
) -> dict:
    """
    Assemble all pipeline results into the final JSON structure.

    Parameters
    ----------
    num_speakers : int
        Auto-detected number of speakers.
    segments : list of (start, end)
        VAD speech segments.
    labels : np.ndarray or None
        Cluster labels per segment.
    speaker_dominant : dict or None
        {speaker_id: dominant_azimuth}.
    speaker_position : dict or None
        {speaker_id: "left"/"center"/"right"}.
    tracking_results : dict or None
        Kalman tracking results per speaker.
    duration : float
        Total audio duration in seconds.
    sr : int
        Sample rate.

    Returns
    -------
    dict — the complete output structure, ready for JSON serialization.
    """
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "audio_duration_sec": round(duration, 3),
            "sample_rate": sr,
            "total_speech_segments": len(segments),
        },
        "num_speakers": num_speakers,
        "speakers": [],
    }

    # ── Build diarized segments grouped by speaker ──
    speaker_segments = {}
    if labels is not None:
        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            speaker_id = int(labels[seg_idx])
            if speaker_id not in speaker_segments:
                speaker_segments[speaker_id] = []
            speaker_segments[speaker_id].append({
                "start": round(seg_start, 3),
                "end": round(seg_end, 3),
                "duration": round(seg_end - seg_start, 3),
            })

    for speaker_id in range(num_speakers):
        speaker_entry = {
            "id": f"S{speaker_id + 1}",
            "segments": speaker_segments.get(speaker_id, []),
            "total_speech_time": round(
                sum(s["duration"] for s in speaker_segments.get(speaker_id, [])), 3
            ),
        }

        # ── Spatial information (Stage 2) ──
        if speaker_dominant is not None and speaker_id in speaker_dominant:
            speaker_entry["dominant_azimuth_deg"] = round(
                speaker_dominant[speaker_id], 2
            )
        if speaker_position is not None and speaker_id in speaker_position:
            speaker_entry["dominant_position"] = speaker_position[speaker_id]

        # ── Tracking information (Stage 3) ──
        if tracking_results is not None and speaker_id in tracking_results:
            track = tracking_results[speaker_id]
            speaker_entry["trajectory"] = [
                {"time": round(p["time"], 3), "azimuth": round(p["azimuth"], 2)}
                for p in track["trajectory"]
            ]
            speaker_entry["movement_detected"] = track["movement_detected"]
            speaker_entry["azimuth_range_deg"] = track["azimuth_range"]
            speaker_entry["crossings"] = track["crossings"]

        output["speakers"].append(speaker_entry)

    return output


def save_json(output: dict, file_path: str) -> None:
    """Write the output dict to a JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[Output] Results saved to {file_path}")


def print_summary(output: dict) -> None:
    """Print a human-readable summary to the console."""
    print("\n" + "=" * 60)
    print("  PIPELINE RESULTS SUMMARY")
    print("=" * 60)
    meta = output["metadata"]
    print(f"  Audio duration:     {meta['audio_duration_sec']} s")
    print(f"  Sample rate:        {meta['sample_rate']} Hz")
    print(f"  Speech segments:    {meta['total_speech_segments']}")
    print(f"  Speakers detected:  {output['num_speakers']}")
    print("-" * 60)

    for speaker in output["speakers"]:
        print(f"\n  Speaker {speaker['id']}:")
        print(f"    Segments:         {len(speaker['segments'])}")
        print(f"    Total speech:     {speaker['total_speech_time']} s")

        if "dominant_position" in speaker:
            print(f"    Position:         {speaker['dominant_position']} "
                  f"({speaker.get('dominant_azimuth_deg', '?')}°)")

        if "movement_detected" in speaker:
            movement = "YES" if speaker["movement_detected"] else "no"
            print(f"    Movement:         {movement} "
                  f"(range: {speaker.get('azimuth_range_deg', '?')}°)")

        if speaker.get("crossings"):
            for c in speaker["crossings"]:
                print(f"    Crossing at {c['time']}s with speaker S{c['with_speaker'] + 1}")

    print("\n" + "=" * 60)
