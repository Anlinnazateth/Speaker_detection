"""
Stereo Speaker Localization Demo — Minimal Streamlit Frontend.

Launch:
    streamlit run app.py

Output: text only (speaker count, positions, movement).
No visualizations, no plots, no files written to disk.
"""

from io import BytesIO

import streamlit as st
import numpy as np
import soundfile as sf

# ── Torchaudio compatibility patch (must run before speechbrain import) ──
import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["default"]

from src.config import PipelineConfig
from src.audio_loader import validate_audio
from src.preprocessing import preprocess_stereo
from src.vad import detect_speech_segments
from src.embeddings import extract_embeddings
from src.clustering import cluster_speakers
from src.gcc_phat import estimate_tdoa
from src.azimuth import tdoa_to_azimuth
from src.ild import estimate_ild
from src.association import associate_speakers_with_locations
from src.kalman_tracker import track_speakers
from src.output_formatter import format_output


# ──────────────────────────────────────────────────────────────
# Pipeline runner — fully in-memory, no disk I/O
# ──────────────────────────────────────────────────────────────
def run_analysis(audio: np.ndarray, sr: int) -> dict:
    """
    Run the full Stage-3 pipeline on an in-memory stereo array.

    Parameters
    ----------
    audio : np.ndarray
        Raw stereo audio, any dtype/range (will be validated).
    sr : int
        Sample rate in Hz.

    Returns
    -------
    dict — pipeline result (JSON-serializable).
    """
    config = PipelineConfig()

    # ── Validate & normalize: enforces (N,2), float32, [-1,1] ──
    audio_data = validate_audio(audio, sr, config)

    # ── Preprocess: denoising, HPF, RMS norm, mono downmix ──
    preprocessed = preprocess_stereo(audio_data["audio"], audio_data["sr"], config)

    # ── Stage 1: VAD → Embeddings → Clustering ──
    segments = detect_speech_segments(preprocessed["mono"], preprocessed["sr"], config)

    if len(segments) == 0:
        return format_output(0, [], None, None, None, None,
                             audio_data["duration"], audio_data["sr"])

    embeddings, segment_times = extract_embeddings(
        preprocessed["mono"], preprocessed["sr"], segments, config
    )
    cluster_result = cluster_speakers(embeddings, config, segment_times=segment_times)
    labels = cluster_result["labels"]
    num_speakers = cluster_result["num_speakers"]
    working_segments = segment_times

    # ── Stage 2: TDOA + ILD → Azimuth → Association ──
    tdoa_result = estimate_tdoa(
        preprocessed["left"], preprocessed["right"], preprocessed["sr"], config,
        speech_segments=segments,
    )
    azimuth_array = tdoa_to_azimuth(tdoa_result["tdoa"], config)
    ild_result = estimate_ild(
        preprocessed["left"], preprocessed["right"], preprocessed["sr"], config
    )
    assoc_result = associate_speakers_with_locations(
        working_segments, labels,
        tdoa_result["frame_times"], azimuth_array,
        tdoa_result["gcc_peak"], config,
        ild_azimuth=ild_result["ild_azimuth"],
    )

    # ── Stage 3: Kalman tracking ──
    gcc_peak_by_time = {
        float(t): float(p)
        for t, p in zip(tdoa_result["frame_times"], tdoa_result["gcc_peak"])
    }
    tracking_results = track_speakers(
        assoc_result["speaker_tracks"], gcc_peak_by_time, config
    )

    return format_output(
        num_speakers=num_speakers,
        segments=working_segments,
        labels=labels,
        speaker_dominant=assoc_result["speaker_dominant"],
        speaker_position=assoc_result["speaker_position"],
        tracking_results=tracking_results,
        duration=audio_data["duration"],
        sr=audio_data["sr"],
    )


# ──────────────────────────────────────────────────────────────
# Main app — text output only
# ──────────────────────────────────────────────────────────────
def main():
    st.title("Stereo Speaker Localization Demo")

    uploaded_file = st.file_uploader("Upload a stereo .wav file", type=["wav"])

    if uploaded_file is None:
        st.info("Upload a stereo .wav file to begin.")
        return

    # ── Read directly into memory — no temp file ──
    file_bytes = BytesIO(uploaded_file.getvalue())

    try:
        info = sf.info(file_bytes)
    except Exception:
        st.error("Could not read the file. It may be corrupted.")
        return

    if info.channels != 2:
        st.error(
            f"This file has **{info.channels} channel(s)**. "
            f"Please upload a **stereo (2-channel)** .wav file."
        )
        return

    st.write(
        f"**File loaded** — {info.channels} ch, {info.samplerate} Hz, "
        f"{info.duration:.1f}s, {info.subtype}"
    )

    # ── Analyze button ──
    if not st.button("Analyze", type="primary"):
        return

    # ── Read audio from buffer into memory ──
    file_bytes.seek(0)
    try:
        audio_raw, sr = sf.read(file_bytes, dtype="float32")
    except Exception as e:
        st.error(f"Failed to read audio data: {e}")
        return

    # ── Run pipeline — fully in-memory, no disk I/O ──
    try:
        with st.spinner("Running analysis pipeline (this may take a moment)..."):
            result = run_analysis(audio_raw, sr)

        speakers = result["speakers"]
        num_speakers = result["num_speakers"]

        # ── Results ──
        st.markdown("---")
        st.subheader(f"Detected Speakers: {num_speakers}")

        if num_speakers == 0:
            st.warning("No speech detected in the audio.")
            return

        # ── Per-speaker table ──
        rows = []
        for sp in speakers:
            rows.append({
                "Speaker": sp["id"],
                "Position": sp.get("dominant_position", "—").upper(),
                "Azimuth": f"{sp.get('dominant_azimuth_deg', 0):.1f}°",
                "Movement": "Yes" if sp.get("movement_detected") else "No",
            })
        st.table(rows)

    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()
