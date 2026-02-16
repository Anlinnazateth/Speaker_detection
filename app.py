"""
Stereo Speaker Localization Demo — Minimal Streamlit Frontend.

Launch:
    streamlit run app.py
"""

import os
import tempfile

import streamlit as st
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Torchaudio compatibility patch (must run before speechbrain import) ──
import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["default"]

from src.config import PipelineConfig
from src.audio_loader import load_audio
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

COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
          "#00BCD4", "#795548", "#607D8B"]


# ──────────────────────────────────────────────────────────────
# Pipeline runner — calls module functions directly, no subprocess
# ──────────────────────────────────────────────────────────────
def run_analysis(file_path: str) -> tuple:
    """
    Run the full Stage-3 pipeline.
    Returns (result_dict, audio_ndarray, sample_rate).
    """
    config = PipelineConfig()

    # Stage 1: Load -> Preprocess -> VAD -> Embeddings -> Clustering
    audio_data = load_audio(file_path, config)
    preprocessed = preprocess_stereo(audio_data["audio"], audio_data["sr"], config)
    segments = detect_speech_segments(preprocessed["mono"], preprocessed["sr"], config)

    if len(segments) == 0:
        result = format_output(0, [], None, None, None, None,
                               audio_data["duration"], audio_data["sr"])
        return result, audio_data["audio"], audio_data["sr"]

    embeddings, segment_times = extract_embeddings(
        preprocessed["mono"], preprocessed["sr"], segments, config
    )
    cluster_result = cluster_speakers(embeddings, config)
    labels = cluster_result["labels"]
    num_speakers = cluster_result["num_speakers"]
    working_segments = segment_times

    # Stage 2: TDOA + ILD -> Azimuth -> Association
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

    # Stage 3: Kalman tracking
    gcc_peak_by_time = {
        float(t): float(p)
        for t, p in zip(tdoa_result["frame_times"], tdoa_result["gcc_peak"])
    }
    tracking_results = track_speakers(
        assoc_result["speaker_tracks"], gcc_peak_by_time, config
    )

    result = format_output(
        num_speakers=num_speakers,
        segments=working_segments,
        labels=labels,
        speaker_dominant=assoc_result["speaker_dominant"],
        speaker_position=assoc_result["speaker_position"],
        tracking_results=tracking_results,
        duration=audio_data["duration"],
        sr=audio_data["sr"],
    )
    return result, audio_data["audio"], audio_data["sr"]


# ──────────────────────────────────────────────────────────────
# Matplotlib trajectory plot
# ──────────────────────────────────────────────────────────────
def plot_trajectory(speakers: list, duration: float) -> plt.Figure:
    """Simple line plot: time vs azimuth, one line per speaker."""
    fig, ax = plt.subplots(figsize=(10, 4))

    for idx, sp in enumerate(speakers):
        traj = sp.get("trajectory", [])
        if not traj:
            continue
        times = [p["time"] for p in traj]
        azs = [p["azimuth"] for p in traj]
        color = COLORS[idx % len(COLORS)]
        mov = "moving" if sp.get("movement_detected") else "stationary"
        ax.plot(times, azs, color=color, linewidth=1.5, alpha=0.8,
                label=f"{sp['id']} ({mov})")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Azimuth (degrees)")
    ax.set_title("Speaker Azimuth Trajectory")
    ax.set_xlim(0, duration)
    ax.set_ylim(-95, 95)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.4)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# Matplotlib stereo energy bar chart
# ──────────────────────────────────────────────────────────────
def plot_stereo_energy(audio: np.ndarray, sr: int, speakers: list) -> plt.Figure:
    """Bar chart comparing left vs right channel energy per speaker."""
    fig, ax = plt.subplots(figsize=(6, 3.5))

    ids, l_vals, r_vals, colors_list = [], [], [], []
    for idx, sp in enumerate(speakers):
        segs = sp.get("segments", [])
        if not segs:
            continue
        l_energy, r_energy, n = 0.0, 0.0, 0
        for seg in segs:
            s, e = int(seg["start"] * sr), int(seg["end"] * sr)
            l_energy += np.sum(audio[s:e, 0] ** 2)
            r_energy += np.sum(audio[s:e, 1] ** 2)
            n += (e - s)
        if n == 0:
            continue
        ids.append(sp["id"])
        l_vals.append(np.sqrt(l_energy / n))
        r_vals.append(np.sqrt(r_energy / n))
        colors_list.append(COLORS[idx % len(COLORS)])

    x = np.arange(len(ids))
    w = 0.35
    ax.bar(x - w / 2, l_vals, w, label="Left", color=colors_list, alpha=0.8)
    ax.bar(x + w / 2, r_vals, w, label="Right", color=colors_list, alpha=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(ids)
    ax.set_ylabel("RMS Energy")
    ax.set_title("Stereo Energy per Speaker")
    ax.legend()
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────────────────────
def main():
    st.title("Stereo Speaker Localization Demo")

    uploaded_file = st.file_uploader("Upload a stereo .wav file", type=["wav"])

    if uploaded_file is None:
        st.info("Upload a stereo .wav file to begin.")
        return

    # Save to temp file so pipeline can read it by path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # ── Validate stereo ──
        try:
            info = sf.info(tmp_path)
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

        # ── Run pipeline ──
        with st.spinner("Running analysis pipeline (this may take a moment)..."):
            result, audio, sr = run_analysis(tmp_path)

        speakers = result["speakers"]
        num_speakers = result["num_speakers"]
        duration = result["metadata"]["audio_duration_sec"]

        # ── 1. Number of speakers ──
        st.markdown("---")
        st.subheader(f"Detected Speakers: {num_speakers}")

        if num_speakers == 0:
            st.warning("No speech detected in the audio.")
            return

        # ── 2. Per-speaker details ──
        rows = []
        for sp in speakers:
            rows.append({
                "Speaker": sp["id"],
                "Position": sp.get("dominant_position", "—").upper(),
                "Azimuth": f"{sp.get('dominant_azimuth_deg', 0):.1f}°",
                "Movement": "Yes" if sp.get("movement_detected") else "No",
                "Speech Time": f"{sp['total_speech_time']:.1f}s",
                "Segments": len(sp["segments"]),
            })
        st.table(rows)

        # ── 3. Trajectory plot ──
        has_traj = any(sp.get("trajectory") for sp in speakers)
        if has_traj:
            st.subheader("Azimuth Trajectory")
            fig = plot_trajectory(speakers, duration)
            st.pyplot(fig)
            plt.close(fig)

        # ── 4. Stereo energy ──
        has_pos = any("dominant_position" in sp for sp in speakers)
        if has_pos:
            st.subheader("Stereo Energy")
            fig2 = plot_stereo_energy(audio, sr, speakers)
            st.pyplot(fig2)
            plt.close(fig2)

    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.exception(e)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
