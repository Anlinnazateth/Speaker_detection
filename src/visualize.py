"""
Module 11: Visualization
Generates plots for speaker localization and tracking results.

Produces:
1. Waveform + VAD overlay (which segments are speech)
2. Speaker diarization timeline (who speaks when)
3. Azimuth trajectory per speaker over time (spatial tracking)
4. Stereo field polar plot (speaker positions in azimuth space)
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection


# Color palette for up to 8 speakers
SPEAKER_COLORS = [
    "#2196F3",  # blue
    "#F44336",  # red
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#795548",  # brown
    "#607D8B",  # grey
]


def visualize_results(
    results: dict,
    audio: np.ndarray = None,
    sr: int = None,
    output_prefix: str = "output",
) -> list[str]:
    """
    Generate all visualization plots from pipeline results.

    Parameters
    ----------
    results : dict
        Pipeline output (the JSON structure from output_formatter).
    audio : np.ndarray or None
        (N, 2) stereo audio for waveform plot. If None, waveform is skipped.
    sr : int or None
        Sample rate (required if audio is provided).
    output_prefix : str
        Prefix for output PNG files.

    Returns
    -------
    list of str — file paths of generated plots.
    """
    saved_files = []

    num_speakers = results["num_speakers"]
    speakers = results["speakers"]
    duration = results["metadata"]["audio_duration_sec"]

    # ── Plot 1: Waveform + VAD overlay ──
    if audio is not None and sr is not None:
        path = f"{output_prefix}_1_waveform_vad.png"
        _plot_waveform_vad(audio, sr, speakers, duration, path)
        saved_files.append(path)

    # ── Plot 2: Speaker diarization timeline ──
    if num_speakers > 0:
        path = f"{output_prefix}_2_diarization.png"
        _plot_diarization_timeline(speakers, duration, path)
        saved_files.append(path)

    # ── Plot 3: Azimuth trajectory over time ──
    has_trajectory = any("trajectory" in s and s["trajectory"] for s in speakers)
    if has_trajectory:
        path = f"{output_prefix}_3_azimuth_trajectory.png"
        _plot_azimuth_trajectory(speakers, duration, path)
        saved_files.append(path)

    # ── Plot 4: Polar stereo field ──
    has_position = any("dominant_azimuth_deg" in s for s in speakers)
    if has_position:
        path = f"{output_prefix}_4_stereo_field.png"
        _plot_stereo_field(speakers, path)
        saved_files.append(path)

    # ── Plot 5: Combined dashboard ──
    path = f"{output_prefix}_dashboard.png"
    _plot_dashboard(results, audio, sr, path)
    saved_files.append(path)

    print(f"[Visualize] Saved {len(saved_files)} plots: {saved_files}")
    return saved_files


def _plot_waveform_vad(audio, sr, speakers, duration, path):
    """Plot stereo waveform with VAD speech segments highlighted."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    time_axis = np.arange(audio.shape[0]) / sr

    for ch, label in enumerate(["Left Channel", "Right Channel"]):
        axes[ch].plot(time_axis, audio[:, ch], color="#90CAF9", linewidth=0.3, alpha=0.7)
        axes[ch].set_ylabel(label, fontsize=9)
        axes[ch].set_ylim(-1, 1)
        axes[ch].grid(True, alpha=0.2)

        # Overlay VAD segments colored by speaker
        for speaker in speakers:
            color = SPEAKER_COLORS[int(speaker["id"][1:]) - 1]
            for seg in speaker.get("segments", []):
                axes[ch].axvspan(seg["start"], seg["end"], alpha=0.25, color=color)

    axes[1].set_xlabel("Time (s)")
    fig.suptitle("Stereo Waveform with Speech Segments", fontsize=12, fontweight="bold")

    # Legend
    patches = [mpatches.Patch(color=SPEAKER_COLORS[int(s["id"][1:]) - 1], alpha=0.4,
               label=s["id"]) for s in speakers]
    axes[0].legend(handles=patches, loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_diarization_timeline(speakers, duration, path):
    """Plot horizontal bars showing who speaks when."""
    fig, ax = plt.subplots(figsize=(14, max(2, len(speakers) * 0.8 + 1)))

    for idx, speaker in enumerate(speakers):
        color = SPEAKER_COLORS[idx % len(SPEAKER_COLORS)]
        for seg in speaker.get("segments", []):
            ax.barh(
                idx, seg["duration"], left=seg["start"],
                height=0.6, color=color, alpha=0.8, edgecolor="white", linewidth=0.5,
            )

    ax.set_yticks(range(len(speakers)))
    ax.set_yticklabels([s["id"] for s in speakers], fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_xlim(0, duration)
    ax.set_title("Speaker Diarization Timeline", fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    # Add speech time annotations
    for idx, speaker in enumerate(speakers):
        total = speaker.get("total_speech_time", 0)
        pos = speaker.get("dominant_position", "")
        label = f"{total:.1f}s"
        if pos and pos != "unknown":
            label += f" ({pos})"
        ax.text(duration + 0.3, idx, label, va="center", fontsize=9, color="#555")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_azimuth_trajectory(speakers, duration, path):
    """Plot azimuth over time for each speaker with movement indicators."""
    fig, ax = plt.subplots(figsize=(14, 5))

    for idx, speaker in enumerate(speakers):
        traj = speaker.get("trajectory", [])
        if not traj:
            continue

        color = SPEAKER_COLORS[idx % len(SPEAKER_COLORS)]
        times = [p["time"] for p in traj]
        azimuths = [p["azimuth"] for p in traj]

        ax.scatter(times, azimuths, c=color, s=8, alpha=0.5, zorder=3)
        ax.plot(times, azimuths, color=color, alpha=0.3, linewidth=1, zorder=2)

        # Mark dominant position with a horizontal dashed line
        dom_az = speaker.get("dominant_azimuth_deg", None)
        if dom_az is not None:
            ax.axhline(y=dom_az, color=color, linestyle="--", alpha=0.5, linewidth=1)

        # Movement annotation
        movement = speaker.get("movement_detected", False)
        label = f"{speaker['id']}"
        if movement:
            label += f" (MOVING, range={speaker.get('azimuth_range_deg', '?')}deg)"
        else:
            label += " (stationary)"
        ax.plot([], [], color=color, linewidth=3, label=label)

    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Azimuth (degrees)", fontsize=10)
    ax.set_title("Speaker Azimuth Trajectories", fontsize=12, fontweight="bold")
    ax.set_xlim(0, duration)
    ax.set_ylim(-95, 95)
    ax.axhline(y=0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axhspan(-5, 5, alpha=0.05, color="gray", label="Center zone")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_stereo_field(speakers, path):
    """Polar plot showing speaker positions in the stereo field."""
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})

    # Set 0 degrees at top, clockwise
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # Only show front hemisphere (-90 to +90)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)

    for idx, speaker in enumerate(speakers):
        dom_az = speaker.get("dominant_azimuth_deg", None)
        if dom_az is None:
            continue

        color = SPEAKER_COLORS[idx % len(SPEAKER_COLORS)]
        theta_rad = np.radians(dom_az)

        # Plot speaker position
        ax.scatter([theta_rad], [1.0], c=color, s=200, zorder=5,
                   edgecolors="white", linewidths=2)
        ax.annotate(
            speaker["id"],
            xy=(theta_rad, 1.0),
            xytext=(theta_rad, 1.25),
            fontsize=12, fontweight="bold", color=color,
            ha="center", va="center",
        )

        # Draw trajectory spread if available
        az_range = speaker.get("azimuth_range_deg", 0)
        if az_range > 1:
            theta_min = np.radians(dom_az - az_range / 2)
            theta_max = np.radians(dom_az + az_range / 2)
            thetas = np.linspace(theta_min, theta_max, 50)
            ax.fill_between(thetas, 0.8, 1.2, alpha=0.15, color=color)

    ax.set_rticks([])
    ax.set_title("Stereo Field — Speaker Positions\n", fontsize=12, fontweight="bold")

    # Add L/R/C labels
    ax.text(np.radians(-90), 1.4, "LEFT", ha="center", fontsize=11,
            fontweight="bold", color="#2196F3")
    ax.text(np.radians(90), 1.4, "RIGHT", ha="center", fontsize=11,
            fontweight="bold", color="#F44336")
    ax.text(np.radians(0), 1.4, "CENTER", ha="center", fontsize=10, color="#666")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_dashboard(results, audio, sr, path):
    """Combined 4-panel dashboard with all visualizations."""
    speakers = results["speakers"]
    duration = results["metadata"]["audio_duration_sec"]
    num_speakers = results["num_speakers"]

    has_audio = audio is not None and sr is not None
    has_traj = any("trajectory" in s and s["trajectory"] for s in speakers)
    has_pos = any("dominant_azimuth_deg" in s for s in speakers)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Multi-Speaker Localization Dashboard — {num_speakers} Speaker(s) Detected",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # Layout: 2 rows x 2 cols, polar plot in bottom-right
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # ── Panel 1: Waveform ──
    if has_audio:
        ax1 = fig.add_subplot(gs[0, 0])
        time_axis = np.arange(audio.shape[0]) / sr
        ax1.plot(time_axis, audio[:, 0], color="#64B5F6", linewidth=0.3, alpha=0.6, label="Left")
        ax1.plot(time_axis, audio[:, 1], color="#EF5350", linewidth=0.3, alpha=0.6, label="Right")
        for sp in speakers:
            c = SPEAKER_COLORS[int(sp["id"][1:]) - 1]
            for seg in sp.get("segments", []):
                ax1.axvspan(seg["start"], seg["end"], alpha=0.15, color=c)
        ax1.set_title("Stereo Waveform + VAD", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Time (s)")
        ax1.set_xlim(0, duration)
        ax1.legend(fontsize=8, loc="upper right")
        ax1.grid(True, alpha=0.2)

    # ── Panel 2: Diarization timeline ──
    ax2 = fig.add_subplot(gs[0, 1])
    for idx, sp in enumerate(speakers):
        c = SPEAKER_COLORS[idx % len(SPEAKER_COLORS)]
        for seg in sp.get("segments", []):
            ax2.barh(idx, seg["duration"], left=seg["start"],
                     height=0.5, color=c, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax2.set_yticks(range(len(speakers)))
    ax2.set_yticklabels([s["id"] for s in speakers], fontsize=10, fontweight="bold")
    ax2.set_xlabel("Time (s)")
    ax2.set_xlim(0, duration)
    ax2.set_title("Speaker Diarization", fontsize=10, fontweight="bold")
    ax2.grid(True, axis="x", alpha=0.3)
    ax2.invert_yaxis()

    # ── Panel 3: Azimuth trajectory ──
    if has_traj:
        ax3 = fig.add_subplot(gs[1, 0])
        for idx, sp in enumerate(speakers):
            traj = sp.get("trajectory", [])
            if not traj:
                continue
            c = SPEAKER_COLORS[idx % len(SPEAKER_COLORS)]
            times = [p["time"] for p in traj]
            azs = [p["azimuth"] for p in traj]
            ax3.scatter(times, azs, c=c, s=6, alpha=0.4)
            ax3.plot(times, azs, color=c, alpha=0.3, linewidth=0.8)
            mov = "MOVING" if sp.get("movement_detected") else "stationary"
            ax3.plot([], [], color=c, linewidth=3, label=f"{sp['id']} ({mov})")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Azimuth (deg)")
        ax3.set_xlim(0, duration)
        ax3.set_ylim(-95, 95)
        ax3.axhline(0, color="gray", linewidth=0.5, alpha=0.4)
        ax3.set_title("Azimuth Trajectories", fontsize=10, fontweight="bold")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.2)

    # ── Panel 4: Polar stereo field ──
    if has_pos:
        ax4 = fig.add_subplot(gs[1, 1], projection="polar")
        ax4.set_theta_zero_location("N")
        ax4.set_theta_direction(-1)
        ax4.set_thetamin(-90)
        ax4.set_thetamax(90)
        for idx, sp in enumerate(speakers):
            dom = sp.get("dominant_azimuth_deg", None)
            if dom is None:
                continue
            c = SPEAKER_COLORS[idx % len(SPEAKER_COLORS)]
            ax4.scatter([np.radians(dom)], [1.0], c=c, s=180, zorder=5,
                        edgecolors="white", linewidths=2)
            ax4.annotate(sp["id"], xy=(np.radians(dom), 1.0),
                         xytext=(np.radians(dom), 1.3),
                         fontsize=11, fontweight="bold", color=c, ha="center")
        ax4.set_rticks([])
        ax4.set_title("Stereo Field\n", fontsize=10, fontweight="bold")
        ax4.text(np.radians(-90), 1.5, "L", ha="center", fontsize=11,
                 fontweight="bold", color="#2196F3")
        ax4.text(np.radians(90), 1.5, "R", ha="center", fontsize=11,
                 fontweight="bold", color="#F44336")

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualize] Dashboard saved to {path}")
