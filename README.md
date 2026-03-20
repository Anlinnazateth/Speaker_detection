# Stereo Speaker Detection & Localization

A real-time multi-speaker detection and spatial localization system that processes stereo audio to identify speakers, determine their spatial positions, and track their movement over time.

## How It Works

The system processes stereo `.wav` files through a **3-stage pipeline**:

### Stage 1 — Speaker Identification (Diarization)

1. **Preprocessing** — DC offset removal, denoising (Wiener filter or spectral subtraction with shared inter-channel mask to preserve phase), 80 Hz high-pass filter, RMS normalization, and mono downmix.
2. **Voice Activity Detection** — Uses the pre-trained [Silero VAD](https://github.com/snakers4/silero-vad) model to identify speech segments, with post-processing to merge close segments and reject noise bursts.
3. **Speaker Embeddings** — Each speech segment is converted into a 192-dimensional vector using [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) (via SpeechBrain), capturing speaker identity.
4. **Clustering** — Spectral clustering with an eigen-gap heuristic automatically determines the number of speakers without any prior labels. Falls back to agglomerative hierarchical clustering when needed.

### Stage 2 — Spatial Localization

5. **TDOA Estimation** — [GCC-PHAT](https://en.wikipedia.org/wiki/Generalized_cross-correlation) (Generalized Cross-Correlation with Phase Transform) computes the time difference of arrival between left and right channels per frame, with band-pass filtering, energy gating, and parabolic sub-sample interpolation.
6. **ILD Estimation** — Interaural Level Difference measures per-frame energy differences between channels, useful for amplitude-panned stereo where time delays are negligible.
7. **Azimuth Calculation** — TDOA values are converted to azimuth angles using the far-field plane-wave model: `theta = arcsin((tau * c) / d)`.
8. **Speaker-Location Association** — Confidence-weighted fusion of TDOA and ILD azimuths maps each speaker cluster to a spatial trajectory with median smoothing and outlier rejection.

### Stage 3 — Tracking & Event Detection

9. **Kalman Tracking** — A 1D Kalman filter (constant-velocity model) per speaker smooths azimuth trajectories with adaptive measurement noise scaled by GCC-PHAT confidence.
10. **Movement Detection** — Identifies sustained spatial shifts using median filtering, noise-adaptive thresholds, and sliding-window verification.
11. **Crossing Detection** — Detects when two speakers' trajectories intersect (spatial position swap).

## Pipeline Data Flow

```
Stereo .wav File
       |
  [Audio Loader] --- validate stereo, float32, peak-normalize
       |
  [Preprocessing] -- denoise, HPF, RMS normalize, mono downmix
       |
  [Silero VAD] ----- speech segment detection
       |
  [ECAPA-TDNN] ----- 192-dim speaker embeddings
       |
  [Clustering] ----- speaker count + labels        --> STAGE 1
       |
  [GCC-PHAT + ILD] - per-frame TDOA & azimuth
       |
  [Association] ---- speaker-location mapping       --> STAGE 2
       |
  [Kalman Tracker] - smoothed trajectories          --> STAGE 3
       |
  [Output] --------- structured JSON results
```

## Project Structure

```
├── app.py                  # Streamlit web interface
├── src/
│   ├── config.py           # 70+ tunable pipeline parameters
│   ├── audio_loader.py     # Audio ingestion & validation
│   ├── preprocessing.py    # Denoising, filtering, normalization
│   ├── vad.py              # Voice Activity Detection (Silero)
│   ├── embeddings.py       # Speaker embeddings (ECAPA-TDNN)
│   ├── clustering.py       # Spectral / agglomerative clustering
│   ├── gcc_phat.py         # GCC-PHAT TDOA estimation
│   ├── ild.py              # Interaural Level Difference
│   ├── azimuth.py          # TDOA-to-angle conversion
│   ├── association.py      # Speaker-location fusion
│   ├── kalman_tracker.py   # Kalman filtering & event detection
│   ├── output_formatter.py # JSON result assembly
│   └── pipeline.py         # Pipeline orchestrator & CLI
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/Anlinnazateth/Speaker_detection.git
cd Speaker_detection
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** PyTorch will be installed as a dependency. For GPU support, install the appropriate CUDA version of PyTorch first — see [pytorch.org](https://pytorch.org/get-started/locally/).

## Usage

### Web Interface (Streamlit)

```bash
streamlit run app.py
```

Upload a stereo `.wav` file and click **Analyze**. The app displays detected speaker count, positions (left/center/right), azimuth angles, and movement status.

### Command Line

```bash
python -m src path/to/stereo.wav
```

Optional flags:
- `--mic-distance` — Microphone spacing in meters (default: 0.2)
- `--vad-threshold` — VAD sensitivity (default: 0.5)
- `--stage` — Stop after stage 1, 2, or 3

## Output

The pipeline returns a JSON structure containing:

- **Speaker count** — automatically detected
- **Per-speaker details:**
  - Speech segments with timestamps
  - Total speaking time
  - Dominant position (left / center / right)
  - Dominant azimuth angle in degrees
  - Full azimuth trajectory over time
  - Movement detected (boolean)
  - Crossing events with other speakers

## Requirements

- Python 3.9+
- stereo (2-channel) `.wav` audio input
- See [requirements.txt](requirements.txt) for dependencies

## Key Technologies

| Component | Technology |
|-----------|-----------|
| VAD | Silero VAD (PyTorch) |
| Speaker Embeddings | ECAPA-TDNN via SpeechBrain |
| TDOA | GCC-PHAT with parabolic interpolation |
| Clustering | Spectral Clustering (scikit-learn) |
| Tracking | Kalman Filter (custom) |
| Denoising | Wiener Filter / Spectral Subtraction |
| Web UI | Streamlit |
