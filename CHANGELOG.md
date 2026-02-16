# Changelog — Stereo Speaker Localization Pipeline

All logic changes, feature additions, and bug fixes are documented here in reverse chronological order.

---

## [v0.3] — 2026-02-16 — Noise Robustness Improvements

**Problem:** Pipeline tested with environmental noise (white noise, forest ambience). System failed to:
- Correctly detect 2 speakers (collapsed to 1)
- Differentiate speakers via embeddings
- Maintain stable stereo localization

**What Changed:**

### 1. Pre-Denoising Stage (`src/preprocessing.py`)
- **Added** `noisereduce` spectral gating **before** high-pass filter
- Denoising is applied **per-channel** (left and right independently) to preserve stereo structure
- Controlled by 3 new config params: `denoise_enabled`, `denoise_stationary`, `denoise_prop_decrease`
- Uses `noisereduce.reduce_noise()` with `n_fft=2048, hop_length=512`

### 2. VAD Stability (`src/vad.py`, `src/config.py`)
- **Lowered** `vad_threshold`: `0.5` → `0.35` (catches quieter speech buried in noise)
- **Raised** `vad_min_duration`: `0.25s` → `0.30s` (rejects segments < 300ms, filters short noise bursts)
- **Raised** `vad_merge_gap`: `0.30s` → `0.40s` (merges fragmented speech split by noise)
- **Added** `_reject_low_energy()` function — drops segments whose RMS energy is < 5% of the median segment energy (filters noise bursts that Silero VAD mis-detected as speech)

### 3. Embedding Quality (`src/embeddings.py`)
- **Added** per-slice **peak normalization** before embedding extraction
- Each audio slice is normalized to `[-1, 1]` range so the ECAPA-TDNN model sees consistent input levels regardless of background noise amplitude
- Prevents noise-dominated segments from producing weak/noisy embeddings

### 4. Clustering Robustness (`src/clustering.py`, `src/config.py`)
- **Lowered** eigen-gap threshold: `0.05` → `0.02` (configurable via `cluster_eigengap_min`)
  - Prevents premature collapse to k=1 when noise makes embeddings more similar
- **Lowered** `cluster_min_members`: `2` → `1` (keeps small clusters that may be valid speakers)
- **Raised** `cluster_cosine_threshold`: `0.3` → `0.4` (wider AHC separation for noisy embeddings)
- **Added force-split fallback**: if eigen-gap says k=1 but the max pairwise cosine distance between any two embeddings exceeds `cluster_force_split_dist` (0.25), forces k=2 and runs clustering anyway

### 5. Localization Stability (`src/gcc_phat.py`, `src/association.py`)
- **GCC-PHAT speech-only mode**: new `speech_segments` parameter; when `gcc_speech_only=True`, only computes GCC-PHAT on frames overlapping with VAD speech segments. Non-speech frames get TDOA=0 and confidence=0.
- **Median filter smoothing**: applies `scipy.ndimage.median_filter` (window=5 frames) to fused azimuth trajectory to remove frame-to-frame jitter
- **Outlier rejection**: frames where azimuth deviates > 2.5σ from local median (31-frame rolling window) are replaced with the local median and their confidence is zeroed

### 6. Pipeline & App Wiring (`src/pipeline.py`, `app.py`)
- Both `pipeline.py` and `app.py` now pass VAD `segments` to `estimate_tdoa()` via the new `speech_segments` parameter

### 7. Dependencies (`requirements.txt`)
- **Added** `noisereduce>=3.0`

**Test Result (post-fix):**
```
Speakers: 2
  S1: position=left, azimuth=-60.3°, movement=False
  S2: position=right, azimuth=+55.4°, movement=False
GCC-PHAT: 536/1099 speech-active frames (skipped 563 non-speech)
```

---

## [v0.2] — 2026-02-16 — ILD Fusion + Visualization + Streamlit Frontend

**Problem:** Test audio (`test_stereo_2speakers.wav`) used amplitude panning for stereo separation, but GCC-PHAT only detects **time delays**. Amplitude-panned audio has near-zero TDOA, so both speakers showed 0° azimuth.

**What Changed:**

### 1. ILD Module (`src/ild.py`) — NEW FILE
- Computes per-frame Interaural Level Difference (energy ratio in dB between L/R channels)
- Maps ILD to azimuth via `azimuth = 90 * tanh(ILD_dB / 6.0)`
- Complements GCC-PHAT for amplitude-panned stereo where time delays are negligible

### 2. Azimuth Fusion (`src/association.py`)
- Updated to accept optional `ild_azimuth` parameter
- Fuses TDOA and ILD estimates with confidence-weighted blending:
  - TDOA weight scales with `confidence * (|tdoa_az| / (|tdoa_az| + 1))`
  - ILD weight is always 1.0 (always informative for panned sources)

### 3. Visualization Module (`src/visualize.py`) — NEW FILE
- Matplotlib-based: waveform+VAD overlay, diarization timeline, azimuth trajectory, polar stereo field, combined dashboard
- Generates PNG files automatically at end of pipeline run

### 4. Streamlit Frontend (`app.py`) — NEW FILE
- Minimal single-page app: file uploader, "Analyze" button, results table, trajectory plot, stereo energy chart
- Uses matplotlib only (no plotly), no sidebar, direct module imports

### 5. Synthetic Test Audio (`test_stereo_2speakers.wav`)
- Created synthetic 2-speaker stereo file with:
  - Speaker 1: panned left (amplitude ratio 0.85/0.45 L/R + 27-sample time delay)
  - Speaker 2: panned right (amplitude ratio 0.45/0.85 L/R + opposite time delay)
  - Timbre differentiation via resampling for speaker 2

**Test Result (post-fix):**
```
Speakers: 2
  S1: position=left, azimuth=-53.59°, movement=False
  S2: position=right, azimuth=+52.14°, movement=False
```

---

## [v0.1] — 2026-02-16 — Initial Pipeline Implementation

**What was built:**

Full 3-stage multi-speaker localization and tracking pipeline:

### Stage 1: Speaker Detection & Diarization
- `src/audio_loader.py` — Load/validate stereo WAV, normalize peak to [-1,1]
- `src/preprocessing.py` — DC removal, 2nd-order Butterworth HPF at 80Hz, RMS normalization to -20dBFS, mono downmix
- `src/vad.py` — Silero VAD (PyTorch JIT), resamples to 16kHz, merges close segments, drops short ones
- `src/embeddings.py` — ECAPA-TDNN via SpeechBrain, 192-dim L2-normalized vectors, splits long segments into 3s sub-segments
- `src/clustering.py` — Spectral clustering with eigen-gap heuristic + AHC fallback, silhouette score comparison, small cluster merging

### Stage 2: Spatial Localization
- `src/gcc_phat.py` — Per-frame GCC-PHAT with Hann windowing, PHAT whitening, tau_max constraint, parabolic interpolation
- `src/azimuth.py` — Far-field model: `θ = arcsin((τ·c)/d)`, clips sin_theta to [-1,1]
- `src/association.py` — Speaker-location mapping using median azimuth per speaker

### Stage 3: Tracking
- `src/kalman_tracker.py` — 1D constant-velocity Kalman filter, adaptive measurement noise from GCC confidence, innovation gating at 3σ, crossing detection via interpolated trajectory comparison

### Infrastructure
- `src/config.py` — Centralized `PipelineConfig` dataclass with 28 tunable parameters
- `src/output_formatter.py` — Structured JSON output with metadata, per-speaker segments, spatial info, trajectories
- `src/pipeline.py` — Orchestrator with CLI argparse, stages 1–3

**Initial test:** 1 speaker at 0° (test audio was mono-in-stereo-wrapper with identical L/R channels)
