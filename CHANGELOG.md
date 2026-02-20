# Changelog — Stereo Speaker Localization Pipeline

All logic changes, feature additions, and bug fixes are documented here in reverse chronological order.

---

## [v0.7] — 2026-02-20 — Modular Denoising, Bug Fixes ERR-011/ERR-012

**Problem:** Noisy audio (especially at low SNR) caused three cascading failures:
1. **Clustering collapse (ERR-011):** `_select_best_partition()` picked degenerate partitions like [13,1] over balanced [8,6] because silhouette was marginally higher. Then `_merge_small_clusters(min=2)` collapsed the 1-member cluster → k=1.
2. **Movement false positives (ERR-012):** Static `movement_threshold=10°` could not accommodate noise-induced azimuth jitter (55–89° ranges at low SNR).
3. **Noise estimation was wrong:** First-0.5s approach included speech (starts at 0.194s), contaminating the noise profile.

**What Changed:**

### 1. Modular Stereo-Safe Denoising (`src/preprocessing.py`)
- **Added** `denoise_method` parameter: `"wiener"` | `"spectral"` | `"none"`
- **Added** `_denoise_stereo()` dispatcher — selects method from config
- **Added** `_spectral_subtract_stereo()` — spectral subtraction with over-subtraction (α=2.0) and spectral floor (β=0.02)
- **Added** `_estimate_noise_floor()` — shared noise estimation function used by both methods
- **Fixed** noise estimation: replaced first-0.5s approach with **per-frequency-bin percentile** across all frames. For each frequency bin, takes the Nth percentile of power values across all time frames — naturally captures the noise floor regardless of where speech occurs.
- All denoisers compute a **single shared spectral gain mask** and apply it to **both L and R channels identically** — preserving inter-channel phase (ITD) and amplitude (ILD)
- **Added** `_compute_diagnostics()` — prints and returns: max amplitude per channel, RMS per channel, inter-channel Pearson correlation
- **Added** correlation sanity check: warns if correlation > 0.95 (channels nearly identical → spatial separation may not exist)
- **Added** NaN guard in `_normalize_rms()` — prevents NaN/Inf propagation

### 2. Clustering Fix — ERR-011 (`src/clustering.py`)
- **Fixed** `_select_best_partition()` — added viability pre-check:
  - A partition is "viable" if every cluster has ≥ `min_members` segments
  - If only one partition is viable, use it (regardless of silhouette)
  - If both viable, fall back to silhouette comparison
- This prevents degenerate partitions like [13, 1] from winning over balanced [8, 6]
- At 0dB SNR: spectral [8, 6] now correctly selected over AHC [13, 1]

### 3. Movement Fix — ERR-012 (`src/kalman_tracker.py`)
- **Fixed** `_detect_sustained_movement()` — noise-adaptive threshold:
  - `effective_threshold = max(base_threshold, noise_scale × observed_std)`
  - At low SNR, observed azimuth std is high → threshold scales up automatically
  - Eliminates all movement false positives across tested SNR range
- Config: `movement_noise_scale=2.5`

### 4. Config Updates (`src/config.py`)
**New parameters:**
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `denoise_method` | `"wiener"` | Denoising algorithm selection |
| `spectral_n_fft` | 2048 | STFT window for spectral subtraction |
| `spectral_hop` | 512 | STFT hop for spectral subtraction |
| `spectral_alpha` | 2.0 | Over-subtraction factor |
| `spectral_beta` | 0.02 | Spectral floor factor |
| `spectral_noise_percentile` | 15.0 | Noise floor percentile for spectral subtraction |
| `correlation_warn_threshold` | 0.95 | Warning threshold for inter-channel correlation |
| `movement_noise_scale` | 2.5 | Movement threshold scale factor × observed std |

**Changed parameters:**
| Parameter | Old | New | Reason |
|-----------|-----|-----|--------|
| `wiener_noise_percentile` | 20.0 | 15.0 | Per-bin percentile (was frame-based) |

**Test Results (all SNR levels):**
```
Clean:  2 speakers  L=-56.2°  R=+66.4°  movement=no/no
20dB:   2 speakers  L=-60.9°  R=+65.9°  movement=no/no
10dB:   2 speakers  L=-61.3°  R=+62.0°  movement=no/no
5dB:    2 speakers  L=-57.9°  R=+59.6°  movement=no/no
0dB:    2 speakers  L=-45.7°  R=+right  movement=no/no   ← was 1 speaker/center
```

**Files changed:**
| File | Change |
|------|--------|
| `src/preprocessing.py` | Full rewrite: modular denoisers, per-bin percentile noise, diagnostics |
| `src/config.py` | Added denoise_method, spectral params, correlation threshold, movement_noise_scale |
| `src/clustering.py` | ERR-011 fix: viability pre-check in `_select_best_partition()` |
| `src/kalman_tracker.py` | ERR-012 fix: noise-adaptive movement threshold |

---

## [v0.6] — 2026-02-18 — Joint Stereo Wiener Denoising

**Problem:** The `noisereduce` spectral gating denoiser was applied per-channel independently. Each channel estimated its own noise profile and applied its own spectral gain, which destroyed inter-channel phase coherence. GCC-PHAT relies on precise phase differences between L/R channels for TDOA estimation, so per-channel denoising degraded localization accuracy under noise.

**What Changed:**

### 1. Joint Stereo Wiener Filter (`src/preprocessing.py`)
- **Replaced** `noisereduce.reduce_noise()` (per-channel) with `_wiener_denoise_stereo()`
- **Removed** `import noisereduce as nr` dependency
- New Wiener filter design:
  1. Computes STFT of both L and R channels
  2. Estimates a **single shared noise power spectrum** from the lowest-energy frames (bottom 20%) across both channels jointly
  3. Over-estimates noise by 1.5× for more aggressive removal
  4. Computes Wiener gain: `G(f) = max(S(f) - N(f), 0) / max(S(f), eps)` with gain floor of 0.08
  5. Applies the **SAME gain mask** to both L and R channels
  6. Reconstructs via inverse STFT
- Phase coherence is preserved because both channels receive identical spectral attenuation

### 2. Config Parameter Changes (`src/config.py`)
- **Removed** `denoise_stationary`, `denoise_prop_decrease` (noisereduce-specific)
- **Added** Wiener filter parameters:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `wiener_n_fft` | 2048 | STFT window size |
| `wiener_hop` | 512 | STFT hop length |
| `wiener_gain_floor` | 0.08 | Minimum spectral gain (prevents musical noise) |
| `wiener_noise_percentile` | 20.0 | Use lowest-energy N% of frames for noise estimate |
| `wiener_noise_overestimate` | 1.5 | Over-estimate noise power for aggressive removal |

### 3. Existing Features Confirmed
- **Frame-level VAD gating**: already active in `gcc_phat.py` via `gcc_speech_only=True` — GCC-PHAT only computed on speech-active frames
- **Azimuth temporal smoothing**: already active in `association.py` via `_median_smooth()` with `azimuth_median_window=5`
- Both remain unchanged

**Processing pipeline (updated):**
```
Raw stereo → DC removal → Joint Wiener denoise (shared gain) → HPF 80Hz → RMS norm → Mono downmix
```

**Files changed:**
| File | Change |
|------|--------|
| `src/preprocessing.py` | Replaced noisereduce with `_wiener_denoise_stereo()`, removed noisereduce import |
| `src/config.py` | Replaced noisereduce params with Wiener filter params |

---

## [v0.5.2] — 2026-02-18 — Remove All Visualization / Text-Only Output

**Problem:** Pipeline generated matplotlib plots (azimuth trajectories, stereo energy, waveform+VAD, polar stereo field, combined dashboard) and saved PNG files to disk. These cluttered the project directory and were not needed for the demo.

**What Changed:**

### 1. Streamlit App (`app.py`)
- **Removed** `import matplotlib`, `matplotlib.use("Agg")`, `import matplotlib.pyplot as plt`
- **Removed** `COLORS` palette constant
- **Removed** `plot_trajectory()` function (azimuth line chart)
- **Removed** `plot_stereo_energy()` function (L/R energy bar chart)
- **Removed** all `st.pyplot()` calls and trajectory/energy sections from `main()`
- **Removed** `audio` and `sr` from `run_analysis()` return value (no longer needed for plots)
- `run_analysis()` now returns only `dict` (was `tuple`)
- UI shows only: speaker count, table with (Speaker ID, Position, Azimuth, Movement)

### 2. Pipeline Orchestrator (`src/pipeline.py`)
- **Removed** `from .visualize import visualize_results`
- **Removed** entire VISUALIZATION section that called `visualize_results()` and saved 5 PNG files
- Pipeline now returns the result dict immediately after `print_summary()` + optional `save_json()`

### 3. No Files Written to Disk
- `app.py`: zero disk I/O (no temp files, no plots)
- `pipeline.py`: only writes JSON when `--output` is explicitly passed via CLI
- `visualize.py`: still exists on disk but is **not imported or called** by anything

**Removed visualization components:**
| Component | Was in | Status |
|-----------|--------|--------|
| Waveform + VAD overlay plot | `visualize.py` → `pipeline.py` | Call removed |
| Speaker diarization timeline | `visualize.py` → `pipeline.py` | Call removed |
| Azimuth trajectory plot | `visualize.py` → `pipeline.py` + `app.py` | Both calls removed |
| Polar stereo field plot | `visualize.py` → `pipeline.py` | Call removed |
| Combined dashboard PNG | `visualize.py` → `pipeline.py` | Call removed |
| Stereo energy bar chart | `app.py` | Function deleted |
| `st.pyplot()` calls | `app.py` | Deleted |
| matplotlib import | `app.py` | Deleted |
| `src/visualize.py` (362 lines) | Dead code — no longer imported | **File deleted** |
| `visualize.cpython-313.pyc` | Stale bytecode cache | **File deleted** |
| 15 stale `results_*.png` files | Leftover from previous runs | **Files deleted** |

---

## [v0.5.1] — 2026-02-18 — Audio Data Flow Consistency Fix

**Problem:** Inconsistent behavior between in-memory processing and file-based loading. When noisy audio was processed directly in-memory, localization was unstable. Saving to `.wav` and re-uploading produced correct results. Root cause: the Streamlit app wrote uploads to a temp file then re-read them, creating a different code path from programmatic callers. No input validation existed anywhere — dtype (float32 vs float64), shape ((N,2) vs (2,N)), and amplitude range were never checked.

**What Changed:**

### 1. New `validate_audio()` — single source of truth (`src/audio_loader.py`)
- **Added** `validate_audio(audio, sr, config)` — validates and normalizes any in-memory stereo array
  - Enforces shape `(N, 2)` (rejects mono, transposed, or multi-channel)
  - Coerces dtype to `float32` (prevents silent float64 promotion)
  - Peak-normalizes to `[-1, 1]`
  - Post-validation assertions catch invariant violations immediately
- **Added** `load_audio_from_buffer(buffer, config)` — reads from `io.BytesIO` (for Streamlit uploads)
- **Refactored** `load_audio()` to delegate to `validate_audio()` internally
- All three entry points now guarantee the same output contract

### 2. Temp file elimination (`app.py`)
- **Removed** `tempfile.NamedTemporaryFile` write + `os.unlink` cleanup
- **Removed** `import os`, `import tempfile`
- Upload bytes are now read directly via `sf.read(BytesIO(...), dtype="float32")`
- `sf.info(BytesIO(...))` used for metadata display (no disk needed)
- **Refactored** `run_analysis(file_path: str)` → `run_analysis(audio: np.ndarray, sr: int)`
  - Now calls `validate_audio()` directly instead of `load_audio()`
  - Zero disk I/O in the entire Streamlit analysis path

### 3. In-memory support for CLI pipeline (`src/pipeline.py`)
- **Added** optional `audio_array` + `sample_rate` parameters to `run_pipeline()`
- When provided, skips `load_audio()` and calls `validate_audio()` directly
- Backward-compatible: `audio_path` still works for CLI usage
- Both paths converge through the same `validate_audio()` → `preprocess_stereo()` chain

### 4. Input assertions at every module boundary
- **`preprocessing.py`**: asserts `(N,2)`, `float32`, peak within `[-1,1]` at entry; asserts `float32` on all output channels
- **`gcc_phat.py`**: asserts 1-D, `float32`, matching length for L/R
- **`ild.py`**: asserts 1-D, `float32`, matching length for L/R
- **`vad.py`**: asserts 1-D, `float32` for mono input
- **`embeddings.py`**: asserts 1-D, `float32` for mono input
- **`preprocessing.py` output**: mono downmix explicitly cast to `float32` (prevents silent float64 promotion from `/ 2.0`)

### 5. No algorithmic changes
- GCC-PHAT, ILD, clustering, Kalman tracking, movement detection — all untouched
- Only data plumbing and validation changed

**Data flow (before):**
```
Streamlit upload → tmp.write(bytes) → disk → sf.read(path) → load_audio() → preprocess → pipeline
```

**Data flow (after):**
```
Streamlit upload → BytesIO(bytes) → sf.read(buffer) → validate_audio() → preprocess → pipeline
```

**Files changed:**
| File | Change |
|------|--------|
| `src/audio_loader.py` | Added `validate_audio()`, `load_audio_from_buffer()`, refactored `load_audio()` |
| `src/preprocessing.py` | Added input/output assertions, fixed mono float32 cast |
| `app.py` | Removed temp file I/O, refactored to in-memory path |
| `src/pipeline.py` | Added `audio_array`/`sample_rate` params, imports `validate_audio` |
| `src/gcc_phat.py` | Added input assertions |
| `src/ild.py` | Added input assertions |
| `src/vad.py` | Added input assertion |
| `src/embeddings.py` | Added input assertion |

---

## [v0.5] — 2026-02-17 — Pipeline Stabilization Under Noise

**Problem:** Under environmental noise the pipeline exhibited three new failure modes:
- Speaker over-detection (3 detected instead of 2) — noise fragments forming their own clusters
- False movement detection — noise-induced azimuth jitter exceeding the simple max-min threshold
- Denoising too aggressive — `prop_decrease=0.75` was smearing stereo phase cues

**What Changed:**

### 1. Controlled Denoising (`src/config.py`)
- **Lowered** `denoise_prop_decrease`: `0.75` → `0.6`
- Less aggressive noise reduction preserves inter-channel phase/timing better
- Better trade-off: still removes noise floor but doesn't corrupt TDOA-critical stereo cues

### 2. Speaker Over-Detection Fix (`src/clustering.py`)
- **Raised** `cluster_min_members`: `1` → `2` (single-segment clusters now get merged)
- **Added** `_merge_short_duration_clusters()` — new post-clustering step that merges any cluster whose total speech duration is below `cluster_min_duration` (1.0s) into the nearest larger cluster by centroid cosine similarity
- **Added** optional `segment_times` parameter to `cluster_speakers()` (backward-compatible, defaults to None)
- **Updated** `pipeline.py` and `app.py` to pass `segment_times` to the clustering step
- Config: `cluster_min_duration=1.0` (seconds)

### 3. Movement Detection Rewrite (`src/kalman_tracker.py`)
- **Replaced** simple `max - min > threshold` logic with `_detect_sustained_movement()`:
  1. **Median filter** the Kalman-smoothed trajectory (window=11) to suppress spike noise
  2. **Half-split check**: compare mean azimuth of first half vs second half — must differ by > `movement_threshold` (10°)
  3. **Sustained shift check**: sliding window of `movement_min_duration` (1.0s) scans trajectory — the local mean must differ from the start mean by > threshold for at least one window position
- This means: short noise spikes are ignored, only real sustained spatial movement is flagged
- Config: `movement_min_duration=1.0`, `movement_median_window=11`

### 4. GCC-PHAT — No Changes
- Already fully hardened in v0.4 (band-pass, energy gating, peak/mean, median filter, outlier rejection)

**New/Changed Config Parameters:**
| Parameter | Old | New | Purpose |
|-----------|-----|-----|---------|
| `denoise_prop_decrease` | 0.75 | 0.6 | Gentler denoising preserves stereo phase |
| `cluster_min_members` | 1 | 2 | Reject single-segment clusters |
| `cluster_min_duration` | — | 1.0s | Reject clusters < 1s total speech |
| `movement_min_duration` | — | 1.0s | Sustained shift window |
| `movement_median_window` | — | 11 | Spike suppression filter |

**Test Result (post-fix):**
```
Speakers: 2
  S1: position=left, azimuth=-61.3°, movement=False
  S2: position=right, azimuth=+69.6°, movement=False
GCC-PHAT: 521/1099 speech-active (skipped 550 non-speech, 28 low-energy)
```

---

## [v0.4] — 2026-02-17 — GCC-PHAT Localization Hardening

**Problem:** Even after v0.3 denoising, localization still collapsed to 0° (Center) under environmental noise. Clean audio worked correctly. Root cause: the GCC-PHAT cross-correlation itself was failing under low SNR — noise frames produced random/biased TDOA values that overwhelmed real estimates.

**What Changed (all in `src/gcc_phat.py` + `src/config.py`):**

### 1. Band-Pass Filtering Before Cross-Correlation
- **Added** `_bandpass_stereo()` — 4th-order Butterworth band-pass (300–3400 Hz) applied per-channel before any GCC computation
- Focuses cross-correlation on the speech frequency range, rejecting low-frequency rumble and high-frequency noise that corrupt TDOA
- Config: `gcc_bandpass_lo=300`, `gcc_bandpass_hi=3400`, `gcc_bandpass_order=4`

### 2. Energy Gating
- **Added** per-frame RMS energy check before GCC computation
- Frames where `max(rms_left, rms_right) < gcc_energy_floor` are skipped entirely (TDOA=0, confidence=0)
- Prevents silent/near-silent frames from producing garbage cross-correlation peaks
- Config: `gcc_energy_floor=1e-4`

### 3. Peak-to-Mean Ratio Validation
- **Added** `peak_to_mean` ratio computation inside `_gcc_phat_single_frame()`
- After cross-correlation, computes `peak / mean(|gcc|)` across the valid delay range
- Frames with ratio < `gcc_peak_to_mean_min` are rejected — the correlation has no clear peak, meaning noise dominates
- Config: `gcc_peak_to_mean_min=3.0`

### 4. Median Filtering on Raw TDOA
- **Added** `_median_smooth_tdoa()` — applies `scipy.ndimage.median_filter` (window=5) to raw TDOA values
- Only smooths frames with non-zero confidence (preserves silence gaps)
- Removes sporadic noisy delay spikes while preserving real speaker transitions
- Config: `gcc_tdoa_median_window=5`

### 5. Outlier Rejection for Extreme Delays
- **Added** `_reject_tdoa_outliers()` — two rejection rules:
  - **Physical limit**: |TDOA| > 1.2 × tau_max is physically impossible → zeroed
  - **Jump limit**: frame-to-frame TDOA change > `gcc_tdoa_jump_max × tau_max` → zeroed
- Prevents sudden delay spikes from corrupting downstream azimuth/tracking
- Config: `gcc_tdoa_jump_max=0.5`

### 6. Updated `_gcc_phat_single_frame()` Return Signature
- Now returns `(tdoa_seconds, peak_value, peak_to_mean_ratio)` instead of `(tdoa_seconds, peak_value)`
- Internal change only — `estimate_tdoa()` public signature unchanged

**New Config Parameters Added:**
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `gcc_bandpass_lo` | 300.0 | Band-pass lower edge (Hz) |
| `gcc_bandpass_hi` | 3400.0 | Band-pass upper edge (Hz) |
| `gcc_bandpass_order` | 4 | Butterworth filter order |
| `gcc_energy_floor` | 1e-4 | RMS floor for energy gating |
| `gcc_peak_to_mean_min` | 3.0 | Min peak/mean ratio |
| `gcc_tdoa_median_window` | 5 | Median filter on TDOA |
| `gcc_tdoa_jump_max` | 0.5 | Max jump (× tau_max) |

**Test Result (post-fix):**
```
Speakers: 2
  S1: position=left, azimuth=-64.6°, movement=False
  S2: position=right, azimuth=+72.6°, movement=False
GCC-PHAT: 515/1099 speech-active (skipped 563 non-speech, 21 low-energy, 0 weak-peak)
Valid TDOA in 269 frames
```

**Processing pipeline per frame (order of operations):**
```
Raw stereo → Band-pass (300-3400 Hz) → Speech mask check → Energy gate →
GCC-PHAT cross-correlation → Peak/mean validation → [collect all frames] →
Median filter on TDOA → Outlier rejection → Output
```

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
