# Error Log — Stereo Speaker Localization Pipeline

Live log of all errors encountered, their root causes, and resolutions.

---

## ERR-001: `ModuleNotFoundError: No module named 'torch'`

| Field       | Value |
|-------------|-------|
| **Date**    | 2026-02-16 |
| **Stage**   | Initial setup |
| **Trigger** | Running `python -m src.pipeline src/test_audio.wav` |
| **Root Cause** | PyTorch not installed in the environment |
| **Fix**     | `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu` |
| **Status**  | RESOLVED |

---

## ERR-002: `ModuleNotFoundError: No module named 'speechbrain'`

| Field       | Value |
|-------------|-------|
| **Date**    | 2026-02-16 |
| **Stage**   | Initial setup |
| **Trigger** | Running pipeline after installing torch |
| **Root Cause** | SpeechBrain not installed |
| **Fix**     | `pip install speechbrain` |
| **Status**  | RESOLVED |

---

## ERR-003: `AttributeError: module 'torchaudio' has no attribute 'list_audio_backends'`

| Field       | Value |
|-------------|-------|
| **Date**    | 2026-02-16 |
| **Stage**   | Embedding extraction |
| **Trigger** | SpeechBrain import calls `torchaudio.list_audio_backends()` |
| **Root Cause** | `torchaudio 2.10+` removed the `list_audio_backends()` function, but SpeechBrain 1.0.x still references it at import time |
| **Fix**     | Added monkey-patch in `src/embeddings.py` and `src/pipeline.py`: `if not hasattr(torchaudio, "list_audio_backends"): torchaudio.list_audio_backends = lambda: ["default"]` |
| **Status**  | RESOLVED (patched) |

---

## ERR-004: `TypeError: hf_hub_download() got an unexpected keyword argument 'use_auth_token'`

| Field       | Value |
|-------------|-------|
| **Date**    | 2026-02-16 |
| **Stage**   | Embedding model download |
| **Trigger** | SpeechBrain's `EncoderClassifier.from_hparams()` calling HuggingFace Hub |
| **Root Cause** | SpeechBrain 1.0.3 uses the deprecated `use_auth_token` parameter, which was removed in `huggingface_hub >= 1.0` |
| **Fix**     | `pip install "huggingface_hub<0.26"` (done twice — once for global Python, once for venv) |
| **Status**  | RESOLVED |
| **Note**    | This error recurred when user created a new venv — same fix applied again |

---

## ERR-005: Test audio had identical L/R channels (mono in stereo wrapper)

| Field       | Value |
|-------------|-------|
| **Date**    | 2026-02-16 |
| **Stage**   | Pipeline validation |
| **Trigger** | Running pipeline on `src/test_audio.wav` |
| **Symptom** | Pipeline output: 1 speaker, 0° azimuth. Channel correlation = 1.0000, max L-R difference = 0.0 |
| **Root Cause** | The uploaded `test_audio.wav` was mono content duplicated into both channels — no stereo separation existed |
| **Fix**     | Created synthetic `test_stereo_2speakers.wav` with real stereo separation (time delays + amplitude panning + timbre shift) |
| **Status**  | RESOLVED |

---

## ERR-006: GCC-PHAT showed 0° azimuth for amplitude-panned stereo

| Field       | Value |
|-------------|-------|
| **Date**    | 2026-02-16 |
| **Stage**   | Stage 2 — Localization |
| **Trigger** | Running pipeline on synthetic stereo audio with amplitude panning |
| **Symptom** | Both speakers mapped to 0° azimuth despite clear L/R panning |
| **Root Cause** | Amplitude panning creates **level differences** but no **time delays**. GCC-PHAT only measures time delays (TDOA), so it cannot detect amplitude-panned source direction. |
| **Fix**     | Created new `src/ild.py` module for Interaural Level Difference (energy-based localization). Updated `src/association.py` to fuse TDOA + ILD with confidence-weighted blending. Updated `src/pipeline.py` to call ILD estimation. |
| **Status**  | RESOLVED |

---

## ERR-007: Noisy audio causes speaker count collapse (2 → 1)

| Field       | Value |
|-------------|-------|
| **Date**    | 2026-02-16 |
| **Stage**   | Full pipeline — noisy audio test |
| **Trigger** | Testing with environmental noise (white noise, forest ambience) added to audio |
| **Symptom** | Pipeline detects only 1 speaker instead of 2. Embeddings become too similar under noise. Localization unstable. |
| **Root Cause** | Multiple compounding issues: (1) No denoising — noise corrupts all downstream stages. (2) VAD threshold too high — misses quieter speech in noise, detects noise bursts. (3) Embeddings computed on noisy audio — reduced discriminability. (4) Eigen-gap threshold too aggressive (0.05) — collapses to k=1 when embedding similarity increases. (5) GCC-PHAT computed on non-speech frames — noise frames inject random TDOA values. |
| **Fix**     | 5-part noise robustness improvement (see CHANGELOG.md v0.3): spectral gating denoising, VAD threshold/duration tuning + noise burst rejection, per-slice embedding normalization, clustering force-split fallback, speech-only GCC-PHAT + median smoothing + outlier rejection |
| **Status**  | RESOLVED |
| **Post-fix test** | 2 speakers detected: S1 at -60.3° (left), S2 at +55.4° (right) |

---

## ERR-008: GCC-PHAT localization collapses to 0° (Center) under noise

| Field       | Value |
|-------------|-------|
| **Date**    | 2026-02-17 |
| **Stage**   | Stage 2 — GCC-PHAT TDOA Estimation |
| **Trigger** | Testing pipeline with environmental noise (white noise, forest ambience) added to stereo audio |
| **Symptom** | Azimuth values collapse to 0° for all speakers. Speakers show position="center" instead of left/right. Clean audio still works correctly. |
| **Root Cause** | The GCC-PHAT cross-correlation itself fails under low SNR. Five compounding issues: (1) Full-bandwidth input — noise outside speech range (< 300 Hz, > 3400 Hz) corrupts cross-correlation. (2) No energy gating — silent/noisy frames produce random TDOA peaks. (3) No peak quality check — flat/noisy GCC correlation accepted as valid. (4) No TDOA smoothing — sporadic noise spikes in raw delay estimates. (5) No jump rejection — extreme frame-to-frame TDOA changes accepted as real. |
| **Fix**     | 5-part GCC-PHAT hardening (see CHANGELOG.md v0.4): (1) Band-pass filter 300–3400 Hz before cross-correlation. (2) Energy gating with RMS floor. (3) Peak-to-mean ratio validation (min 3.0). (4) Median filter on raw TDOA. (5) Outlier rejection for impossible delays and extreme jumps. |
| **Status**  | RESOLVED |
| **Post-fix test** | 2 speakers detected: S1 at -64.6° (left), S2 at +72.6° (right). GCC-PHAT: 515/1099 speech-active frames, 21 energy-gated, 269 valid TDOA. |

---

## ERR-009: Speaker over-detection under noise (3 instead of 2)

| Field       | Value |
|-------------|-------|
| **Date**    | 2026-02-17 |
| **Stage**   | Stage 1 — Clustering |
| **Trigger** | Testing with environmental noise added to stereo audio |
| **Symptom** | Pipeline detects 3 speakers instead of 2. Noise fragments form their own cluster with < 1s of speech. |
| **Root Cause** | `cluster_min_members=1` allowed single-segment noise clusters to survive. No duration-based validation — a cluster with 0.3s of noise-triggered speech could become a "speaker". |
| **Fix**     | (1) Raised `cluster_min_members` from 1 → 2. (2) Added `_merge_short_duration_clusters()` — clusters with total speech < `cluster_min_duration` (1.0s) get merged into nearest large cluster. (3) Pass `segment_times` to clustering for duration calculation. See CHANGELOG.md v0.5. |
| **Status**  | RESOLVED |
| **Post-fix test** | 2 speakers detected correctly. |

---

## ERR-010: False movement detection under noise

| Field       | Value |
|-------------|-------|
| **Date**    | 2026-02-17 |
| **Stage**   | Stage 3 — Kalman Tracking |
| **Trigger** | Testing with environmental noise |
| **Symptom** | Stationary speakers flagged as "MOVING". Azimuth range inflated by noise spikes. |
| **Root Cause** | Movement detection used simple `max(smoothed) - min(smoothed) > 10°`. Even after Kalman smoothing, a few noise-induced azimuth outliers inflated the range past the threshold. No check for sustained vs. momentary shift. |
| **Fix**     | Replaced with `_detect_sustained_movement()`: (1) Median filter (window=11) on trajectory. (2) Half-split mean comparison — first half vs second half must differ by > 10°. (3) Sliding window (1.0s) sustained shift verification. Short spikes ignored. See CHANGELOG.md v0.5. |
| **Status**  | RESOLVED |
| **Post-fix test** | Both stationary speakers correctly flagged as "STATIONARY". |

---

## ERR-011: Clustering collapse at 5 dB SNR — spectral partition discarded for AHC

| Field       | Value |
|-------------|-------|
| **Date**    | 2026-02-18 |
| **Stage**   | Stage 1 — Clustering (`_select_best_partition`) |
| **Trigger** | EOD testing at 5 dB and 0 dB SNR white noise |
| **Symptom** | Eigen-gap correctly estimates k=2. Spectral clustering produces correct [8,8] split. But final result is 1 speaker. |
| **Root Cause** | `_select_best_partition()` compares silhouette scores: spectral=0.3488 vs AHC=0.3598. AHC wins by 0.01 margin, but AHC's partition is [15,1] — one outlier cluster. `_merge_small_clusters` then merges the 1-member cluster, collapsing to k=1. The correct spectral [8,8] split is discarded. Bug: partition selection doesn't check if the chosen partition will survive post-processing. |
| **Fix**     | Added viability pre-check in `_select_best_partition()`: partitions where any cluster < `min_members` are rejected in favor of viable alternatives. Spectral [8,6] now selected over degenerate AHC [13,1]. See CHANGELOG v0.7. |
| **Status**  | RESOLVED |
| **Severity** | Critical — broke speaker detection at moderate noise levels |

---

## ERR-012: Movement false positives at 20 dB and 10 dB SNR

| Field       | Value |
|-------------|-------|
| **Date**    | 2026-02-18 |
| **Stage**   | Stage 3 — Movement detection (`_detect_sustained_movement`) |
| **Trigger** | EOD testing at 20 dB and 10 dB SNR white noise |
| **Symptom** | Both stationary speakers flagged as "MOVING" at 20 dB and 10 dB. Azimuth std is 20-25 deg (vs 1-2 deg clean). |
| **Root Cause** | ILD noise creates high azimuth variance (~20 deg std) under moderate noise. The `_detect_sustained_movement()` half-split mean comparison sees >10 deg difference from random ILD drift, not actual speaker movement. The `movement_threshold=10 deg` is static and does not account for the noise floor. |
| **Fix**     | Made movement threshold noise-adaptive: `effective_threshold = max(base_threshold, noise_scale × observed_std)`. At low SNR, observed azimuth std is large → threshold automatically scales up. Config: `movement_noise_scale=2.5`. See CHANGELOG v0.7. |
| **Status**  | RESOLVED |
| **Severity** | High — false movement flags under any noise |

---

## ERR-013: Inconsistent audio handling between in-memory and file-based paths

| Field       | Value |
|-------------|-------|
| **Date**    | 2026-02-18 |
| **Stage**   | Pre-processing / Audio Loading |
| **Trigger** | Processing noisy audio directly in-memory vs. saving to `.wav` and re-uploading |
| **Symptom** | In-memory processing produces unstable localization. Saving processed audio to `.wav` then re-uploading gives correct results. |
| **Root Cause** | Three compounding issues: (1) Streamlit app wrote uploads to a temp `.wav` file then re-read them — this disk roundtrip silently enforced float32 and peak normalization that in-memory callers skipped. (2) No `validate_audio()` function existed — `load_audio()` only accepted file paths, so any in-memory array bypassed all validation (dtype, shape, range). (3) No assertions at module boundaries — float64 promotion, transposed arrays, or un-normalized amplitudes propagated silently, causing GCC-PHAT energy thresholds and peak-to-mean ratios to produce wrong results. (4) Mono downmix used `/ 2.0` (Python float64), silently promoting float32 channels to float64 before downstream modules received them. |
| **Fix**     | (1) Added `validate_audio()` as single source of truth for all audio entry points — enforces (N,2), float32, [-1,1]. (2) Eliminated temp file writing in Streamlit — reads from `BytesIO` directly. (3) Added input contract assertions to `preprocessing.py`, `gcc_phat.py`, `ild.py`, `vad.py`, `embeddings.py`. (4) Fixed mono downmix to explicitly maintain float32. (5) Added `audio_array`/`sample_rate` params to `run_pipeline()` for consistent in-memory usage. See CHANGELOG.md v0.5.1. |
| **Status**  | RESOLVED |
| **Post-fix** | All paths (CLI file, Streamlit upload, programmatic array) converge through `validate_audio()` → `preprocess_stereo()` with identical guarantees. |

---

*This document is updated every time an error is encountered or resolved.*
