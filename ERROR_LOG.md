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

*This document is updated every time an error is encountered or resolved.*
