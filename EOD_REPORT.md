# EOD Engineering Report — Stereo Speaker Localization Pipeline

**Date:** 2026-02-18
**Version:** v0.5
**Reviewer:** Senior ML Engineer
**Test Audio:** Synthetic 2-speaker stereo (22s, 48kHz), panned L/R with TDOA + amplitude separation

---

## 1. Pipeline Status

### Module Health

| Module | File | Status | Notes |
|--------|------|--------|-------|
| Audio Loader | `audio_loader.py` | OK | Validates stereo, normalizes peak |
| Preprocessing | `preprocessing.py` | OK | Denoising + HPF + RMS norm |
| VAD | `vad.py` | OK | Silero VAD + energy rejection |
| Embeddings | `embeddings.py` | OK | ECAPA-TDNN + per-slice normalization |
| Clustering | `clustering.py` | DEFECT | Partition selection bug (see Section 4) |
| GCC-PHAT | `gcc_phat.py` | OK | 5-layer defense hardened |
| ILD | `ild.py` | OK | Complementary to GCC-PHAT |
| Azimuth | `azimuth.py` | OK | Far-field model |
| Association | `association.py` | OK | Fusion + median + outlier rejection |
| Kalman Tracker | `kalman_tracker.py` | WEAK | Movement false positives at moderate noise |
| Output Formatter | `output_formatter.py` | OK | JSON structure |
| Visualization | `visualize.py` | OK | Matplotlib dashboard |
| Streamlit App | `app.py` | OK | Minimal frontend working |

### Runtime Errors
- **None.** Pipeline completes at all SNR levels (clean through 0 dB) without exceptions.

### Deprecation Warnings
- `torch.cuda.amp.custom_fwd` — FutureWarning from SpeechBrain. Non-blocking. Will need SpeechBrain upgrade eventually.
- `huggingface_hub` — Pinned to `<0.26` due to `use_auth_token` removal. Technical debt.

---

## 2. Clean Audio Performance

| Metric | Value | Grade |
|--------|-------|-------|
| Speakers detected | 2 (correct) | PASS |
| S1 position | Left, -61.3 deg | PASS |
| S2 position | Right, +69.6 deg | PASS |
| S1 movement | Stationary (correct) | PASS |
| S2 movement | Stationary (correct) | PASS |
| S1 azimuth std | 0.76 deg | Excellent |
| S2 azimuth std | 2.34 deg | Good |
| GCC valid frames | 272/1099 (24.7%) | OK |
| GCC peak mean | 0.5919 | Strong |
| Cluster silhouette | 0.6932 | Good |
| Trajectory points | 283 (S1), 266 (S2) | Good |

**Assessment:** Clean audio works correctly across all three requirements: localization, speaker recognition, and movement detection. Azimuth is stable with low standard deviation. All stages function as designed.

---

## 3. Noisy Audio Performance

### Summary Matrix

| Metric | Clean | 20 dB | 10 dB | 5 dB | 0 dB |
|--------|-------|-------|-------|------|------|
| Speakers detected | 2 | 2 | 2 | **1** | **1** |
| Correct count | YES | YES | YES | **NO** | **NO** |
| L/R separation | YES | YES | YES | **NO** | **NO** |
| S1 azimuth | -61.3 | -70.1 | -71.2 | -1.1 | -0.5 |
| S2 azimuth | +69.6 | +75.4 | +71.6 | N/A | N/A |
| S1 az_std | 0.76 | **23.81** | **22.82** | 9.05 | 9.26 |
| S2 az_std | 2.34 | **20.28** | **25.18** | N/A | N/A |
| Movement false positive | NO | **YES** | **YES** | NO | NO |
| GCC valid % | 24.7% | 8.3% | 2.5% | **0.5%** | **0.0%** |
| GCC peak mean | 0.59 | 0.17 | 0.11 | 0.10 | 0.09 |
| Silhouette | 0.69 | 0.63 | 0.47 | N/A | N/A |
| Frame reject % | 75.3% | 84.9% | 79.6% | 82.1% | 82.3% |

### Failure Breakdown

**20 dB SNR (mild noise):**
- Speaker count: PASS (2 detected)
- Localization: PASS (correct L/R)
- Movement: **FAIL** (false positive — both speakers flagged as moving)
- Azimuth stability: **DEGRADED** (std jumped from ~1-2 deg to ~20-24 deg)

**10 dB SNR (moderate noise):**
- Speaker count: PASS (2 detected)
- Localization: PASS (correct L/R)
- Movement: **FAIL** (false positive)
- Azimuth stability: **DEGRADED** (std ~23-25 deg)
- GCC valid down to 2.5% of frames

**5 dB SNR (severe noise):**
- Speaker count: **FAIL** (1 instead of 2)
- Localization: **FAIL** (collapses to center 0 deg)
- Movement: PASS (correctly says stationary, but only because it's 1 merged speaker)

**0 dB SNR (extreme noise):**
- Speaker count: **FAIL** (1 instead of 2)
- Localization: **FAIL** (collapses to center 0 deg)
- GCC-PHAT: **TOTAL FAILURE** (0 valid TDOA frames)

---

## 4. Root Cause Analysis

### DEFECT-1: Clustering Collapse at 5 dB / 0 dB

**Origin:** `clustering.py` — `_select_best_partition()` + `_merge_small_clusters()`

**Traced failure chain (5 dB):**

```
1. Eigen-gap correctly estimates k=2
2. Spectral clustering produces [0,1,0,1,0,1,0,1...] → balanced 8/8 split (CORRECT)
3. AHC clustering produces [0,0,0,...,1,0,0] → imbalanced 15/1 split (WRONG)
4. Silhouette comparison: spectral=0.3488, AHC=0.3598
5. AHC wins by 0.01 margin → [15,1] selected as "best"
6. _merge_small_clusters: cluster 1 has 1 member < min_members=2 → merged into cluster 0
7. Result: 1 speaker (INCORRECT)
```

**Why it happens technically:** Under noise, embeddings become more uniformly distributed. AHC with cosine distance and average linkage tends to isolate the most distant single embedding as its own cluster rather than finding the natural split. The silhouette score slightly favors this "one outlier" partition because the dense 15-member cluster has high internal coherence. But this partition is unusable — the min_members merge immediately collapses it.

**Core bug:** `_select_best_partition()` evaluates partitions in isolation without considering that the min_members post-processing will destroy the AHC partition. The spectral partition [8,8] is the correct and usable answer, but it loses the silhouette comparison.

### DEFECT-2: Movement False Positives at 20 dB / 10 dB

**Origin:** `kalman_tracker.py` — `_detect_sustained_movement()` + ILD noise

**Root cause:** At 20 dB / 10 dB, the azimuth standard deviation is 20-25 degrees (vs 1-2 deg clean). This comes predominantly from ILD noise — the ILD estimator sees noise energy on both channels, which randomizes the level difference. Even after median filtering (window=11), the half-split mean comparison can show >10 deg difference from random noise drift because:

1. ILD noise is not zero-mean per half — random noise on stereo channels creates asymmetric energy patterns in the first vs. second half of the file
2. The 11-point median filter is insufficient to suppress 20+ deg std variation
3. The `movement_threshold=10 deg` is too low relative to the ILD noise floor at moderate SNR

### DEFECT-3: GCC-PHAT Total Failure at 0 dB

**Origin:** `gcc_phat.py` — fundamental SNR limit

**Not a bug.** At 0 dB SNR, the noise power equals the signal power. The peak-to-mean ratio check (`gcc_peak_to_mean_min=3.0`) correctly rejects all frames because there genuinely is no detectable cross-correlation peak. The band-pass filter helps but cannot recover signal-level TDOA at 0 dB. This is a physics limitation, not a code defect.

At 0 dB the pipeline correctly falls back to ILD-only localization, but ILD is also degraded (range shrinks from [-9.8, 8.4] to [-7.2, 7.5] dB).

### DEFECT-4: Embedding Quality Under Noise

**Origin:** `embeddings.py` / test audio design

**Contributing factor, not root cause.** The synthetic test audio uses resampling-based timbre shift, which produces weak speaker differentiation even on clean audio (inter-cluster gap is *negative*: -0.0067). The ECAPA-TDNN model struggles to differentiate these synthetic voices. Under noise, the gap worsens to -0.016. With real human speakers, embedding separation would likely be much stronger.

**However:** the pipeline should not depend on large embedding gaps. The spectral clustering correctly finds the split even at 5 dB — the problem is the partition selection bug discarding it.

---

## 5. Stability Metrics

### Azimuth Variance by SNR

| SNR | S1 az_std (deg) | S2 az_std (deg) | Degradation factor |
|-----|-----------------|-----------------|-------------------|
| Clean | 0.76 | 2.34 | 1.0x (baseline) |
| 20 dB | 23.81 | 20.28 | ~13x |
| 10 dB | 22.82 | 25.18 | ~16x |
| 5 dB | 9.05 (1 spk) | N/A | N/A (collapsed) |
| 0 dB | 9.26 (1 spk) | N/A | N/A (collapsed) |

**Assessment:** Azimuth variance jumps ~13x at 20 dB SNR. The ILD-based azimuth is the primary contributor since GCC-PHAT valid frames drop from 24.7% to 8.3%. The pipeline relies heavily on ILD under noise, and ILD is inherently noisier than TDOA.

### Cluster Compactness

| SNR | Silhouette | Max Pairwise Dist | Inter-Intra Gap |
|-----|-----------|-------------------|-----------------|
| Clean | 0.693 | 0.512 | -0.007 |
| 20 dB | 0.632 | 0.453 | -0.010 |
| 10 dB | 0.474 | 0.561 | -0.016 |
| 5 dB | N/A (1 spk) | 0.584 | -0.008 |
| 0 dB | N/A (1 spk) | 0.587 | -0.010 |

**Assessment:** The negative inter-intra gap across ALL conditions (including clean) indicates that the synthetic test speakers have fundamentally weak embedding separation. The clustering works despite this because spectral methods exploit global structure, but the margin is thin. Real speakers would show positive gaps.

### GCC-PHAT Frame Acceptance

| SNR | Active Frames | Energy Gated | Weak Peak | Valid TDOA | Accept Rate |
|-----|--------------|-------------|-----------|-----------|-------------|
| Clean | 521 | 28 | 0 | 272 | 24.7% |
| 20 dB | 504 | 0 | 57 | 91 | 8.3% |
| 10 dB | 468 | 0 | 110 | 28 | 2.5% |
| 5 dB | 422 | 0 | 155 | 5 | 0.5% |
| 0 dB | 398 | 0 | 200 | 0 | 0.0% |

**Assessment:** The peak-to-mean validation is the primary rejection mechanism under noise (not energy gating). The GCC-PHAT defense layers are working as designed — they correctly identify and reject unreliable frames. The problem is that at low SNR, too few reliable frames remain for robust TDOA.

---

## 6. Recommendations

### Short-Term Fixes (< 1 day)

**FIX-1: Partition selection bug** (Critical)
In `_select_best_partition()`, add a viability pre-check: reject any partition where the smallest cluster has fewer than `cluster_min_members` segments. If only one partition is viable, use it regardless of silhouette score. This directly fixes the 5 dB collapse.

```python
# Reject partitions that would be destroyed by min_members merge
viable_a = min(np.bincount(labels_a)) >= min_members
viable_b = min(np.bincount(labels_b)) >= min_members
if viable_a and not viable_b: return labels_a
if viable_b and not viable_a: return labels_b
# Both viable (or both non-viable): fall back to silhouette
```

**FIX-2: Movement threshold scaling** (High)
Scale `movement_threshold` by the observed azimuth noise floor. If `az_std > 15 deg`, raise threshold to `2 * az_std` to prevent false positives:

```python
effective_threshold = max(config.movement_threshold, 2.0 * np.std(filtered))
```

**FIX-3: ILD confidence weighting** (Medium)
In `association.py` `_fuse_azimuth()`, reduce ILD weight when the ILD range is small (indicating noise dominance). Currently ILD weight is always 1.0, which is too aggressive under noise.

### Medium-Term Improvements (1–3 days)

**IMP-1: SNR-adaptive pipeline**
Estimate per-segment SNR from the ratio of speech energy to inter-speech noise energy. Use this to:
- Scale `gcc_peak_to_mean_min` dynamically (lower threshold at high SNR for more valid frames)
- Adjust `denoise_prop_decrease` per-file
- Set azimuth confidence weights

**IMP-2: Wiener filter or MVDR beamforming**
Replace `noisereduce` spectral gating (which operates per-channel independently) with a stereo-aware denoising method that estimates noise from both channels jointly and preserves inter-channel phase.

**IMP-3: Multi-resolution GCC-PHAT**
Run GCC-PHAT at multiple frame lengths (20ms, 40ms, 80ms) and fuse the results. Longer frames improve SNR at the cost of time resolution. At low SNR, the 80ms estimates would survive when 40ms fails.

### Architecture-Level Upgrades (> 1 week)

**ARCH-1: Neural TDOA estimation**
Replace GCC-PHAT with a learned cross-correlation model (e.g., SeldNet-style DOA network) trained on noisy stereo pairs. Would handle low-SNR scenarios where GCC-PHAT fundamentally cannot.

**ARCH-2: End-to-end diarization**
Replace VAD + embedding + clustering chain with an end-to-end model like pyannote 3.x. Would eliminate the partition selection bug entirely and handle overlapping speech.

**ARCH-3: Real test audio**
The synthetic test audio is a significant limitation. The resampling-based timbre shift creates unrealistically similar embeddings. All clustering metrics (silhouette, inter-intra gap) are pessimistic. Testing with real multi-speaker recordings is essential for meaningful evaluation.

---

## 7. Overall Project Status

### Maturity Level: **Late Prototype / Early Beta**

### Confidence by Component

| Component | Confidence | Production-Ready? |
|-----------|-----------|-------------------|
| Audio loading + preprocessing | High | Yes |
| Spectral denoising | Medium | Yes (with tuning) |
| VAD (Silero) | High | Yes |
| Speaker embeddings | High | Yes (model is proven) |
| Clustering | **Low** | **No** — partition selection bug |
| GCC-PHAT TDOA | Medium | Yes (above 10 dB SNR) |
| ILD estimation | Medium | Yes |
| Azimuth fusion | Medium | Partially (ILD weight needs tuning) |
| Kalman tracking | Medium | Yes (above 15 dB SNR) |
| Movement detection | **Low** | **No** — false positives under noise |
| Crossing detection | Medium | Not tested with real crossings |
| Streamlit frontend | High | Yes |
| Config system | High | Yes |
| Visualization | High | Yes |

### What Works Reliably
- Full pipeline on clean stereo audio
- Speaker count detection at >= 10 dB SNR (with the partition bug fixed)
- L/R localization at >= 10 dB SNR
- GCC-PHAT defense layers (correctly reject bad frames)
- Denoising + VAD chain
- Frontend, visualization, output formatting

### What Is Unstable
- **Clustering** at < 10 dB SNR — partition selection bug causes collapse even when spectral clustering finds the correct split
- **Movement detection** at < 20 dB SNR — azimuth noise from ILD triggers false positives
- **Localization** at < 5 dB SNR — GCC-PHAT produces zero valid frames, ILD alone cannot compensate
- **Embedding separation** with synthetic audio — negative inter-intra gaps make all clustering metrics unreliable

### Blocking Issues for Production
1. **DEFECT-1** (partition selection) must be fixed before any noisy-audio deployment
2. Movement detection needs noise-adaptive thresholding
3. Pipeline must be tested with real multi-speaker recordings

### Honest Assessment
The pipeline architecture is sound and well-modularized. The clean-audio path works correctly. The noise robustness improvements (v0.3–v0.5) genuinely helped — the pipeline would be far worse without them. But there is one specific bug in clustering that undermines medium-noise performance, and the movement detection logic doesn't account for the noise floor of ILD-based azimuth. These are fixable without architectural changes. The deeper limitation is that below ~5 dB SNR, the physics of stereo TDOA estimation with a 20cm mic spacing break down, and no amount of signal processing can recover what isn't there.

---

*Report generated: 2026-02-18*
*Pipeline version: v0.5*
*Test audio: synthetic 2-speaker stereo, 48kHz, 22 seconds*
