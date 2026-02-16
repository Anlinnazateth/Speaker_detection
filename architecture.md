# Stereo Multi-Speaker Localization & Tracking System — Technical Architecture

## 1. System Overview

An offline, Python-based pipeline that takes a stereo `.wav` file of unknown speaker count and produces per-speaker identity labels, spatial positions (azimuth), and movement trajectories.

---

## 2. Block Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          STEREO .WAV FILE                               │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
                   ┌─────────────────────────┐
                   │     1. AUDIO LOADER      │
                   │  (load, validate, meta)  │
                   └────────────┬─────────────┘
                                │  PCM float32 [2, N]
                                ▼
                   ┌─────────────────────────┐
                   │ 2. STEREO PREPROCESSING  │
                   │  (norm, HPF, mono mix)   │
                   └────────────┬─────────────┘
                                │  L[N], R[N], mono[N]
                                ▼
                   ┌─────────────────────────┐
                   │  3. VOICE ACTIVITY       │
                   │     DETECTION (VAD)      │
                   └────────────┬─────────────┘
                                │  speech segments [(t_start, t_end), ...]
                                ▼
              ┌─────────────────┴──────────────────┐
              │                                    │
              ▼                                    ▼
┌──────────────────────────┐         ┌──────────────────────────┐
│ 4. SPEAKER EMBEDDING     │         │ 6. GCC-PHAT TDOA         │
│    EXTRACTION            │         │    ESTIMATION             │
│  (per-segment d-vectors) │         │  (per-frame delay est.)   │
└────────────┬─────────────┘         └────────────┬─────────────┘
             │  embeddings                        │  TDOA per frame
             ▼                                    ▼
┌──────────────────────────┐         ┌──────────────────────────┐
│ 5. UNSUPERVISED          │         │ 7. TDOA → AZIMUTH        │
│    CLUSTERING            │         │    CONVERSION             │
│  (auto speaker count)    │         │  (geometric mapping)      │
└────────────┬─────────────┘         └────────────┬─────────────┘
             │  cluster labels                    │  azimuth per frame
             └─────────────┬──────────────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ 8. SPEAKER-LOCATION     │
              │    ASSOCIATION          │
              │  (label ↔ azimuth)      │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ 9. KALMAN FILTER        │
              │    TRACKING             │
              │  (smooth trajectories)  │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ 10. OUTPUT FORMATTING   │
              │   (JSON report)         │
              └─────────────────────────┘
```

---

## 3. Module Specifications

---

### 3.1 Audio Loader

**Purpose:** Load a stereo WAV file into memory, validate format, extract metadata.

**Algorithm:** Direct PCM read via `soundfile` (wraps libsndfile).

**Mathematics:** None — raw I/O.

**Python Libraries:**
- `soundfile` (primary) or `scipy.io.wavfile`

**Input:**
- File path → `str`

**Output:**
```python
{
    "audio": np.ndarray,    # shape (N, 2), float32, range [-1, 1]
    "sr": int,              # sample rate (e.g. 16000, 44100, 48000)
    "duration": float,      # seconds
    "n_samples": int
}
```

**Processing Steps:**
1. Read file with `sf.read(path, dtype='float32')`.
2. Validate `channels == 2`; raise if mono or multi-channel.
3. Validate sample rate ≥ 16 kHz (warn if < 16 kHz).
4. Normalize peak to [-1, 1] if not already.

**Failure Cases:**
| Failure | Mitigation |
|---|---|
| Mono file | Abort with clear error |
| Compressed format (mp3) | Convert with `pydub`/`ffmpeg` first, or reject |
| Corrupt header | Catch `sf.SoundFileError` |
| Extremely long file (>1 hr) | Process in chunks or warn about memory |

**Complexity:** O(N) time, O(N) memory where N = total samples.

---

### 3.2 Stereo Preprocessing

**Purpose:** Condition the raw stereo signal for downstream analysis — normalize, filter, and produce channel-separated and mono-mixed signals.

**Algorithm:**

1. **DC offset removal** — subtract per-channel mean.
2. **High-pass filter** — 2nd-order Butterworth at 80 Hz to remove rumble and mechanical noise.
3. **RMS normalization** — scale each channel so RMS = −20 dBFS (reference level).
4. **Mono mix** — arithmetic mean of L and R for modules that operate on a single channel.

**Mathematics:**

DC removal:
```
x_L[n] = x_L[n] - mean(x_L)
```

Butterworth HPF transfer function (2nd order, cutoff ω_c):
```
H(s) = s² / (s² + √2·ω_c·s + ω_c²)
```

RMS normalization to target A (in linear amplitude):
```
x_norm[n] = x[n] · (A / RMS(x))
where RMS(x) = sqrt( (1/N) · Σ x[n]² )
```

Mono mix:
```
x_mono[n] = (x_L[n] + x_R[n]) / 2
```

**Python Libraries:**
- `scipy.signal` — `butter`, `sosfilt` (second-order sections for numerical stability)
- `numpy`

**Input:**
```python
audio: np.ndarray  # (N, 2) float32
sr: int
```

**Output:**
```python
{
    "left": np.ndarray,     # (N,) preprocessed left channel
    "right": np.ndarray,    # (N,) preprocessed right channel
    "mono": np.ndarray,     # (N,) mono downmix
    "sr": int
}
```

**Failure Cases:**
| Failure | Mitigation |
|---|---|
| Silent channel (RMS ≈ 0) | Skip normalization for that channel; flag warning |
| Clipped signal (many samples at ±1.0) | Log warning; proceed (may degrade TDOA) |
| Extremely low SNR | VAD downstream will produce few/no segments |

**Complexity:** O(N) time, O(N) memory.

---

### 3.3 Voice Activity Detection (VAD)

**Purpose:** Identify temporal regions containing speech, discarding silence and noise.

**Algorithm:** `silero-vad` — a compact ONNX-based neural VAD model. Chosen because:
- No training needed.
- Works on 16 kHz mono.
- Frame-level probabilities → easy thresholding.

Alternative (lighter): energy + ZCR thresholding, but far less robust.

**Mathematics:**

Silero outputs a speech probability p(t) ∈ [0, 1] per frame. We threshold:
```
speech(t) = 1  if p(t) ≥ τ     (τ = 0.5 default)
            0  otherwise
```

Then merge adjacent speech frames with gap ≤ g_max (e.g. 300 ms) and discard segments shorter than d_min (e.g. 250 ms).

**Python Libraries:**
- `torch` (Silero VAD ships as a PyTorch JIT model)
- Alternative: `webrtcvad` (lighter, C-based, less accurate)

**Input:**
```python
mono: np.ndarray   # (N,) 16 kHz float32
sr: int
```

**Output:**
```python
segments: list[tuple[float, float]]  # [(start_sec, end_sec), ...]
```

**Processing Steps:**
1. Resample to 16 kHz if needed (`torchaudio.transforms.Resample`).
2. Run Silero VAD with 512-sample window (32 ms at 16 kHz).
3. Threshold at τ = 0.5.
4. Merge segments separated by ≤ 300 ms.
5. Drop segments shorter than 250 ms.

**Failure Cases:**
| Failure | Mitigation |
|---|---|
| Music/noise misdetected as speech | Raise threshold τ to 0.7 |
| Overlapping speakers → continuous speech | Acceptable; clustering handles speaker separation |
| Very quiet speaker missed | Lower threshold or use adaptive energy gating |

**Complexity:** O(N / hop) forward passes; each pass is O(1) for the small RNN. Effectively O(N).

---

### 3.4 Speaker Embedding Extraction

**Purpose:** Convert each speech segment into a fixed-dimensional vector (embedding) that captures voice identity (timbre, pitch, formant structure) independent of content.

**Algorithm:** Pre-trained `speechbrain` ECAPA-TDNN model.

ECAPA-TDNN architecture highlights:
- 1D Squeeze-Excitation Res2Net blocks.
- Multi-layer feature aggregation.
- Channel- and context-dependent statistics pooling.
- Produces a 192-dim embedding per utterance.

**Mathematics:**

The model learns a mapping f_θ: ℝ^T → ℝ^d (d = 192) such that:
```
cos_sim(f_θ(x_i), f_θ(x_j)) ≈ 1   if same speaker
cos_sim(f_θ(x_i), f_θ(x_j)) ≈ 0   if different speakers
```

where cosine similarity:
```
cos_sim(a, b) = (a · b) / (‖a‖ · ‖b‖)
```

The model is trained with AAM-Softmax loss (additive angular margin):
```
L = -log( exp(s·cos(θ_y + m)) / (exp(s·cos(θ_y + m)) + Σ_{j≠y} exp(s·cos(θ_j))) )
```
where s = scale, m = margin, θ_y = angle to correct class.

**Python Libraries:**
- `speechbrain` — `EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")`
- `torch`, `torchaudio`

**Input:**
```python
mono: np.ndarray       # full mono signal
segments: list[tuple]  # VAD segments
sr: int
```

**Output:**
```python
embeddings: np.ndarray          # (S, 192) where S = number of segments
segment_times: list[tuple]      # corresponding (start, end) per embedding
```

**Processing Steps:**
1. For each VAD segment, extract the mono audio slice.
2. If segment > 10 s, split into 3 s sub-segments with 1.5 s overlap (the model is trained on short utterances).
3. Resample to 16 kHz if needed.
4. Pass through ECAPA-TDNN → 192-dim L2-normalized embedding.
5. Collect all embeddings into a matrix.

**Failure Cases:**
| Failure | Mitigation |
|---|---|
| Segment too short (< 0.5 s) | Pad with zeros or skip; short segments yield unreliable embeddings |
| Overlapping speakers in segment | Embedding will be a blend; downstream clustering may create a "mixed" cluster — post-process by splitting |
| Non-speech captured by VAD | Embedding will be noise-like; outlier detection in clustering removes it |

**Complexity:** O(S · T_seg) where T_seg is average segment length. GPU recommended for > 100 segments.

---

### 3.5 Unsupervised Clustering (Automatic Speaker Count)

**Purpose:** Group embeddings by speaker identity without knowing the number of speakers in advance.

**Algorithm:** Agglomerative Hierarchical Clustering (AHC) with automatic threshold OR Spectral Clustering with eigen-gap heuristic.

**Recommended approach — two-stage:**

1. **Primary: Spectral Clustering with eigen-gap auto-k**
2. **Fallback/validation: AHC with cosine distance**

**Mathematics:**

**Affinity matrix** from cosine similarity:
```
A_ij = max(0, cos_sim(e_i, e_j))
```

**Laplacian:**
```
D = diag(Σ_j A_ij)
L_norm = I - D^{-1/2} · A · D^{-1/2}
```

**Eigen-gap heuristic** for automatic k:
```
k* = argmax_k (λ_{k+1} - λ_k)     for k ∈ [2, k_max]
```
where λ_1 ≤ λ_2 ≤ ... are eigenvalues of L_norm sorted ascending.

If the maximum eigen-gap is below a threshold (i.e. no clear cluster structure), fall back to k = 1 (single speaker).

**AHC alternative:**
- Distance metric: `1 - cos_sim(e_i, e_j)`.
- Linkage: average.
- Threshold: cosine distance = 0.3 (tunable; corresponds to cos_sim = 0.7).

**Python Libraries:**
- `scikit-learn` — `SpectralClustering`, `AgglomerativeClustering`
- `scipy.cluster.hierarchy` — for dendrogram analysis
- `numpy`, `scipy.spatial.distance`

**Input:**
```python
embeddings: np.ndarray  # (S, 192)
```

**Output:**
```python
{
    "labels": np.ndarray,    # (S,) int cluster labels [0, 1, ..., K-1]
    "num_speakers": int,     # K (auto-detected)
    "centroids": np.ndarray  # (K, 192) cluster centers
}
```

**Processing Steps:**
1. Compute pairwise cosine similarity matrix A (S × S).
2. Run spectral clustering:
   a. Compute normalized Laplacian.
   b. Find eigenvalues λ_1 ... λ_min(S, 20).
   c. Select k* via eigen-gap.
   d. Run k-means on top-k eigenvectors.
3. Validate: if any cluster has < 2 members, merge with nearest.
4. Optional: re-run AHC and compare silhouette scores; pick the better partition.

**Failure Cases:**
| Failure | Mitigation |
|---|---|
| Only 1 speaker | Eigen-gap finds no gap > threshold → output k=1 |
| Speakers with very similar voices | Cosine threshold may under-segment; use stricter threshold or add spatial features |
| Too many very short segments | Noisy embeddings → inflate apparent k; filter short segments earlier |
| > 10 speakers | Rare in stereo; spectral clustering scales as O(S³) — switch to mini-batch k-means with BIC |

**Complexity:** O(S² · d) for affinity, O(S³) for eigendecomposition. Fine for S < 5000 segments.

---

### 3.6 GCC-PHAT TDOA Estimation

**Purpose:** Estimate the Time Difference of Arrival (TDOA) between the left and right channels for each short frame, which encodes the spatial direction of the dominant sound source.

**Algorithm:** Generalized Cross-Correlation with Phase Transform (GCC-PHAT).

**Mathematics:**

Given left signal x_L(t) and right signal x_R(t), for a frame of length W:

1. **Cross-power spectrum:**
```
G_LR(f) = X_L(f) · X_R*(f)
```
where X_L(f) = FFT(x_L), X_R*(f) = conjugate of FFT(x_R).

2. **PHAT weighting** (whitening):
```
Ψ_PHAT(f) = G_LR(f) / |G_LR(f)|
```

3. **GCC-PHAT correlation:**
```
R_PHAT(τ) = IFFT(Ψ_PHAT(f))
```

4. **TDOA estimate:**
```
τ* = argmax_τ R_PHAT(τ)
```

Constrained to |τ*| ≤ τ_max = d / c, where d = microphone spacing, c = 343 m/s.

**Sub-sample precision** via parabolic interpolation around the peak:
```
τ_refined = τ* + 0.5 · (R[τ*-1] - R[τ*+1]) / (R[τ*-1] - 2·R[τ*] + R[τ*+1])
```

**Python Libraries:**
- `numpy.fft` — `rfft`, `irfft`
- `scipy.signal` — windowing

**Input:**
```python
left: np.ndarray     # (N,) preprocessed left channel
right: np.ndarray    # (N,) preprocessed right channel
sr: int
frame_len: float     # frame duration in seconds (e.g. 0.04 = 40 ms)
hop_len: float       # hop in seconds (e.g. 0.02 = 20 ms)
```

**Output:**
```python
{
    "tdoa": np.ndarray,        # (F,) TDOA in seconds per frame
    "gcc_peak": np.ndarray,    # (F,) peak correlation value (confidence)
    "frame_times": np.ndarray  # (F,) center time of each frame
}
```

**Processing Steps:**
1. Segment L and R into overlapping frames (Hann window, 40 ms, 50% overlap).
2. For each frame:
   a. Compute FFTs of L and R.
   b. Compute GCC-PHAT.
   c. Find peak within ±τ_max.
   d. Refine with parabolic interpolation.
   e. Record TDOA and peak magnitude.
3. Output TDOA time series.

**Design Choices:**
- Frame length 40 ms: long enough for frequency resolution, short enough for temporal resolution.
- PHAT weighting preferred over standard GCC because it sharpens the correlation peak, improving robustness in reverberant environments.
- Peak magnitude serves as a confidence score: low peaks indicate diffuse noise or overlapping sources.

**Failure Cases:**
| Failure | Mitigation |
|---|---|
| Reverberant room → spurious peaks | Use peak magnitude threshold; discard frames with gcc_peak < 0.15 |
| Two speakers simultaneously → ambiguous TDOA | Peak is biased toward louder source; multi-peak detection possible but complex |
| Silence frames → random TDOA | VAD mask: only compute TDOA on speech frames |
| Correlated noise (e.g. fan) in both channels | HPF in preprocessing helps; residual handled by confidence threshold |

**Complexity:** O(F · W · log W) due to FFT per frame. Very fast in practice.

---

### 3.7 TDOA → Azimuth Conversion

**Purpose:** Map the estimated TDOA (in seconds) to a horizontal angle (azimuth in degrees).

**Algorithm:** Far-field plane-wave assumption with stereo microphone pair.

**Mathematics:**

For two microphones separated by distance d, a plane wave arriving at angle θ from broadside:
```
τ = (d / c) · sin(θ)
```

Solving for θ:
```
θ = arcsin( (τ · c) / d )
```

where:
- c = 343 m/s (speed of sound at 20°C)
- d = microphone spacing (meters)
- θ = azimuth: 0° = center, −90° = full left, +90° = full right

**Important: Microphone spacing parameter `d`.**

For typical stereo recordings, the "effective" microphone spacing is not always known. Options:
1. **Known spacing** (e.g. binaural dummy head: d ≈ 0.18 m) — use directly.
2. **Unknown spacing** — estimate d from the maximum observed |τ| across the file:
   ```
   d_est = max(|τ|) · c
   ```
   This assumes at least one source is near ±90°. If not, underestimates d → compressed azimuth range. Acceptable for relative positioning.
3. **Default assumption** — d = 0.20 m (reasonable for spaced pair or ORTF-like setups).

**Python Libraries:**
- `numpy` — `arcsin`, `degrees`

**Input:**
```python
tdoa: np.ndarray       # (F,) TDOA in seconds
mic_distance: float    # d in meters (or estimated)
speed_of_sound: float  # default 343.0 m/s
```

**Output:**
```python
azimuth: np.ndarray    # (F,) azimuth in degrees [-90, +90]
```

**Processing Steps:**
1. Compute `sin_theta = (tdoa * c) / d`.
2. Clip to [-1, 1] to avoid domain errors from noise.
3. `theta = np.degrees(np.arcsin(sin_theta))`.

**Failure Cases:**
| Failure | Mitigation |
|---|---|
| |τ · c / d| > 1 → arcsin undefined | Clip to [-1, 1]; these are noise outliers |
| Unknown mic spacing → incorrect absolute angles | Normalize to relative scale; document assumption |
| Near-field source (close to mics) → plane-wave model inaccurate | Accept as limitation; still gives directional trend |

**Complexity:** O(F) — trivial.

---

### 3.8 Speaker-Location Association

**Purpose:** Link each speaker identity (from clustering) with their spatial trajectory (from TDOA/azimuth), producing per-speaker azimuth time series.

**Algorithm:** Direct mapping via temporal alignment.

**Mathematics:**

Each speech segment s_i has:
- A speaker label `k_i ∈ {0, 1, ..., K-1}` (from clustering).
- A time interval `[t_start_i, t_end_i]`.

Each TDOA frame f_j has:
- A time `t_j`.
- An azimuth `θ_j`.

Association rule:
```
For segment s_i with label k_i:
    azimuth_observations[k_i] += { (t_j, θ_j) : t_start_i ≤ t_j ≤ t_end_i }
```

**Dominant position** per speaker:
```
θ_dominant(k) = median( { θ_j : (t_j, θ_j) ∈ azimuth_observations[k] } )
```

Using median (not mean) for robustness against outlier frames.

**Spatial label:**
```
position(k) = "left"    if θ_dominant(k) < -5°
               "right"   if θ_dominant(k) > +5°
               "center"  otherwise
```

**Python Libraries:**
- `numpy` — median, indexing

**Input:**
```python
segments: list[tuple[float, float]]   # VAD segment times
labels: np.ndarray                     # (S,) cluster labels
frame_times: np.ndarray               # (F,) TDOA frame center times
azimuth: np.ndarray                    # (F,) azimuth per frame
gcc_peak: np.ndarray                   # (F,) confidence per frame
```

**Output:**
```python
speaker_tracks: dict[int, list[tuple[float, float]]]
# { speaker_id: [(time, azimuth), ...] }

speaker_dominant: dict[int, float]
# { speaker_id: dominant_azimuth_degrees }
```

**Processing Steps:**
1. For each segment, find overlapping TDOA frames.
2. Weight by GCC-PHAT peak confidence: discard frames with peak < 0.15.
3. Assign surviving (time, azimuth) pairs to the segment's speaker label.
4. Compute median azimuth per speaker for dominant position.
5. Classify as left/center/right.

**Failure Cases:**
| Failure | Mitigation |
|---|---|
| Overlapping speech → TDOA points to louder speaker | Accept bias; flag low-confidence frames |
| Speaker with very few segments → unreliable position | Require minimum 3 segments for valid localization |
| Speakers at same position → no spatial discrimination | Rely on voice embeddings only; flag identical positions |

**Complexity:** O(S · F/S) = O(F) overall.

---

### 3.9 Kalman Filter Tracking

**Purpose:** Smooth the noisy per-frame azimuth observations into a clean trajectory per speaker and detect movement vs. stationarity.

**Algorithm:** 1D Kalman filter on azimuth per speaker.

**State Model:**

State vector (per speaker):
```
x = [θ, θ̇]ᵀ
```
where θ = azimuth (degrees), θ̇ = azimuth rate of change (degrees/sec).

**Prediction step** (constant velocity model):
```
x_{k|k-1} = F · x_{k-1|k-1}

F = [[1, Δt],
     [0,  1]]

P_{k|k-1} = F · P_{k-1|k-1} · Fᵀ + Q
```

Process noise:
```
Q = q · [[Δt³/3, Δt²/2],
         [Δt²/2, Δt   ]]
```
where q = process noise intensity (tunable; start with q = 1.0 deg²/s³).

**Update step:**
```
H = [1, 0]    (we observe θ directly)
y_k = z_k - H · x_{k|k-1}         (innovation)
S_k = H · P_{k|k-1} · Hᵀ + R      (innovation covariance)
K_k = P_{k|k-1} · Hᵀ · S_k⁻¹      (Kalman gain)
x_{k|k} = x_{k|k-1} + K_k · y_k
P_{k|k} = (I - K_k · H) · P_{k|k-1}
```

Measurement noise R: derived from GCC-PHAT confidence. High confidence → low R; low confidence → high R:
```
R_k = R_base / max(gcc_peak_k, 0.01)
```
with R_base ≈ 25 deg².

**Movement detection:**

After filtering, compute the range of the smoothed trajectory:
```
range(k) = max(θ_smoothed[k]) - min(θ_smoothed[k])

movement_detected(k) = True  if range(k) > δ   (δ = 10° default)
```

**Python Libraries:**
- `filterpy` — `KalmanFilter` (or manual implementation with `numpy`)
- `numpy`

**Input:**
```python
speaker_tracks: dict[int, list[tuple[float, float]]]
# { speaker_id: [(time, azimuth), ...] }
```

**Output:**
```python
{
    speaker_id: {
        "trajectory": list[dict],       # [{"time": t, "azimuth": θ}, ...]
        "movement_detected": bool,
        "azimuth_range": float,         # degrees
        "smoothed_positions": np.ndarray
    }
}
```

**Processing Steps:**
1. For each speaker, sort observations by time.
2. Initialize Kalman filter at first observation.
3. For each subsequent observation:
   a. Predict to current time.
   b. Update with observation (adaptive R based on confidence).
4. Extract smoothed trajectory.
5. Compute range → movement flag.
6. Optionally resample to uniform time grid for output.

**Failure Cases:**
| Failure | Mitigation |
|---|---|
| Very sparse observations → large gaps | Increase process noise q during gaps; or skip prediction and re-initialize |
| Outlier TDOA observation | Kalman naturally down-weights if R is set high; additionally gate innovations > 3σ |
| Speaker silent for long period then reappears | Re-initialize filter if gap > 5 s |

**Complexity:** O(M) per speaker where M = number of observations. Negligible.

---

### 3.10 Output Formatting

**Purpose:** Assemble all results into the specified JSON structure.

**Processing Steps:**
1. For each speaker k ∈ {0, ..., K-1}:
   a. Assign human-readable ID: `"S1"`, `"S2"`, etc.
   b. Include `dominant_position` (left/right/center).
   c. Include `trajectory` — list of `{"time": t, "azimuth": θ}` dicts.
   d. Include `movement_detected` flag.
2. Wrap in top-level `{"num_speakers": K, "speakers": [...]}`.
3. Serialize to JSON with `json.dump()`.

**Output Format:**
```json
{
  "num_speakers": 2,
  "speakers": [
    {
      "id": "S1",
      "dominant_position": "left",
      "trajectory": [
        {"time": 0.1, "azimuth": -35.2},
        {"time": 0.5, "azimuth": -30.1}
      ],
      "movement_detected": true,
      "azimuth_range_deg": 5.1
    },
    {
      "id": "S2",
      "dominant_position": "right",
      "trajectory": [
        {"time": 0.2, "azimuth": 40.5},
        {"time": 0.6, "azimuth": 41.0}
      ],
      "movement_detected": false,
      "azimuth_range_deg": 0.5
    }
  ]
}
```

---

## 4. End-to-End Pipeline Summary

```
                         Input: stereo.wav
                              │
                 ┌────────────┴────────────┐
                 │        Audio Loader      │
                 └────────────┬────────────┘
                              │
                 ┌────────────┴────────────┐
                 │   Stereo Preprocessing   │
                 └──┬─────────────────┬────┘
                    │                 │
            ┌───────┴──────┐  ┌──────┴────────┐
            │  mono signal │  │   L, R signals │
            └───────┬──────┘  └──────┬────────┘
                    │                │
            ┌───────┴──────┐  ┌──────┴────────┐
            │     VAD      │  │   GCC-PHAT    │
            └───────┬──────┘  └──────┬────────┘
                    │                │
            ┌───────┴──────┐  ┌──────┴────────┐
            │  Embeddings  │  │  TDOA→Azimuth │
            └───────┬──────┘  └──────┬────────┘
                    │                │
            ┌───────┴──────┐        │
            │  Clustering  │        │
            └───────┬──────┘        │
                    │                │
                    └────────┬───────┘
                             │
                 ┌───────────┴────────────┐
                 │  Speaker-Location      │
                 │  Association           │
                 └───────────┬────────────┘
                             │
                 ┌───────────┴────────────┐
                 │  Kalman Tracking       │
                 └───────────┬────────────┘
                             │
                 ┌───────────┴────────────┐
                 │  JSON Output           │
                 └────────────────────────┘
```

## 5. Dependency Summary

| Package | Purpose | Version |
|---|---|---|
| `numpy` | Array ops, FFT | ≥ 1.24 |
| `scipy` | Filtering, clustering, signal processing | ≥ 1.11 |
| `soundfile` | WAV I/O | ≥ 0.12 |
| `torch` | Neural model inference | ≥ 2.0 |
| `torchaudio` | Audio transforms | ≥ 2.0 |
| `speechbrain` | ECAPA-TDNN speaker embeddings | ≥ 1.0 |
| `scikit-learn` | Spectral/agglomerative clustering, metrics | ≥ 1.3 |
| `filterpy` | Kalman filter (optional; can implement manually) | ≥ 1.4 |

## 6. Configuration Parameters

| Parameter | Default | Module | Description |
|---|---|---|---|
| `vad_threshold` | 0.5 | VAD | Speech probability threshold |
| `vad_min_duration` | 0.25 s | VAD | Minimum speech segment duration |
| `vad_merge_gap` | 0.30 s | VAD | Maximum gap to merge adjacent segments |
| `embedding_segment_max` | 10.0 s | Embedding | Max segment length before splitting |
| `embedding_sub_len` | 3.0 s | Embedding | Sub-segment length for long segments |
| `embedding_sub_overlap` | 1.5 s | Embedding | Sub-segment overlap |
| `cluster_max_k` | 10 | Clustering | Maximum number of speakers to consider |
| `cluster_cosine_threshold` | 0.3 | Clustering | AHC distance threshold |
| `gcc_frame_len` | 0.04 s | GCC-PHAT | Frame length for TDOA |
| `gcc_hop_len` | 0.02 s | GCC-PHAT | Hop length for TDOA |
| `gcc_confidence_min` | 0.15 | GCC-PHAT | Minimum peak for valid TDOA |
| `mic_distance` | 0.20 m | Azimuth | Assumed microphone spacing |
| `speed_of_sound` | 343.0 m/s | Azimuth | Speed of sound |
| `kalman_q` | 1.0 | Kalman | Process noise intensity |
| `kalman_r_base` | 25.0 | Kalman | Base measurement noise |
| `movement_threshold` | 10.0° | Kalman | Azimuth range to flag movement |

## 7. Known Limitations & Future Work

1. **Stereo-only constraint** — azimuth only in horizontal plane; no elevation. Ambiguity between front and back (cone of confusion).
2. **Overlapping speech** — GCC-PHAT produces a single TDOA per frame. During overlap, it biases toward the louder source. Multi-source TDOA (e.g., MUSIC, SRP-PHAT) could improve this but requires more channels.
3. **Reverberation** — high RT60 environments degrade GCC-PHAT peaks. De-reverberation (WPE) could be added as a preprocessing step.
4. **Near-field sources** — the plane-wave model is approximate for sources close to the microphone pair.
5. **Speaker count upper bound** — spectral clustering with stereo data realistically supports 2–4 speakers. Beyond that, spatial discrimination collapses.
6. **No overlap-aware diarization** — current pipeline assigns each segment to exactly one speaker. Future: integrate overlap detection (e.g., pyannote overlap detector) and multi-label assignment.
