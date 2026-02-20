"""
Module 4: Speaker Embedding Extraction
Converts each speech segment into a 192-dim ECAPA-TDNN embedding vector
that captures voice identity independent of spoken content.
"""

import numpy as np
import torch
import torchaudio

# Compatibility patch: torchaudio 2.10+ removed list_audio_backends(),
# but SpeechBrain 1.0.x still references it at import time.
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["default"]

from speechbrain.inference.speaker import EncoderClassifier

from .config import PipelineConfig


# Module-level cache to avoid reloading the model on every call
_model_cache: dict = {}


def _get_embedding_model(config: PipelineConfig) -> EncoderClassifier:
    """Load and cache the ECAPA-TDNN speaker embedding model."""
    if "model" not in _model_cache:
        print(f"[Embeddings] Loading model: {config.embedding_model}")
        _model_cache["model"] = EncoderClassifier.from_hparams(
            source=config.embedding_model,
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )
    return _model_cache["model"]


def extract_embeddings(
    mono: np.ndarray,
    sr: int,
    segments: list[tuple[float, float]],
    config: PipelineConfig = None,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """
    Extract speaker embeddings for each VAD segment.

    Long segments (> embedding_segment_max) are split into overlapping
    sub-segments for more reliable embeddings.

    Parameters
    ----------
    mono : np.ndarray
        (N,) mono audio signal, float32.
    sr : int
        Sample rate in Hz.
    segments : list of (start_sec, end_sec)
        VAD speech segments.
    config : PipelineConfig
        Pipeline configuration.

    Returns
    -------
    embeddings : np.ndarray
        (S, 192) matrix of L2-normalized speaker embeddings.
    segment_times : list of (start_sec, end_sec)
        Corresponding time interval for each embedding row.
    """
    if config is None:
        config = PipelineConfig()

    # ── Input contract assertions ──
    assert mono.ndim == 1, f"[Embeddings] Expected 1-D mono, got {mono.shape}"
    assert mono.dtype == np.float32, f"[Embeddings] Expected float32, got {mono.dtype}"

    model = _get_embedding_model(config)

    all_embeddings = []
    all_times = []

    for seg_start, seg_end in segments:
        # ── Split long segments into sub-segments ──
        sub_segments = _split_segment(
            seg_start, seg_end,
            max_len=config.embedding_segment_max,
            sub_len=config.embedding_sub_len,
            overlap=config.embedding_sub_overlap,
        )

        for sub_start, sub_end in sub_segments:
            duration = sub_end - sub_start

            # Skip segments too short for a reliable embedding
            if duration < config.embedding_min_duration:
                continue

            # ── Extract audio slice ──
            start_sample = int(sub_start * sr)
            end_sample = int(sub_end * sr)
            audio_slice = mono[start_sample:end_sample].copy()

            # ── Per-slice peak normalization (stabilizes embeddings in noisy audio) ──
            peak = np.max(np.abs(audio_slice))
            if peak > 1e-6:
                audio_slice = audio_slice / peak

            # ── Resample to 16 kHz if needed ──
            waveform = torch.from_numpy(audio_slice).unsqueeze(0)  # (1, T)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)

            # ── Extract embedding (192-dim, L2-normalized) ──
            with torch.no_grad():
                embedding = model.encode_batch(waveform)  # (1, 1, 192)
                embedding = embedding.squeeze().cpu().numpy()  # (192,)

            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            all_embeddings.append(embedding)
            all_times.append((sub_start, sub_end))

    if not all_embeddings:
        raise ValueError(
            "No valid speech segments for embedding extraction. "
            "Check VAD settings or audio quality."
        )

    embeddings = np.stack(all_embeddings, axis=0)  # (S, 192)
    print(f"[Embeddings] Extracted {embeddings.shape[0]} embeddings "
          f"(dim={embeddings.shape[1]})")

    return embeddings, all_times


def _split_segment(
    start: float,
    end: float,
    max_len: float,
    sub_len: float,
    overlap: float,
) -> list[tuple[float, float]]:
    """
    Split a segment into sub-segments if it exceeds max_len.
    Short segments are returned as-is.
    """
    duration = end - start
    if duration <= max_len:
        return [(start, end)]

    # Sliding window with overlap
    sub_segments = []
    hop = sub_len - overlap
    t = start
    while t < end:
        sub_end = min(t + sub_len, end)
        sub_segments.append((t, sub_end))
        t += hop
        if sub_end >= end:
            break

    return sub_segments
