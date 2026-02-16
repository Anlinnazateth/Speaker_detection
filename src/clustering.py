"""
Module 5: Unsupervised Speaker Clustering
Groups speaker embeddings by identity without knowing the number of speakers.
Uses spectral clustering with eigen-gap heuristic for automatic speaker count,
with AHC as a fallback/validation.
"""

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score

from .config import PipelineConfig


def cluster_speakers(
    embeddings: np.ndarray, config: PipelineConfig = None
) -> dict:
    """
    Cluster speaker embeddings to determine speaker count and assign labels.

    Parameters
    ----------
    embeddings : np.ndarray
        (S, 192) matrix of L2-normalized speaker embeddings.
    config : PipelineConfig
        Pipeline configuration.

    Returns
    -------
    dict with keys:
        "labels"       : np.ndarray (S,) int cluster labels [0..K-1]
        "num_speakers" : int, auto-detected speaker count K
        "centroids"    : np.ndarray (K, 192) cluster centers
    """
    if config is None:
        config = PipelineConfig()

    n_segments = embeddings.shape[0]

    # ── Edge case: single segment → single speaker ──
    if n_segments == 1:
        return {
            "labels": np.array([0]),
            "num_speakers": 1,
            "centroids": embeddings.copy(),
        }

    # ── Build cosine similarity affinity matrix ──
    # A_ij = max(0, cosine_similarity(e_i, e_j))
    similarity = embeddings @ embeddings.T  # Cosine sim since embeddings are L2-normed
    affinity = np.maximum(similarity, 0)
    np.fill_diagonal(affinity, 1.0)

    # ── Determine optimal speaker count via eigen-gap heuristic ──
    k_optimal = _find_optimal_k(affinity, config.cluster_max_k, config.cluster_eigengap_min)

    # ── Edge case: only 2 segments, can't have more than 2 speakers ──
    k_optimal = min(k_optimal, n_segments)

    # ── Force-split fallback: if eigen-gap says k=1 but pairwise distances
    #    show clear separation, override to k=2 ──
    if k_optimal <= 1 and n_segments >= 2:
        cosine_dist = 1.0 - similarity  # cosine distance matrix
        np.fill_diagonal(cosine_dist, 0.0)
        max_dist = np.max(cosine_dist)
        if max_dist > config.cluster_force_split_dist:
            k_optimal = 2
            print(f"[Clustering] Force-split: max cosine distance {max_dist:.3f} "
                  f"> threshold {config.cluster_force_split_dist}, trying k=2")

    if k_optimal <= 1:
        # Single speaker — all segments belong to cluster 0
        labels = np.zeros(n_segments, dtype=int)
        centroids = np.mean(embeddings, axis=0, keepdims=True)
        return {
            "labels": labels,
            "num_speakers": 1,
            "centroids": centroids,
        }

    # ── Primary: Spectral Clustering ──
    spectral_labels = _spectral_cluster(affinity, k_optimal)

    # ── Fallback: AHC for validation ──
    ahc_labels = _ahc_cluster(embeddings, k_optimal, config.cluster_cosine_threshold)

    # ── Pick the partition with the higher silhouette score ──
    labels = _select_best_partition(
        embeddings, spectral_labels, ahc_labels
    )

    # ── Merge tiny clusters (< min_members) into nearest neighbor ──
    labels = _merge_small_clusters(embeddings, labels, config.cluster_min_members)

    # ── Compute cluster centroids ──
    num_speakers = len(np.unique(labels))
    centroids = np.array([
        embeddings[labels == k].mean(axis=0) for k in range(num_speakers)
    ])

    print(f"[Clustering] Detected {num_speakers} speaker(s) "
          f"from {n_segments} segments")

    return {
        "labels": labels,
        "num_speakers": num_speakers,
        "centroids": centroids,
    }


def _find_optimal_k(affinity: np.ndarray, max_k: int, eigengap_min: float = 0.02) -> int:
    """
    Determine optimal number of clusters using the eigen-gap heuristic
    on the normalized Laplacian of the affinity matrix.
    """
    n = affinity.shape[0]
    max_k = min(max_k, n)

    # Degree matrix and normalized Laplacian
    degree = np.sum(affinity, axis=1)
    degree_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0)
    D_inv_sqrt = np.diag(degree_inv_sqrt)
    L_norm = np.eye(n) - D_inv_sqrt @ affinity @ D_inv_sqrt

    # Compute eigenvalues (sorted ascending)
    eigenvalues = np.sort(np.real(np.linalg.eigvalsh(L_norm)))
    eigenvalues = eigenvalues[:max_k + 1]

    # Eigen-gap: find largest gap between consecutive eigenvalues
    # Start from k=2 (at least 2 speakers to be meaningful)
    if len(eigenvalues) < 3:
        return 1

    gaps = np.diff(eigenvalues[1:])  # gaps between lambda_2..lambda_max
    if len(gaps) == 0:
        return 1

    # k* = argmax gap, offset by 2 since we skip lambda_0 and lambda_1
    best_gap_idx = np.argmax(gaps)
    k_star = best_gap_idx + 2  # +2 because gap[0] is between lambda_1 and lambda_2

    # If the maximum gap is very small, assume single speaker
    if gaps[best_gap_idx] < eigengap_min:
        return 1

    return k_star


def _spectral_cluster(affinity: np.ndarray, n_clusters: int) -> np.ndarray:
    """Run spectral clustering on the precomputed affinity matrix."""
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42,
    )
    return sc.fit_predict(affinity)


def _ahc_cluster(
    embeddings: np.ndarray, n_clusters: int, threshold: float
) -> np.ndarray:
    """
    Run Agglomerative Hierarchical Clustering with cosine distance
    and average linkage.
    """
    ahc = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average",
    )
    return ahc.fit_predict(embeddings)


def _select_best_partition(
    embeddings: np.ndarray,
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> np.ndarray:
    """Pick the partition with the higher silhouette score."""
    n_unique_a = len(np.unique(labels_a))
    n_unique_b = len(np.unique(labels_b))

    # Silhouette requires at least 2 clusters
    score_a = (
        silhouette_score(embeddings, labels_a, metric="cosine")
        if n_unique_a >= 2
        else -1.0
    )
    score_b = (
        silhouette_score(embeddings, labels_b, metric="cosine")
        if n_unique_b >= 2
        else -1.0
    )

    return labels_a if score_a >= score_b else labels_b


def _merge_small_clusters(
    embeddings: np.ndarray, labels: np.ndarray, min_members: int
) -> np.ndarray:
    """Merge clusters with fewer than min_members into the nearest larger cluster."""
    labels = labels.copy()
    unique, counts = np.unique(labels, return_counts=True)

    small_clusters = unique[counts < min_members]
    large_clusters = unique[counts >= min_members]

    if len(large_clusters) == 0:
        # All clusters are small — just keep them
        return labels

    # Compute centroids of large clusters
    large_centroids = np.array([
        embeddings[labels == k].mean(axis=0) for k in large_clusters
    ])

    for small_k in small_clusters:
        small_centroid = embeddings[labels == small_k].mean(axis=0)
        # Find nearest large cluster by cosine similarity
        similarities = large_centroids @ small_centroid
        nearest_idx = np.argmax(similarities)
        labels[labels == small_k] = large_clusters[nearest_idx]

    # Re-label to consecutive integers starting from 0
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels])

    return labels
