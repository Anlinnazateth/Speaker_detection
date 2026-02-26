"""
Tests for src/clustering.py — unsupervised speaker clustering.
"""

import numpy as np
import pytest

from src.clustering import cluster_speakers
from src.config import PipelineConfig


@pytest.fixture
def two_cluster_embeddings():
    """
    Generate 20 L2-normalized embeddings that form 2 well-separated clusters
    in 192-D space.
    """
    rng = np.random.RandomState(42)
    dim = 192
    n_per_cluster = 10

    base_a = np.zeros(dim, dtype=np.float32)
    base_a[0] = 1.0
    cluster_a = np.tile(base_a, (n_per_cluster, 1)) + rng.randn(n_per_cluster, dim).astype(np.float32) * 0.05

    base_b = np.zeros(dim, dtype=np.float32)
    base_b[1] = 1.0
    cluster_b = np.tile(base_b, (n_per_cluster, 1)) + rng.randn(n_per_cluster, dim).astype(np.float32) * 0.05

    embeddings = np.vstack([cluster_a, cluster_b])

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-10)

    return embeddings


@pytest.fixture
def single_cluster_embeddings():
    """Generate 10 L2-normalized embeddings that all cluster together."""
    rng = np.random.RandomState(123)
    dim = 192
    n = 10

    base = np.zeros(dim, dtype=np.float32)
    base[0] = 1.0
    embeddings = np.tile(base, (n, 1)) + rng.randn(n, dim).astype(np.float32) * 0.02

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-10)

    return embeddings


class TestTwoClusterCase:
    """Test clustering with two well-separated speaker groups."""

    def test_detects_two_speakers(self, two_cluster_embeddings):
        result = cluster_speakers(two_cluster_embeddings)
        assert result["num_speakers"] == 2

    def test_labels_have_correct_length(self, two_cluster_embeddings):
        result = cluster_speakers(two_cluster_embeddings)
        assert len(result["labels"]) == len(two_cluster_embeddings)

    def test_labels_are_zero_and_one(self, two_cluster_embeddings):
        result = cluster_speakers(two_cluster_embeddings)
        unique_labels = set(result["labels"])
        assert unique_labels == {0, 1}

    def test_centroids_shape(self, two_cluster_embeddings):
        result = cluster_speakers(two_cluster_embeddings)
        assert result["centroids"].shape == (2, 192)

    def test_clusters_are_consistent(self, two_cluster_embeddings):
        """The first 10 and second 10 embeddings should get different labels."""
        result = cluster_speakers(two_cluster_embeddings)
        labels = result["labels"]
        first_half = set(labels[:10])
        second_half = set(labels[10:])
        assert len(first_half) <= 2
        assert len(second_half) <= 2


class TestSingleSpeaker:
    """Test that tightly grouped embeddings are detected as one speaker."""

    def test_detects_one_speaker(self, single_cluster_embeddings):
        config = PipelineConfig(cluster_force_split_dist=0.9)
        result = cluster_speakers(single_cluster_embeddings, config=config)
        assert result["num_speakers"] == 1

    def test_all_labels_zero(self, single_cluster_embeddings):
        config = PipelineConfig(cluster_force_split_dist=0.9)
        result = cluster_speakers(single_cluster_embeddings, config=config)
        assert np.all(result["labels"] == 0)

    def test_single_centroid(self, single_cluster_embeddings):
        config = PipelineConfig(cluster_force_split_dist=0.9)
        result = cluster_speakers(single_cluster_embeddings, config=config)
        assert result["centroids"].shape[0] == 1


class TestMinimumClusterSize:
    """Test minimum cluster member enforcement."""

    def test_single_segment_returns_one_speaker(self):
        embedding = np.random.randn(1, 192).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        embedding = embedding.reshape(1, 192)

        result = cluster_speakers(embedding)
        assert result["num_speakers"] == 1
        assert result["labels"][0] == 0

    def test_two_segments_max_two_speakers(self):
        emb_a = np.zeros(192, dtype=np.float32)
        emb_a[0] = 1.0
        emb_b = np.zeros(192, dtype=np.float32)
        emb_b[1] = 1.0
        embeddings = np.vstack([emb_a, emb_b])

        result = cluster_speakers(embeddings)
        assert result["num_speakers"] <= 2

    def test_small_cluster_merged(self, two_cluster_embeddings):
        """With min_members set very high, small clusters should be merged."""
        config = PipelineConfig(cluster_min_members=15)
        result = cluster_speakers(two_cluster_embeddings, config=config)
        unique = np.unique(result["labels"])
        assert len(unique) == result["num_speakers"]
        assert all(l >= 0 for l in unique)
