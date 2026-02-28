"""
Tests for src/output_formatter.py — JSON result assembly and printing.
"""

import json
import numpy as np
import pytest

from src.output_formatter import format_output, print_summary


@pytest.fixture
def minimal_output():
    """Minimal valid output with 1 speaker and Stage 1 data only."""
    return format_output(
        num_speakers=1,
        segments=[(0.5, 1.5), (2.0, 3.0)],
        labels=np.array([0, 0]),
        speaker_dominant=None,
        speaker_position=None,
        tracking_results=None,
        duration=5.0,
        sr=16000,
    )


@pytest.fixture
def full_output():
    """Full output with 2 speakers and all 3 stages populated."""
    return format_output(
        num_speakers=2,
        segments=[(0.5, 1.5), (2.0, 3.0), (3.5, 4.5), (5.0, 6.0)],
        labels=np.array([0, 0, 1, 1]),
        speaker_dominant={0: -15.3, 1: 22.7},
        speaker_position={0: "left", 1: "right"},
        tracking_results={
            0: {
                "trajectory": [
                    {"time": 0.5, "azimuth": -14.0},
                    {"time": 1.0, "azimuth": -16.0},
                ],
                "movement_detected": False,
                "azimuth_range": 2.0,
                "crossings": [],
            },
            1: {
                "trajectory": [
                    {"time": 3.5, "azimuth": 21.0},
                    {"time": 4.0, "azimuth": 24.0},
                ],
                "movement_detected": True,
                "azimuth_range": 3.0,
                "crossings": [{"time": 4.2, "with_speaker": 0}],
            },
        },
        duration=7.0,
        sr=16000,
    )


class TestJSONStructure:
    """Test the overall JSON structure."""

    def test_output_is_dict(self, minimal_output):
        assert isinstance(minimal_output, dict)

    def test_output_is_json_serializable(self, minimal_output):
        json_str = json.dumps(minimal_output)
        assert len(json_str) > 0

    def test_full_output_is_json_serializable(self, full_output):
        json_str = json.dumps(full_output)
        assert len(json_str) > 0

    def test_speakers_is_list(self, minimal_output):
        assert isinstance(minimal_output["speakers"], list)

    def test_speaker_count_matches_list(self, full_output):
        assert len(full_output["speakers"]) == full_output["num_speakers"]

    def test_metadata_is_dict(self, minimal_output):
        assert isinstance(minimal_output["metadata"], dict)

    def test_zero_speakers_empty_list(self):
        result = format_output(
            num_speakers=0,
            segments=[],
            labels=None,
            speaker_dominant=None,
            speaker_position=None,
            tracking_results=None,
            duration=3.0,
            sr=16000,
        )
        assert result["num_speakers"] == 0
        assert result["speakers"] == []


class TestRequiredFields:
    """Test that all required fields are present."""

    def test_top_level_keys(self, minimal_output):
        assert "metadata" in minimal_output
        assert "num_speakers" in minimal_output
        assert "speakers" in minimal_output

    def test_metadata_keys(self, minimal_output):
        meta = minimal_output["metadata"]
        assert "timestamp" in meta
        assert "audio_duration_sec" in meta
        assert "sample_rate" in meta
        assert "total_speech_segments" in meta

    def test_metadata_values(self, minimal_output):
        meta = minimal_output["metadata"]
        assert meta["audio_duration_sec"] == 5.0
        assert meta["sample_rate"] == 16000
        assert meta["total_speech_segments"] == 2

    def test_speaker_basic_fields(self, minimal_output):
        speaker = minimal_output["speakers"][0]
        assert "id" in speaker
        assert "segments" in speaker
        assert "total_speech_time" in speaker

    def test_speaker_id_format(self, full_output):
        ids = [s["id"] for s in full_output["speakers"]]
        assert ids == ["S1", "S2"]

    def test_speaker_segments_have_timing(self, minimal_output):
        seg = minimal_output["speakers"][0]["segments"][0]
        assert "start" in seg
        assert "end" in seg
        assert "duration" in seg

    def test_full_output_has_spatial_fields(self, full_output):
        speaker = full_output["speakers"][0]
        assert "dominant_azimuth_deg" in speaker
        assert "dominant_position" in speaker

    def test_full_output_has_tracking_fields(self, full_output):
        speaker = full_output["speakers"][0]
        assert "trajectory" in speaker
        assert "movement_detected" in speaker
        assert "azimuth_range_deg" in speaker
        assert "crossings" in speaker

    def test_total_speech_time_correct(self, minimal_output):
        speaker = minimal_output["speakers"][0]
        expected = (1.5 - 0.5) + (3.0 - 2.0)
        assert abs(speaker["total_speech_time"] - expected) < 0.01


class TestPrintSummary:
    """Test that print_summary runs without crashing."""

    def test_print_summary_minimal(self, minimal_output, capsys):
        print_summary(minimal_output)
        captured = capsys.readouterr()
        assert "PIPELINE RESULTS SUMMARY" in captured.out

    def test_print_summary_full(self, full_output, capsys):
        print_summary(full_output)
        captured = capsys.readouterr()
        assert "PIPELINE RESULTS SUMMARY" in captured.out
        assert "S1" in captured.out
        assert "S2" in captured.out

    def test_print_summary_zero_speakers(self, capsys):
        result = format_output(
            num_speakers=0,
            segments=[],
            labels=None,
            speaker_dominant=None,
            speaker_position=None,
            tracking_results=None,
            duration=1.0,
            sr=16000,
        )
        print_summary(result)
        captured = capsys.readouterr()
        assert "Speakers detected:  0" in captured.out

    def test_print_summary_with_movement(self, full_output, capsys):
        print_summary(full_output)
        captured = capsys.readouterr()
        assert "Movement" in captured.out

    def test_print_summary_with_crossings(self, full_output, capsys):
        print_summary(full_output)
        captured = capsys.readouterr()
        assert "Crossing" in captured.out
