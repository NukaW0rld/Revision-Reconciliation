"""Tests for strict ground truth fixture loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from delta_preservation.evaluation import GroundTruthContractError, load_ground_truth_packet


def _write_ground_truth(repo_root: Path, fixture_key: str, payload: object) -> Path:
    fixture_dir = repo_root / "assets" / fixture_key
    fixture_dir.mkdir(parents=True, exist_ok=True)

    truth_path = fixture_dir / "ground_truth.json"
    if isinstance(payload, str):
        truth_path.write_text(payload, encoding="utf-8")
    else:
        truth_path.write_text(json.dumps(payload), encoding="utf-8")
    return truth_path


def _base_packet_payload(characteristics: list[dict[str, object]]) -> dict[str, object]:
    return {
        "part_name": "Fixture Part",
        "general_notes": "",
        "characteristics": characteristics,
    }


def test_missing_fixture_directory_raises_path_specific_error(tmp_path: Path) -> None:
    missing_dir = tmp_path / "assets" / "missing-fixture"

    with pytest.raises(GroundTruthContractError) as exc_info:
        load_ground_truth_packet("missing-fixture", repo_root=tmp_path)

    assert str(missing_dir) in str(exc_info.value)
    assert "fixture directory not found" in str(exc_info.value)


def test_missing_ground_truth_file_raises_path_specific_error(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "assets" / "part-x"
    fixture_dir.mkdir(parents=True)

    with pytest.raises(GroundTruthContractError) as exc_info:
        load_ground_truth_packet("part-x", repo_root=tmp_path)

    assert str(fixture_dir / "ground_truth.json") in str(exc_info.value)
    assert "file not found" in str(exc_info.value)


def test_malformed_json_raises_path_specific_error(tmp_path: Path) -> None:
    truth_path = _write_ground_truth(tmp_path, "part-y", "{not valid json")

    with pytest.raises(GroundTruthContractError) as exc_info:
        load_ground_truth_packet("part-y", repo_root=tmp_path)

    message = str(exc_info.value)
    assert str(truth_path) in message
    assert "malformed" in message
    assert "line 1" in message


def test_status_aware_nullable_fields_allow_removed_and_added_rows(tmp_path: Path) -> None:
    payload = _base_packet_payload(
        [
            {
                "char_no": 4,
                "classification": "removed",
                "requirement_revB": None,
                "snippet_center_revA": [84.0, 270.0],
                "snippet_center_revB": None,
            },
            {
                "classification": "added",
                "requirement_revB": "155",
                "snippet_center_revA": None,
                "snippet_center_revB": [154.0, 620.0],
            },
        ]
    )
    _write_ground_truth(tmp_path, "part-z", payload)

    packet = load_ground_truth_packet("part-z", repo_root=tmp_path)

    assert packet.characteristics[0].classification == "removed"
    assert packet.characteristics[0].requirement_revB is None
    assert packet.characteristics[1].classification == "added"
    assert packet.characteristics[1].char_no is None
    assert packet.characteristics[1].snippet_center_revA is None


def test_canonical_added_row_without_char_no_loads(tmp_path: Path) -> None:
    payload = _base_packet_payload(
        [
            {
                "classification": "added",
                "requirement_revB": "3X Ø18 ↧30",
                "snippet_center_revA": None,
                "snippet_center_revB": [242.0, 66.0],
            }
        ]
    )
    _write_ground_truth(tmp_path, "part-added", payload)

    packet = load_ground_truth_packet("part-added", repo_root=tmp_path)

    assert len(packet.characteristics) == 1
    assert packet.characteristics[0].char_no is None
    assert packet.characteristics[0].classification == "added"


def test_loader_normalizes_fixture_key_to_lowercase(tmp_path: Path) -> None:
    payload = _base_packet_payload(
        [
            {
                "char_no": 1,
                "classification": "unchanged",
                "requirement_revB": "120",
                "snippet_center_revA": [478.4, 717.3],
                "snippet_center_revB": [478.4, 731.5],
            }
        ]
    )
    _write_ground_truth(tmp_path, "part1", payload)

    # Loader normalizes to lowercase so "Part1" resolves to the "part1" directory.
    packet = load_ground_truth_packet("Part1", repo_root=tmp_path)
    assert packet.characteristics[0].char_no == 1
