"""Strict loader for ``assets/{truth_fixture_key}/ground_truth.json`` fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from delta_preservation.evaluation.contracts import GroundTruthContractError, GroundTruthPacket


def _default_repo_root() -> Path:
    """Return the repository root used for exact fixture resolution."""

    return Path(__file__).resolve().parents[2]


def _format_validation_error(error: dict[str, object]) -> str:
    """Render a compact Pydantic validation error for path-specific messages."""

    location = ".".join(str(part) for part in error.get("loc", ()))
    message = str(error.get("msg", "invalid data"))
    return f"{location}: {message}" if location else message


def load_ground_truth_packet(truth_fixture_key: str, repo_root: Path | None = None) -> GroundTruthPacket:
    """Load the exact ``assets/{truth_fixture_key}/ground_truth.json`` fixture."""

    repo_base = (repo_root or _default_repo_root()).resolve()

    # Normalize key: lowercase and strip spaces so "Part 1" resolves to "part1"
    normalized_key = truth_fixture_key.lower().replace(" ", "")
    fixture_dir = repo_base / "assets" / normalized_key
    truth_path = fixture_dir / "ground_truth.json"

    if not fixture_dir.is_dir():
        raise GroundTruthContractError(f"Ground truth fixture directory not found: {fixture_dir}")
    if not truth_path.is_file():
        raise GroundTruthContractError(f"Ground truth file not found: {truth_path}")

    try:
        raw_payload = json.loads(truth_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GroundTruthContractError(
            f"Ground truth JSON is malformed at {truth_path}: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})"
        ) from exc
    except OSError as exc:
        raise GroundTruthContractError(f"Ground truth file could not be read: {truth_path} ({exc})") from exc

    try:
        return GroundTruthPacket.model_validate(raw_payload)
    except ValidationError as exc:
        formatted = "; ".join(_format_validation_error(error) for error in exc.errors())
        raise GroundTruthContractError(f"Ground truth JSON failed validation: {truth_path} ({formatted})") from exc
