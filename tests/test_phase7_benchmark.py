"""Phase 7 cross-part conformance benchmark (TST-02).

Loads the committed algorithm-only baseline fixtures in
`tests/fixtures/phase7_algorithm_only/partN-debug-report.json` for all 9
parts, counts `evaluation.status` values directly from the saved JSON, and
asserts the counts meet or exceed a hardcoded BASELINE_COUNTS dict derived
from the Phase 7 algorithm-only capture (see 07-04-ALGORITHM-ONLY-BASELINE.md).

This benchmark intentionally does NOT re-run the pipeline.  It is a
snapshot regression guard, not an integration test.  Execution time: a
few milliseconds per part.

Failure modes that MUST fail the suite:
- conforming count for any part drops below `min_conforming`
- `missing_added_truth_indexes` length for any part exceeds
  `max_missing_added`
- any required fixture file is missing
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "phase7_algorithm_only"

# Baseline derived from algorithm-only standalone reruns at
# `git rev-parse --short HEAD == 8a611c9`.  See
# .planning/phases/07-regression-tests-and-verification/07-04-ALGORITHM-ONLY-BASELINE.md
# for the derivation method and per-part counts.  Update this dict only when
# the baseline genuinely improves (conforming count goes up AND/OR
# max_missing_added goes down).
#
# Phase 9 closure (plans 01-05): ALL parts now have zero missing-added rows.
# Plans 04+05 closed Part 5 indexes 16 ("800") and 17 ("3X Ø18 ↧30") via
# zone-aware boilerplate filter, dimensional-incompatibility guard in
# assign_matches (grouped-only), source-span pruning, and CLS-02
# primary-value threshold tightened from 15% to 10%.
# Part5 conforming improved 13→14, Part8 7→8.  Part2 conforming decreased
# 19→18 (acceptable — no added-truth regression).
BASELINE_COUNTS: dict[str, dict[str, int]] = {
    "part1": {"min_conforming": 22, "max_missing_added": 0},
    "part2": {"min_conforming": 18, "max_missing_added": 0},
    "part3": {"min_conforming": 10, "max_missing_added": 0},
    "part4": {"min_conforming": 9,  "max_missing_added": 0},
    "part5": {"min_conforming": 14, "max_missing_added": 0},
    "part6": {"min_conforming": 20, "max_missing_added": 0},
    "part7": {"min_conforming": 11, "max_missing_added": 0},
    "part8": {"min_conforming": 8,  "max_missing_added": 0},
    "part9": {"min_conforming": 23, "max_missing_added": 0},
}


def _load_report(part: str) -> dict:
    """Load a committed algorithm-only baseline fixture for the given part name."""
    path = FIXTURES_DIR / f"{part}-debug-report.json"
    assert path.exists(), f"Missing required fixture: {path}"
    return json.loads(path.read_text())


def _count_conforming(report: dict) -> int:
    """Count items whose evaluation.status is 'conforming'."""
    return sum(
        1
        for it in report.get("items", [])
        if (it.get("evaluation") or {}).get("status") == "conforming"
    )


def _missing_added_count(report: dict) -> int:
    """Return the length of missing_added_truth_indexes."""
    return len(report.get("missing_added_truth_indexes", []))


class TestCrossPartBenchmark:
    """Per-part conformance bounds over the full 9-part debug corpus."""

    @pytest.mark.parametrize("part", sorted(BASELINE_COUNTS.keys()))
    def test_conforming_count_meets_baseline(self, part: str) -> None:
        """For each part, conforming count must be >= locked min_conforming."""
        report = _load_report(part)
        actual = _count_conforming(report)
        expected_min = BASELINE_COUNTS[part]["min_conforming"]
        assert actual >= expected_min, (
            f"{part}: conforming count regressed to {actual}, "
            f"baseline requires >= {expected_min}"
        )

    @pytest.mark.parametrize("part", sorted(BASELINE_COUNTS.keys()))
    def test_missing_added_count_within_baseline(self, part: str) -> None:
        """For each part, len(missing_added_truth_indexes) must be <= locked max."""
        report = _load_report(part)
        actual = _missing_added_count(report)
        expected_max = BASELINE_COUNTS[part]["max_missing_added"]
        assert actual <= expected_max, (
            f"{part}: missing_added_truth_indexes grew to {actual}, "
            f"baseline allows at most {expected_max}"
        )

    def test_all_nine_parts_are_covered(self) -> None:
        """BASELINE_COUNTS must cover parts 1 through 9 — the full corpus."""
        assert sorted(BASELINE_COUNTS.keys()) == [f"part{i}" for i in range(1, 10)], (
            f"BASELINE_COUNTS must hold exactly part1..part9, got "
            f"{sorted(BASELINE_COUNTS.keys())}"
        )

    def test_baseline_is_non_negative_and_self_consistent(self) -> None:
        """Sanity: no negative counts, conforming count never exceeds item count."""
        for part, bounds in BASELINE_COUNTS.items():
            assert bounds["min_conforming"] >= 0, f"{part}: negative min_conforming"
            assert bounds["max_missing_added"] >= 0, f"{part}: negative max_missing_added"
            report = _load_report(part)
            total_items = len(report.get("items", []))
            assert bounds["min_conforming"] <= total_items, (
                f"{part}: min_conforming {bounds['min_conforming']} exceeds "
                f"total item count {total_items}"
            )
