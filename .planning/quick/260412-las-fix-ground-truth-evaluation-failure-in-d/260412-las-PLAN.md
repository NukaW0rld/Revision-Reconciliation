---
quick_id: 260412-las
description: Fix ground truth evaluation failure in Docker: mount assets volume and normalize fixture key
date: 2026-04-12
must_haves:
  truths:
    - Assets directory is available inside the Docker container
    - Fixture key "Part 1" resolves to the "part1" directory
  artifacts:
    - docker/docker-compose.yml has assets volume mount
    - delta_preservation/evaluation/loader.py normalizes fixture key
---

# Quick Task 260412-las: Fix Ground Truth Evaluation Failure in Docker

## Root Cause

Two bugs caused `GroundTruthContractError: Ground truth fixture directory not found: /app/assets/Part 1`:

1. **Missing volume mount**: `docker-compose.yml` had no mount for `../assets:/app/assets`, so the fixture directories didn't exist inside the container.
2. **Key normalization**: User-entered part number "Part 1" was used as-is as the fixture directory name, but the actual directory is `part1` (lowercase, no space).

## Tasks

### Task 1: Mount assets directory in Docker

**File:** `docker/docker-compose.yml`
**Action:** Add `- ../assets:/app/assets:ro` to the volumes section.
**Done:** Added read-only assets volume mount.

### Task 2: Normalize fixture key in loader

**File:** `delta_preservation/evaluation/loader.py`
**Action:** Lowercase the key and strip spaces before building the fixture path.
**Done:** `normalized_key = truth_fixture_key.lower().replace(" ", "")` applied before path construction.
