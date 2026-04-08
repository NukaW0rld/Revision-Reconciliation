"""
Auto-verdict script for delta-preservation debug workflow.

For each assets/part*/ directory that contains a ground_truth.json, finds the
most recent matching run in out/, auto-populates correct verdicts in the run's
debug_verdicts.json, prints a table of non-correct items, then writes the final
debug_report_partN.json into assets/partN/ once all verdicts are complete.

Usage:
    python auto_verdict.py                  # process all parts
    python auto_verdict.py assets/part3     # process one part

Environment:
    DATABASE_URL  defaults to sqlite:///./data/shop.db (container-mounted path)
    OUT_DIR       defaults to ./out
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

# Default to the container-mounted DB path
os.environ.setdefault("DATABASE_URL", "sqlite:///./data/shop.db")

OUT_DIR = Path(os.environ.get("OUT_DIR", "./out"))
ASSETS_DIR = Path("./assets")
SNIPPET_CENTER_THRESHOLD_PT = 150.0  # PDF points
DEBUG_VERDICTS_FILENAME = "debug_verdicts.json"

# The container mounts out/ as /app/out; translate stored paths back to local.
_CONTAINER_OUT_PREFIX = "/app/out"
_LOCAL_OUT_PREFIX = str(OUT_DIR.resolve())


def _localize_path(p: str | None) -> Path | None:
    """Translate /app/out/... → ./out/... for paths stored in the container DB."""
    if not p:
        return None
    if p.startswith(_CONTAINER_OUT_PREFIX):
        return Path(_LOCAL_OUT_PREFIX + p[len(_CONTAINER_OUT_PREFIX):])
    return Path(p)


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def load_ground_truth(part_dir: Path) -> dict:
    gt_path = part_dir / "ground_truth.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"No ground_truth.json in {part_dir}")
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    if "characteristics" not in data:
        raise ValueError(f"{gt_path}: missing 'characteristics' key")
    return data


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def find_latest_run_dir(part_name: str) -> Path | None:
    """Return the most recent out/ subdirectory whose name starts with part_name."""
    if not OUT_DIR.exists():
        return None
    candidates = [
        d for d in OUT_DIR.iterdir()
        if d.is_dir() and d.name.startswith(part_name + "_")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda d: d.name, reverse=True)
    return candidates[0]


def load_delta_packet(run_dir: Path) -> dict:
    packet_path = run_dir / "delta_packet.json"
    if not packet_path.exists():
        raise FileNotFoundError(f"No delta_packet.json in {run_dir}")
    return json.loads(packet_path.read_text(encoding="utf-8"))


def load_debug_verdicts(run_dir: Path) -> dict[str, dict]:
    path = run_dir / DEBUG_VERDICTS_FILENAME
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


def write_debug_verdicts(run_dir: Path, verdicts: dict[str, dict]) -> None:
    path = run_dir / DEBUG_VERDICTS_FILENAME
    serializable = {str(k): v for k, v in verdicts.items()}
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix="debug_verdicts.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        json.dump(serializable, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# DB access
# ---------------------------------------------------------------------------

def get_db_session():
    from shop.database import SessionLocal
    return SessionLocal()


def get_run_and_items(run_dir: Path):
    """Return (Run, list[ReviewItem]) for the run whose output_dir matches run_dir."""
    from shop.models import Run, ReviewItem

    target = str(run_dir.resolve())
    db = get_db_session()
    try:
        all_runs = db.query(Run).filter(Run.output_dir.isnot(None)).all()
        matched = [
            r for r in all_runs
            if _localize_path(r.output_dir) is not None
            and str(_localize_path(r.output_dir).resolve()) == target
        ]
        if not matched:
            return None, []
        run = matched[0]
        items = (
            db.query(ReviewItem)
            .filter(ReviewItem.run_id == run.id)
            .order_by(ReviewItem.id)
            .all()
        )
        # Detach from session so we can use objects after close
        db.expunge_all()
        return run, items
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _bbox_center(bbox) -> tuple[float, float] | None:
    if not bbox or len(bbox) != 4:
        return None
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _snippet_ok(actual_bbox, expected_center: list | None) -> bool | None:
    """None = axis not applicable (expected_center is null in ground truth)."""
    if expected_center is None:
        return None
    actual = _bbox_center(actual_bbox)
    if actual is None:
        return False  # pipeline produced no bbox where one was expected
    return _distance(actual, tuple(expected_center)) <= SNIPPET_CENTER_THRESHOLD_PT


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def _normalize_text(s: str | None) -> str:
    if not s:
        return ""
    return " ".join(s.strip().split()).lower()


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def compute_auto_verdict(
    gt_char: dict,
    pipeline_classification: str,
    requirement_revB: str | None,
    revA_bbox,
    revB_bbox,
) -> dict:
    """
    Returns {"verdict": str, "axes": dict}.
    verdict: "correct" | "incorrect" | "partially_correct"
    """
    gt_cls = gt_char["classification"]
    gt_revb = gt_char.get("requirement_revB")
    gt_center_revA = gt_char.get("snippet_center_revA")
    gt_center_revB = gt_char.get("snippet_center_revB")

    cls_ok = pipeline_classification == gt_cls

    if gt_revb is None:
        revb_text_ok = not requirement_revB or requirement_revB.strip() == ""
    else:
        revb_text_ok = _normalize_text(requirement_revB) == _normalize_text(gt_revb)

    reva_snippet_ok = _snippet_ok(revA_bbox, gt_center_revA)
    revb_snippet_ok = _snippet_ok(revB_bbox, gt_center_revB)

    axes = {
        "classification": cls_ok,
        "revb_text": revb_text_ok,
        "revb_snippet": revb_snippet_ok,
        "reva_snippet": reva_snippet_ok,
    }

    if not cls_ok or not revb_text_ok:
        verdict = "incorrect"
    elif revb_snippet_ok is False or reva_snippet_ok is False:
        verdict = "partially_correct"
    else:
        verdict = "correct"

    return {"verdict": verdict, "axes": axes}


# ---------------------------------------------------------------------------
# Changed-char split detection
# ---------------------------------------------------------------------------

def find_split_chars(
    gt_by_charno: dict[int, dict],
    pipeline_items: list[dict],
) -> set[int]:
    """
    Return char_nos where ground truth expects 'changed' but pipeline produced
    no 'changed' item for that char_no.

    This only applies to ground-truth rows that are anchored to revA balloons,
    so added rows must not be present in gt_by_charno.
    """
    changed_gt = {cn for cn, gt in gt_by_charno.items() if gt["classification"] == "changed"}
    pipeline_cls_by_charno: dict[int | None, set[str]] = {}
    for item in pipeline_items:
        cn = item.get("char_no")
        pipeline_cls_by_charno.setdefault(cn, set()).add(item.get("classification", ""))
    return {
        cn for cn in changed_gt
        if "changed" not in pipeline_cls_by_charno.get(cn, set())
    }


def find_best_added_match(
    gt_char: dict,
    review_items: list,
    consumed_item_ids: set[int],
):
    """
    Match a ground-truth added characteristic to an unmatched pipeline added item.

    Match priority:
    1. normalized revB requirement text exact match
    2. revB center distance ascending

    Returns a ReviewItem or None.
    """
    gt_revb = _normalize_text(gt_char.get("requirement_revB"))
    gt_center_revB = gt_char.get("snippet_center_revB")
    if gt_center_revB is None:
        return None

    candidates = []
    for item in review_items:
        if item.id in consumed_item_ids:
            continue
        if item.pipeline_classification != "added":
            continue
        if _normalize_text(item.requirement_revB) != gt_revb:
            continue
        actual_center = _bbox_center(item.revB_bbox)
        if actual_center is None:
            continue
        dist = _distance(actual_center, tuple(gt_center_revB))
        candidates.append((dist, item))

    if not candidates:
        return None

    candidates.sort(key=lambda pair: pair[0])
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Part number extraction
# ---------------------------------------------------------------------------

def extract_part_number(part_dir_name: str) -> str:
    m = re.search(r"part(\d+)", part_dir_name, re.IGNORECASE)
    return m.group(1) if m else part_dir_name


# ---------------------------------------------------------------------------
# Core: process one part
# ---------------------------------------------------------------------------

def process_part(part_dir: Path) -> bool:
    """Returns True if all verdicts are complete and debug_report was written."""
    part_num = extract_part_number(part_dir.name)
    print(f"\n{'='*60}")
    print(f"Part {part_num}  ({part_dir.name})")
    print(f"{'='*60}")

    try:
        gt = load_ground_truth(part_dir)
    except FileNotFoundError:
        print("  SKIP: no ground_truth.json")
        return False

    part_name: str = gt["part_name"]
    general_notes: str = gt.get("general_notes", "")
    gt_chars: list[dict] = gt["characteristics"]
    gt_non_added = [c for c in gt_chars if c.get("classification") != "added"]
    gt_added = [c for c in gt_chars if c.get("classification") == "added"]
    gt_by_charno: dict[int, dict] = {c["char_no"]: c for c in gt_non_added}

    run_dir = find_latest_run_dir(part_name)
    if run_dir is None:
        print(f"  SKIP: no run found in {OUT_DIR} for part_name='{part_name}'")
        return False
    print(f"  Run: {run_dir.name}")

    try:
        packet = load_delta_packet(run_dir)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return False

    raw_items: list[dict] = packet.get("items", [])

    run, review_items = get_run_and_items(run_dir)
    if run is None:
        print(f"  ERROR: run not found in DB for {run_dir}")
        return False

    # Map char_no → list[ReviewItem]
    review_by_charno: dict[int | None, list] = {}
    for item in review_items:
        review_by_charno.setdefault(item.char_no, []).append(item)

    split_chars = find_split_chars(gt_by_charno, raw_items)

    existing_verdicts = load_debug_verdicts(run_dir)
    updated_verdicts = dict(existing_verdicts)

    non_correct: list[dict] = []
    auto_correct_count = 0
    consumed_added_item_ids: set[int] = set()

    for gt_char in gt_non_added:
        char_no: int = gt_char["char_no"]
        gt_cls = gt_char["classification"]

        # --- Split case: ground truth 'changed' but pipeline produced removed+added ---
        if char_no in split_chars and gt_cls == "changed":
            pipeline_items_for_char = [r for r in raw_items if r.get("char_no") == char_no]
            review_items_for_char = review_by_charno.get(char_no, [])
            for raw in pipeline_items_for_char:
                pcls = raw.get("classification", "")
                matching = [ri for ri in review_items_for_char if ri.pipeline_classification == pcls]
                for ri in matching:
                    key = str(ri.id)
                    if key not in updated_verdicts or not updated_verdicts[key].get("verdict"):
                        updated_verdicts[key] = {
                            "item_id": ri.id,
                            "char_no": char_no,
                            "verdict": "partially_correct",
                            "corrected_classification": "changed",
                            "corrected_requirement_revA": None,
                            "corrected_requirement_revB": gt_char.get("requirement_revB"),
                            "explanation": (
                                f"Pipeline split char {char_no} into removed+added pair "
                                f"instead of a single 'changed' item."
                            ),
                        }
                    non_correct.append({
                        "char_no": char_no,
                        "verdict": "partially_correct",
                        "pipeline_cls": pcls,
                        "gt_cls": gt_cls,
                        "issue": "changed split into removed+added",
                        "review_item_id": ri.id,
                    })
            continue

        # --- Normal case ---
        items_for_char = review_by_charno.get(char_no, [])
        if not items_for_char:
            print(f"  WARNING: char_no={char_no} not found in ReviewItems — skipping")
            continue

        ri = items_for_char[0]
        key = str(ri.id)

        result = compute_auto_verdict(
            gt_char=gt_char,
            pipeline_classification=ri.pipeline_classification,
            requirement_revB=ri.requirement_revB,
            revA_bbox=ri.revA_bbox,
            revB_bbox=ri.revB_bbox,
        )
        verdict = result["verdict"]
        axes = result["axes"]

        if verdict == "correct":
            updated_verdicts[key] = {
                "item_id": ri.id,
                "char_no": char_no,
                "verdict": "correct",
                "corrected_classification": None,
                "corrected_requirement_revA": None,
                "corrected_requirement_revB": None,
                "explanation": None,
            }
            auto_correct_count += 1
        else:
            issues = []
            if not axes["classification"]:
                issues.append(f"classification: got '{ri.pipeline_classification}', expected '{gt_cls}'")
            if not axes["revb_text"]:
                issues.append(f"revB text: got '{ri.requirement_revB}', expected '{gt_char.get('requirement_revB')}'")
            if axes["revb_snippet"] is False:
                actual = _bbox_center(ri.revB_bbox)
                issues.append(f"revB snippet off (actual={actual}, expected={gt_char.get('snippet_center_revB')})")
            if axes["reva_snippet"] is False:
                actual = _bbox_center(ri.revA_bbox)
                issues.append(f"revA snippet off (actual={actual}, expected={gt_char.get('snippet_center_revA')})")

            non_correct.append({
                "char_no": char_no,
                "verdict": verdict,
                "pipeline_cls": ri.pipeline_classification,
                "gt_cls": gt_cls,
                "issue": "; ".join(issues),
                "review_item_id": ri.id,
            })

            # Write placeholder — you fill explanation via UI
            if key not in updated_verdicts or not updated_verdicts[key].get("verdict"):
                updated_verdicts[key] = {
                    "item_id": ri.id,
                    "char_no": char_no,
                    "verdict": verdict,
                    "corrected_classification": gt_cls if not axes["classification"] else None,
                    "corrected_requirement_revA": None,
                    "corrected_requirement_revB": (
                        gt_char.get("requirement_revB") if not axes["revb_text"] else None
                    ),
                    "explanation": "",
                }

    for gt_char in gt_added:
        gt_cls = gt_char["classification"]
        matched_item = find_best_added_match(gt_char, review_items, consumed_added_item_ids)
        if matched_item is None:
            non_correct.append({
                "char_no": None,
                "verdict": "incorrect",
                "pipeline_cls": "<missing>",
                "gt_cls": gt_cls,
                "issue": (
                    "no unmatched pipeline added-item matched expected revB text/location "
                    f"(expected text='{gt_char.get('requirement_revB')}')"
                ),
                "review_item_id": None,
            })
            continue

        consumed_added_item_ids.add(matched_item.id)
        key = str(matched_item.id)
        result = compute_auto_verdict(
            gt_char=gt_char,
            pipeline_classification=matched_item.pipeline_classification,
            requirement_revB=matched_item.requirement_revB,
            revA_bbox=matched_item.revA_bbox,
            revB_bbox=matched_item.revB_bbox,
        )
        verdict = result["verdict"]
        axes = result["axes"]

        if verdict == "correct":
            updated_verdicts[key] = {
                "item_id": matched_item.id,
                "char_no": matched_item.char_no,
                "verdict": "correct",
                "corrected_classification": None,
                "corrected_requirement_revA": None,
                "corrected_requirement_revB": None,
                "explanation": None,
            }
            auto_correct_count += 1
        else:
            issues = []
            if not axes["classification"]:
                issues.append(
                    f"classification: got '{matched_item.pipeline_classification}', expected '{gt_cls}'"
                )
            if not axes["revb_text"]:
                issues.append(
                    f"revB text: got '{matched_item.requirement_revB}', expected '{gt_char.get('requirement_revB')}'"
                )
            if axes["revb_snippet"] is False:
                actual = _bbox_center(matched_item.revB_bbox)
                issues.append(
                    f"revB snippet off (actual={actual}, expected={gt_char.get('snippet_center_revB')})"
                )

            non_correct.append({
                "char_no": matched_item.char_no,
                "verdict": verdict,
                "pipeline_cls": matched_item.pipeline_classification,
                "gt_cls": gt_cls,
                "issue": "; ".join(issues),
                "review_item_id": matched_item.id,
            })

            if key not in updated_verdicts or not updated_verdicts[key].get("verdict"):
                updated_verdicts[key] = {
                    "item_id": matched_item.id,
                    "char_no": matched_item.char_no,
                    "verdict": verdict,
                    "corrected_classification": gt_cls if not axes["classification"] else None,
                    "corrected_requirement_revA": None,
                    "corrected_requirement_revB": (
                        gt_char.get("requirement_revB") if not axes["revb_text"] else None
                    ),
                    "explanation": "",
                }

    write_debug_verdicts(run_dir, updated_verdicts)

    # --- Summary ---
    total_gt = len(gt_chars)
    print(f"  Ground truth chars : {total_gt}")
    print(f"  Auto-correct       : {auto_correct_count}")
    print(f"  Needs review       : {len(non_correct)}")
    if general_notes:
        print(f"  Notes              : {general_notes}")

    if non_correct:
        print()
        print(f"  {'CHAR':>5}  {'VERDICT':<18}  {'PIPELINE':<12}  {'EXPECTED':<12}  ISSUE")
        print(f"  {'-'*5}  {'-'*18}  {'-'*12}  {'-'*12}  {'-'*40}")
        for nc in sorted(non_correct, key=lambda x: (x["char_no"] is None, x["char_no"] or 0)):
            print(
                f"  {str(nc['char_no'] or '?'):>5}  "
                f"{nc['verdict']:<18}  "
                f"{nc['pipeline_cls']:<12}  "
                f"{nc['gt_cls']:<12}  "
                f"{nc['issue']}"
            )

    # --- Check completeness: non-correct items need a non-empty explanation ---
    pending = []
    for nc in non_correct:
        review_item_id = nc.get("review_item_id")
        if review_item_id is None:
            pending.append(nc["char_no"])
            continue
        if not updated_verdicts.get(str(review_item_id), {}).get("explanation", "").strip():
            pending.append(nc["char_no"])

    if pending:
        print(f"\n  Pending explanation for chars: {pending}")
        print("  Fill these in via the UI, then re-run to export the report.")
        return False

    return _write_debug_report(
        part_dir=part_dir,
        part_num=part_num,
        part_name=part_name,
        general_notes=general_notes,
        run=run,
        review_items=review_items,
        raw_items=raw_items,
        updated_verdicts=updated_verdicts,
    )


# ---------------------------------------------------------------------------
# Debug report assembly
# ---------------------------------------------------------------------------

def _evidence_center(evidence: dict | None) -> tuple | None:
    if not isinstance(evidence, dict):
        return None
    return _bbox_center(evidence.get("bbox"))


def _write_debug_report(
    *,
    part_dir: Path,
    part_num: str,
    part_name: str,
    general_notes: str,
    run,
    review_items: list,
    raw_items: list[dict],
    updated_verdicts: dict[str, dict],
) -> bool:
    try:
        from shop.services.semantics import shape_semantic_contract
        from shop.types import DeltaItem
        use_semantics = True
    except Exception:
        use_semantics = False

    ordered_packet_items = sorted(
        raw_items,
        key=lambda x: (x.get("char_no") is None, x.get("char_no") or 0),
    )

    rows = []
    for queue_index, (item, raw_item) in enumerate(zip(review_items, ordered_packet_items), start=1):
        semantic_contract = None
        if use_semantics:
            try:
                delta_item = DeltaItem.model_validate(raw_item)
                semantic_contract = shape_semantic_contract(delta_item)
            except Exception:
                pass

        verdict_payload = updated_verdicts.get(str(item.id), {})
        reviewed_at = None
        if hasattr(item, "reviewed_at") and item.reviewed_at:
            reviewed_at = item.reviewed_at.isoformat()

        rows.append({
            "queue_index": queue_index,
            "review_item_id": item.id,
            "char_no": item.char_no,
            "pipeline_classification": item.pipeline_classification,
            "confidence": item.confidence,
            "requirement_revA": item.requirement_revA,
            "requirement_revB": item.requirement_revB,
            "reviewer_decision": item.reviewer_decision,
            "override_classification": item.override_classification,
            "override_note": item.override_note,
            "reviewed_at": reviewed_at,
            "debug_verdict": verdict_payload.get("verdict"),
            "corrected_classification": verdict_payload.get("corrected_classification"),
            "corrected_requirement_revA": verdict_payload.get("corrected_requirement_revA"),
            "corrected_requirement_revB": verdict_payload.get("corrected_requirement_revB"),
            "explanation": verdict_payload.get("explanation"),
            "scores": raw_item.get("scores") or {},
            "reasons": raw_item.get("reasons") or [],
            "semantic_callout": raw_item.get("semantic_callout"),
            "semantic_contract": semantic_contract,
            "revA_center": _evidence_center(raw_item.get("revA")),
            "revB_center": _evidence_center(raw_item.get("revB")),
            "packet_item": raw_item,
        })

    payload = {
        "part_name": part_name,
        "general_notes": general_notes,
        "run_id": run.id,
        "run_status": run.status,
        "debug_total": len(review_items),
        "debug_submitted": len(rows),
        "items": rows,
    }

    out_path = part_dir / f"debug_report_part{part_num}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n  ✓ Written: {out_path}")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Auto-verdict debug reports from ground truth.")
    parser.add_argument(
        "part_dir",
        nargs="?",
        help="Single assets/partN directory to process. Omit to process all.",
    )
    args = parser.parse_args()

    if args.part_dir:
        part_dirs = [Path(args.part_dir)]
    else:
        part_dirs = sorted(
            [d for d in ASSETS_DIR.iterdir() if d.is_dir() and re.match(r"part\d+", d.name, re.IGNORECASE)],
            key=lambda d: int(re.search(r"\d+", d.name).group()),
        )

    if not part_dirs:
        print(f"No part directories found in {ASSETS_DIR}")
        sys.exit(1)

    complete = 0
    for part_dir in part_dirs:
        ok = process_part(part_dir)
        if ok:
            complete += 1

    print(f"\n{'='*60}")
    print(f"Done: {complete}/{len(part_dirs)} parts exported.")
    if complete < len(part_dirs):
        print("Re-run after filling in explanations via the UI.")


if __name__ == "__main__":
    main()
