import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from sqlalchemy.orm import Session
from delta_preservation.evaluation import load_ground_truth_packet
from delta_preservation.evaluation.conformance import ADDED_POOL_TOKEN_PREFIX
from delta_preservation.evaluation.contracts import GroundTruthContractError
from delta_preservation.types import DeltaItem
from shop.models import Run, ReviewItem, User
from shop.services.semantics import shape_semantic_contract
from shop.utils import utcnow

DEBUG_VERDICTS_FILENAME = "debug_verdicts.json"
DEBUG_NOTES_FILENAME = "debug_notes.json"
LEGACY_DEBUG_VERDICTS = {"correct", "incorrect", "partially_correct"}
VALID_DEBUG_VERDICTS = {"algorithm_error", "acceptable_alternate"}
PHASE2_REENTRY_MESSAGE = (
    "Phase 2 re-entry required: re-save legacy debug verdicts with "
    "`algorithm_error` or `acceptable_alternate`."
)


class DebugVerdictValidationError(ValueError):
    """Raised when a debug verdict payload is malformed."""


@dataclass(frozen=True)
class MissingAddedTruthItem:
    """Synthetic debug-queue row for an added ground-truth characteristic missing from the packet."""

    id: int
    truth_index: int
    requirement_revB: str | None
    snippet_center_revA: tuple[float, float] | None
    snippet_center_revB: tuple[float, float] | None
    char_no: int | None = None
    pipeline_classification: str = "added"
    confidence: float | None = None
    requirement_revA: str | None = None
    revA_snippet_path: str | None = None
    revB_snippet_path: str | None = None
    reviewer_decision: str | None = None
    override_classification: str | None = None
    override_note: str | None = None
    reviewed_at: object | None = None
    is_missing_truth: bool = True


def missing_added_truth_item_id(truth_index: int) -> int:
    """Return a stable negative pseudo-id for a synthetic missing-added row."""
    return -(truth_index + 1)


def truth_index_from_missing_added_item_id(item_id: int) -> int:
    """Invert ``missing_added_truth_item_id`` for router/save-path handling."""
    if item_id >= 0:
        raise DebugVerdictValidationError("Synthetic missing-added rows use negative pseudo ids.")
    return abs(item_id) - 1


def _load_delta_packet(run: Run) -> dict:
    if not run.output_dir:
        raise DebugVerdictValidationError("Run output directory is not configured.")
    packet_path = Path(run.output_dir) / "delta_packet.json"
    return json.loads(packet_path.read_text())


def _debug_verdicts_path(run: Run) -> Path:
    if not run.output_dir:
        raise DebugVerdictValidationError("Run output directory is not configured.")
    return Path(run.output_dir) / DEBUG_VERDICTS_FILENAME


def _debug_notes_path(run: Run) -> Path:
    if not run.output_dir:
        raise DebugVerdictValidationError("Run output directory is not configured.")
    return Path(run.output_dir) / DEBUG_NOTES_FILENAME


def load_debug_notes(run: Run) -> str:
    """Return the saved debug notes string, or '' if not yet written."""
    try:
        notes_path = _debug_notes_path(run)
    except DebugVerdictValidationError:
        return ""
    if not notes_path.exists():
        return ""
    try:
        data = json.loads(notes_path.read_text())
        return data.get("notes", "") if isinstance(data, dict) else ""
    except (OSError, json.JSONDecodeError):
        return ""


def save_debug_notes(run: Run, text: str) -> None:
    """Atomically persist run-level debug notes."""
    notes_path = _debug_notes_path(run)
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=notes_path.parent,
        prefix="debug_notes.",
        suffix=".tmp",
        delete=False,
    ) as tmp_file:
        json.dump({"notes": text}, tmp_file, indent=2, ensure_ascii=False)
        tmp_file.write("\n")
        tmp_path = Path(tmp_file.name)
    tmp_path.replace(notes_path)


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def load_debug_verdicts(run: Run) -> dict[int, dict]:
    """Return persisted debug verdicts keyed by ReviewItem.id."""
    verdicts_path = _debug_verdicts_path(run)
    if not verdicts_path.exists():
        return {}

    raw_data = json.loads(verdicts_path.read_text())
    if not isinstance(raw_data, dict):
        raise DebugVerdictValidationError("debug_verdicts.json must contain an object.")

    verdicts: dict[int, dict] = {}
    for raw_key, payload in raw_data.items():
        try:
            item_id = int(raw_key)
        except (TypeError, ValueError) as exc:
            raise DebugVerdictValidationError("Debug verdict keys must be ReviewItem ids.") from exc
        if not isinstance(payload, dict):
            continue
        normalized = validate_debug_verdict_payload(
            verdict=payload.get("verdict"),
            corrected_classification=payload.get("corrected_classification"),
            corrected_requirement_revA=payload.get("corrected_requirement_revA"),
            corrected_requirement_revB=payload.get("corrected_requirement_revB"),
            explanation=payload.get("explanation"),
        )
        verdicts[item_id] = {
            **payload,
            **normalized,
            "item_id": payload.get("item_id", item_id),
            "char_no": payload.get("char_no"),
        }
    return verdicts


def load_debug_verdicts_for_render(run: Run) -> dict[int, dict]:
    """Best-effort loader for queue rendering.

    The debug queue should stay renderable even if persisted data is partially
    malformed. Invalid files or entries are ignored here; strict validation is
    still available through ``load_debug_verdicts`` for write paths and tests.
    """
    try:
        verdicts_path = _debug_verdicts_path(run)
    except DebugVerdictValidationError:
        return {}
    if not verdicts_path.exists():
        return {}

    try:
        raw_data = json.loads(verdicts_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw_data, dict):
        return {}

    verdicts: dict[int, dict] = {}
    for raw_key, payload in raw_data.items():
        try:
            item_id = int(raw_key)
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        try:
            if payload.get("verdict") in LEGACY_DEBUG_VERDICTS:
                continue
            normalized = validate_debug_verdict_payload(
                verdict=payload.get("verdict"),
                corrected_classification=payload.get("corrected_classification"),
                corrected_requirement_revA=payload.get("corrected_requirement_revA"),
                corrected_requirement_revB=payload.get("corrected_requirement_revB"),
                explanation=payload.get("explanation"),
            )
        except DebugVerdictValidationError:
            continue
        verdicts[item_id] = {
            **payload,
            **normalized,
            "item_id": payload.get("item_id", item_id),
            "char_no": payload.get("char_no"),
        }
    return verdicts


def validate_debug_verdict_payload(
    *,
    verdict: str | None,
    corrected_classification: str | None = None,
    corrected_requirement_revA: str | None = None,
    corrected_requirement_revB: str | None = None,
    explanation: str | None = None,
) -> dict:
    normalized_verdict = (verdict or "").strip()
    if not normalized_verdict:
        raise DebugVerdictValidationError("Verdict is required.")
    if normalized_verdict in LEGACY_DEBUG_VERDICTS:
        raise DebugVerdictValidationError(PHASE2_REENTRY_MESSAGE)
    if normalized_verdict not in VALID_DEBUG_VERDICTS:
        raise DebugVerdictValidationError("Unsupported debug verdict.")

    payload = {"verdict": normalized_verdict}
    optional_fields = {
        "corrected_classification": _normalize_optional_text(corrected_classification),
        "corrected_requirement_revA": _normalize_optional_text(corrected_requirement_revA),
        "corrected_requirement_revB": _normalize_optional_text(corrected_requirement_revB),
        "explanation": _normalize_optional_text(explanation),
    }

    if normalized_verdict == "algorithm_error":
        required_for_algorithm_error = {"corrected_classification", "explanation"}
        missing = [
            field
            for field, value in optional_fields.items()
            if field in required_for_algorithm_error and value is None
        ]
        if missing:
            raise DebugVerdictValidationError(
                "Algorithm-error verdicts require corrected classification and explanation."
            )
    elif optional_fields["explanation"] is None:
        raise DebugVerdictValidationError("Acceptable-alternate verdicts require explanation.")

    for field, value in optional_fields.items():
        if value is not None:
            payload[field] = value
    return payload


def write_debug_verdicts(run: Run, verdicts_by_item_id: dict[int, dict]) -> None:
    """Atomically persist the debug verdict map for a run."""
    verdicts_path = _debug_verdicts_path(run)
    verdicts_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {str(item_id): payload for item_id, payload in verdicts_by_item_id.items()}

    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=verdicts_path.parent,
        prefix="debug_verdicts.",
        suffix=".tmp",
        delete=False,
    ) as tmp_file:
        json.dump(serializable, tmp_file, indent=2, sort_keys=True)
        tmp_file.write("\n")
        tmp_path = Path(tmp_file.name)

    tmp_path.replace(verdicts_path)


def save_debug_verdict(run: Run, item: ReviewItem, payload: dict) -> dict[int, dict]:
    """Upsert one debug verdict without mutating normal review decision fields."""
    verdicts_by_item_id = load_debug_verdicts(run)
    verdicts_by_item_id[item.id] = {
        **payload,
        "item_id": item.id,
        "char_no": item.char_no,
    }
    write_debug_verdicts(run, verdicts_by_item_id)
    return verdicts_by_item_id


def save_missing_added_truth_verdict(run: Run, truth_index: int, payload: dict) -> dict[int, dict]:
    """Persist one synthetic missing-added resolution alongside normal debug verdicts."""
    verdicts_by_item_id = load_debug_verdicts(run)
    verdicts_by_item_id[missing_added_truth_item_id(truth_index)] = {
        **payload,
        "item_id": None,
        "truth_index": truth_index,
        "char_no": None,
    }
    write_debug_verdicts(run, verdicts_by_item_id)
    return verdicts_by_item_id


def debug_verdict_state(items: list, verdicts_by_item_id: dict[int, dict]) -> dict:
    """Return template-friendly per-item verdict map and submitted-progress counters."""
    item_ids = {item.id for item in items}
    filtered = {item_id: payload for item_id, payload in verdicts_by_item_id.items() if item_id in item_ids}
    submitted = sum(1 for item in items if item.id in filtered)
    return {
        "by_item_id": filtered,
        "submitted": submitted,
        "total": len(items),
    }


def semantic_contracts_by_char(run: Run) -> dict[int | None, dict]:
    """Return template-safe semantic summaries keyed by char_no for a run."""
    if not run.output_dir:
        return {}

    packet_data = _load_delta_packet(run)
    shaped: dict[int | None, dict] = {}
    for raw_item in packet_data.get("items", []):
        char_no = raw_item.get("char_no")
        shaped[char_no] = shape_semantic_contract(DeltaItem.model_validate(raw_item))
    return shaped


def _bbox_center(raw_evidence: dict | None) -> tuple[float, float] | None:
    if not isinstance(raw_evidence, dict):
        return None
    bbox = raw_evidence.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _missing_added_mismatch(truth_index: int) -> dict:
    return {
        "code": "missing_added_characteristic",
        "message": f"ground truth added row {truth_index} was not captured by the algorithm",
    }



def debug_internals_by_char(run: Run) -> dict[int | None, dict]:
    """Return debug internals (scores, reasons, bbox centers) keyed by char_no."""
    if not run.output_dir:
        return {}
    packet_data = _load_delta_packet(run)
    result: dict[int | None, dict] = {}
    for raw_item in packet_data.get("items", []):
        char_no = raw_item.get("char_no")
        scores = raw_item.get("scores") or {}
        reasons = raw_item.get("reasons") or []
        evaluation = raw_item.get("evaluation")
        mismatches = []
        if isinstance(evaluation, dict):
            mismatches = evaluation.get("mismatches") or []

        result[char_no] = {
            "scores": scores,
            "reasons": reasons,
            "revA_center": _bbox_center(raw_item.get("revA")),
            "revB_center": _bbox_center(raw_item.get("revB")),
            "evaluation": evaluation,
            "mismatches": mismatches,
        }
    return result



def assemble_debug_report_payload(db: Session, run: Run) -> dict:
    """Return a deterministic, read-only debug export payload for a run.

    The merge intentionally preserves queue order using persisted ``ReviewItem.id``
    ordering plus the packet ordering used when the queue was first seeded. This
    avoids collapsing rows by ``char_no`` when values are ``None`` or repeated.
    """
    queue_state = build_debug_queue_state(db, run, activate_review=False)
    review_items = queue_state["all_items"]
    exception_item_ids = {item.id for item in queue_state["exception_items"]}
    packet_items_by_item_id = queue_state["packet_items_by_item_id"]
    raw_packet_items_by_item_id = queue_state["raw_packet_items_by_item_id"]
    missing_added_truth_items = queue_state.get("missing_added_truth_items", [])
    verdicts_by_item_id = load_debug_verdicts(run)
    packet_data = _load_delta_packet(run)

    rows: list[dict] = []
    debug_submitted = 0
    for queue_index, item in enumerate(review_items, start=1):
        delta_item = packet_items_by_item_id[item.id]
        raw_item = raw_packet_items_by_item_id[item.id]
        semantic_contract = shape_semantic_contract(delta_item)
        verdict_payload = verdicts_by_item_id.get(item.id)
        stored_verdict = verdict_payload or {}
        evaluation = delta_item.evaluation.model_dump() if delta_item.evaluation is not None else None
        mismatches = evaluation.get("mismatches", []) if evaluation is not None else []
        history_reference = evaluation.get("history_reference") if evaluation is not None else None
        if delta_item.evaluation is not None and delta_item.evaluation.status == "conforming":
            if delta_item.evaluation.conformance_source == "accepted_alternate":
                row_state = "acceptable_alternate"
            else:
                row_state = "canonical_match"
        elif verdict_payload is None:
            row_state = "unresolved_review_needed"
        else:
            row_state = stored_verdict.get("verdict")
            if item.id in exception_item_ids:
                debug_submitted += 1
        rows.append(
            {
                "queue_index": queue_index,
                "review_item_id": item.id,
                "char_no": item.char_no,
                "row_state": row_state,
                "pipeline_classification": item.pipeline_classification,
                "confidence": item.confidence,
                "requirement_revA": item.requirement_revA,
                "requirement_revB": item.requirement_revB,
                "reviewer_decision": item.reviewer_decision,
                "override_classification": item.override_classification,
                "override_note": item.override_note,
                "reviewed_at": item.reviewed_at.isoformat() if item.reviewed_at else None,
                "debug_verdict": stored_verdict.get("verdict"),
                "corrected_classification": stored_verdict.get("corrected_classification"),
                "corrected_requirement_revA": stored_verdict.get("corrected_requirement_revA"),
                "corrected_requirement_revB": stored_verdict.get("corrected_requirement_revB"),
                "explanation": stored_verdict.get("explanation"),
                "history_reference": history_reference,
                "scores": raw_item.get("scores") or {},
                "reasons": raw_item.get("reasons") or [],
                "evaluation": evaluation,
                "mismatches": mismatches,
                "semantic_callout": raw_item.get("semantic_callout"),
                "semantic_contract": semantic_contract,
                "revA_center": _bbox_center(raw_item.get("revA")),
                "revB_center": _bbox_center(raw_item.get("revB")),
                "packet_item": raw_item,
            }
        )

    for missing_item in missing_added_truth_items:
        verdict_payload = verdicts_by_item_id.get(missing_item.id)
        stored_verdict = verdict_payload or {}
        if verdict_payload is None:
            row_state = "unresolved_review_needed"
        else:
            row_state = stored_verdict.get("verdict")
            debug_submitted += 1
        rows.append(
            {
                "queue_index": len(rows) + 1,
                "review_item_id": None,
                "truth_index": missing_item.truth_index,
                "char_no": None,
                "row_state": row_state,
                "pipeline_classification": "added",
                "confidence": None,
                "requirement_revA": None,
                "requirement_revB": missing_item.requirement_revB,
                "reviewer_decision": None,
                "override_classification": None,
                "override_note": None,
                "reviewed_at": None,
                "debug_verdict": stored_verdict.get("verdict"),
                "corrected_classification": stored_verdict.get("corrected_classification"),
                "corrected_requirement_revA": None,
                "corrected_requirement_revB": stored_verdict.get("corrected_requirement_revB"),
                "explanation": stored_verdict.get("explanation"),
                "history_reference": None,
                "scores": {},
                "reasons": [],
                "evaluation": None,
                "mismatches": [_missing_added_mismatch(missing_item.truth_index)],
                "semantic_callout": None,
                "semantic_contract": None,
                "revA_center": missing_item.snippet_center_revA,
                "revB_center": missing_item.snippet_center_revB,
                "packet_item": None,
            }
        )

    return {
        "run_id": run.id,
        "run_status": run.status,
        "packet_run_id": packet_data.get("run_id"),
        "debug_total": len(exception_item_ids) + len(missing_added_truth_items),
        "debug_submitted": debug_submitted,
        "missing_added_truth_indexes": queue_state.get("missing_added_truth_indexes", []),
        "notes": load_debug_notes(run),
        "items": rows,
    }



def semantic_contract_for_item(run: Run, item: ReviewItem) -> dict | None:
    if item.char_no is None:
        return semantic_contracts_by_char(run).get(None)
    return semantic_contracts_by_char(run).get(item.char_no)


def _review_items_in_packet_order(db: Session, run: Run, *, activate_review: bool) -> list[ReviewItem]:
    existing_items = (
        db.query(ReviewItem)
        .filter(ReviewItem.run_id == run.id)
        .order_by(ReviewItem.id)
        .all()
    )
    if existing_items:
        if activate_review and run.status in ("completed", "warning"):
            run.status = "reviewing"
            db.commit()
        return existing_items

    # Load delta_packet
    packet_data = _load_delta_packet(run)

    # Optionally load requirement text from form3_chars.json
    req_map: dict[int, str] = {}
    chars_path = Path(run.output_dir) / "debug" / "form3_chars.json"
    if chars_path.exists():
        chars_data = json.loads(chars_path.read_text())
        for entry in chars_data:
            cno = entry.get("char_no")
            req = entry.get("requirement") or entry.get("req") or ""
            if cno is not None:
                req_map[int(cno)] = req

    items: list[ReviewItem] = []
    for delta in packet_data.get("items", []):
        char_no = delta.get("char_no")
        revA = delta.get("revA") or {}
        revB = delta.get("revB") or {}
        item = ReviewItem(
            run_id=run.id,
            char_no=char_no,
            pipeline_classification=delta.get("status", "uncertain"),
            confidence=delta.get("confidence", 0.0),
            requirement_revA=req_map.get(char_no) if char_no is not None else None,
            requirement_revB=delta.get("requirement_revB"),
            revA_snippet_path=revA.get("image_path"),
            revB_snippet_path=revB.get("image_path"),
            revA_bbox=revA.get("bbox"),
            revB_bbox=revB.get("bbox"),
        )
        db.add(item)
        items.append(item)

    if activate_review and run.status in ("completed", "warning"):
        run.status = "reviewing"
    db.commit()
    for item in items:
        db.refresh(item)
    return items


def _load_missing_added_truth_items(
    run: Run,
    packet_data: dict,
    packet_rows: list,
) -> list[MissingAddedTruthItem]:
    """Return synthetic rows for truth added characteristics not claimed by any packet row."""
    inputs = packet_data.get("inputs") or {}
    truth_fixture_key = inputs.get("truth_fixture_key") or run.part_number
    if not truth_fixture_key:
        return []
    try:
        truth_packet = load_ground_truth_packet(truth_fixture_key)
    except (GroundTruthContractError, Exception):
        return []

    # Collect truth added row indexes
    truth_added_rows = [
        (i, ch)
        for i, ch in enumerate(truth_packet.characteristics)
        if ch.classification == "added"
    ]
    if not truth_added_rows:
        return []

    # Build a lookup from legacy char_no integer → truth_index for added rows so
    # packets serialized before the canonical ``added:<index>`` token contract was
    # enforced can still claim their truth rows correctly.
    added_char_no_to_truth_index: dict[int, int] = {
        ch.char_no: i
        for i, ch in enumerate(truth_packet.characteristics)
        if ch.classification == "added" and ch.char_no is not None
    }

    # Collect matched added token indexes from packet evaluations.
    # Accept both the canonical string form ``added:<index>`` and a legacy integer
    # token when that integer is the char_no of an added truth row.
    claimed: set[int] = set()
    for _item, delta_item in packet_rows:
        if delta_item.evaluation is None:
            continue
        token = delta_item.evaluation.matched_truth_char_no
        if isinstance(token, str) and token.startswith(f"{ADDED_POOL_TOKEN_PREFIX}:"):
            try:
                idx = int(token.split(":", 1)[1])
                claimed.add(idx)
            except (ValueError, IndexError):
                pass
        elif isinstance(token, int) and token in added_char_no_to_truth_index:
            # Legacy integer token that resolves to an added truth row.
            claimed.add(added_char_no_to_truth_index[token])

    rows: list[MissingAddedTruthItem] = []
    for truth_index, truth_row in truth_added_rows:
        if truth_index in claimed:
            continue
        rows.append(
            MissingAddedTruthItem(
                id=missing_added_truth_item_id(truth_index),
                truth_index=truth_index,
                requirement_revB=truth_row.requirement_revB,
                snippet_center_revA=truth_row.snippet_center_revA,
                snippet_center_revB=truth_row.snippet_center_revB,
            )
        )
    return rows


def build_debug_queue_state(db: Session, run: Run, *, activate_review: bool = True) -> dict:
    """Return stable packet-to-review-item pairing for the admin debug queue."""
    try:
        all_items = _review_items_in_packet_order(db, run, activate_review=activate_review)
        packet_data = _load_delta_packet(run)
    except DebugVerdictValidationError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise DebugVerdictValidationError("delta_packet.json could not be read for debug queue.") from exc

    raw_items = packet_data.get("items")
    if raw_items is None and packet_data.get("characters") == []:
        raw_items = []
    if not isinstance(raw_items, list):
        raise DebugVerdictValidationError("delta_packet.json must contain an items array.")
    if len(raw_items) != len(all_items):
        raise DebugVerdictValidationError(
            "Review queue and delta packet row counts do not align for debug queue."
        )

    packet_items_by_item_id: dict[int, DeltaItem] = {}
    raw_packet_items_by_item_id: dict[int, dict] = {}
    exception_items: list[ReviewItem] = []
    packet_rows: list[tuple[ReviewItem, DeltaItem]] = []
    for item, raw_item in zip(all_items, raw_items):
        delta_item = DeltaItem.model_validate(raw_item)
        packet_items_by_item_id[item.id] = delta_item
        raw_packet_items_by_item_id[item.id] = raw_item
        packet_rows.append((item, delta_item))
        if delta_item.evaluation is not None and delta_item.evaluation.status == "review_needed":
            exception_items.append(item)

    missing_added_truth_items = _load_missing_added_truth_items(run, packet_data, packet_rows)
    missing_added_truth_indexes = [item.truth_index for item in missing_added_truth_items]

    return {
        "all_items": all_items,
        "exception_items": exception_items,
        "missing_added_truth_items": missing_added_truth_items,
        "missing_added_truth_indexes": missing_added_truth_indexes,
        "packet_items_by_item_id": packet_items_by_item_id,
        "raw_packet_items_by_item_id": raw_packet_items_by_item_id,
        "packet_rows": packet_rows,
        "packet_declares_items": "items" in packet_data,
        "debug_total": len(exception_items) + len(missing_added_truth_indexes),
    }


def open_review_queue(db: Session, run: Run) -> list[ReviewItem]:
    """Idempotent: create ReviewItems from delta_packet.json on first call.

    Returns all ReviewItems for this run ordered by ReviewItem.id.
    If ReviewItems already exist for this run, returns them without creating
    duplicates (idempotent for subsequent calls).
    """
    return _review_items_in_packet_order(db, run, activate_review=True)


def semantic_contracts_by_item_id(db: Session, run: Run) -> dict[int, dict]:
    """Return semantic summaries keyed by ReviewItem.id for debug UI rendering."""
    queue_state = build_debug_queue_state(db, run, activate_review=False)
    return {
        item.id: shape_semantic_contract(delta_item)
        for item, delta_item in queue_state["packet_rows"]
    }


def debug_internals_by_item_id(db: Session, run: Run) -> dict[int, dict]:
    """Return debug internals keyed by ReviewItem.id for stable debug UI rendering."""
    queue_state = build_debug_queue_state(db, run, activate_review=False)
    result: dict[int, dict] = {}
    for item, delta_item in queue_state["packet_rows"]:
        raw_item = queue_state["raw_packet_items_by_item_id"][item.id]
        evaluation = delta_item.evaluation.model_dump() if delta_item.evaluation is not None else None
        mismatches = evaluation.get("mismatches", []) if evaluation is not None else []
        result[item.id] = {
            "scores": raw_item.get("scores") or {},
            "reasons": raw_item.get("reasons") or [],
            "revA_center": _bbox_center(raw_item.get("revA")),
            "revB_center": _bbox_center(raw_item.get("revB")),
            "evaluation": evaluation,
            "mismatches": mismatches,
        }
    for missing_item in queue_state.get("missing_added_truth_items", []):
        result[missing_item.id] = {
            "scores": {},
            "reasons": [],
            "revA_center": missing_item.snippet_center_revA,
            "revB_center": missing_item.snippet_center_revB,
            "evaluation": None,
            "mismatches": [_missing_added_mismatch(missing_item.truth_index)],
        }
    return result


def build_run_debug_summary(db: Session, run: Run) -> dict:
    """Return status-page debug summary grouped into conforming and exception rows."""
    queue_state = build_debug_queue_state(db, run, activate_review=False)
    verdicts_by_item_id = load_debug_verdicts_for_render(run)
    conforming_rows: list[dict] = []
    exception_rows: list[dict] = []
    resolved_exception_count = 0
    missing_added_truth_items = queue_state.get("missing_added_truth_items", [])
    missing_added_truth_indexes = [item.truth_index for item in missing_added_truth_items]

    for queue_index, (item, delta_item) in enumerate(queue_state["packet_rows"], start=1):
        raw_item = queue_state["raw_packet_items_by_item_id"][item.id]
        mismatches = []
        evaluation = delta_item.evaluation.model_dump() if delta_item.evaluation is not None else None
        if evaluation is not None:
            mismatches = evaluation.get("mismatches", [])
        row = {
            "queue_index": queue_index,
            "review_item_id": item.id,
            "char_no": item.char_no,
            "pipeline_classification": item.pipeline_classification,
            "requirement_revB": item.requirement_revB,
            "saved_verdict": verdicts_by_item_id.get(item.id, {}).get("verdict"),
            "mismatches": mismatches,
            "packet_item": raw_item,
        }
        if delta_item.evaluation is not None and delta_item.evaluation.status == "conforming":
            conforming_rows.append(row)
            continue
        if row["saved_verdict"] is not None:
            resolved_exception_count += 1
        exception_rows.append(row)

    for missing_item in missing_added_truth_items:
        queue_index = len(conforming_rows) + len(exception_rows) + 1
        saved_verdict = verdicts_by_item_id.get(missing_item.id, {}).get("verdict")
        if saved_verdict is not None:
            resolved_exception_count += 1
        exception_rows.append({
            "queue_index": queue_index,
            "review_item_id": None,
            "truth_index": missing_item.truth_index,
            "char_no": None,
            "pipeline_classification": None,
            "requirement_revB": missing_item.requirement_revB,
            "saved_verdict": saved_verdict,
            "mismatches": [_missing_added_mismatch(missing_item.truth_index)],
            "packet_item": None,
            "row_state": "missing_added_truth",
        })

    unresolved_exception_count = len(exception_rows) - resolved_exception_count
    return {
        "conforming_rows": conforming_rows,
        "exception_rows": exception_rows,
        "resolved_exception_count": resolved_exception_count,
        "unresolved_exception_count": unresolved_exception_count,
        "missing_added_truth_indexes": missing_added_truth_indexes,
        "missing_added_truth_count": len(missing_added_truth_indexes),
        "debug_report_ready": unresolved_exception_count == 0,
    }


def attempt_sign_off(db: Session, run: Run, reviewer_id: int) -> bool:
    """Atomic sign-off using two-phase write pattern.

    Returns True on success (run.status == 'signed_off').
    Returns False on any failure (run.status rolled back to 'reviewing').

    SIGNOFF-03: Returns False immediately if run is already signed_off.
    """
    if run.status == "signed_off":
        return False  # Immutability guard — never re-sign

    # Phase 1: mark as in-progress (visible to SSE polling)
    run.status = "signing_off"
    db.commit()

    try:
        # Phase 2: set signed_at/signed_by_id BEFORE generate so they're in the PDF
        run.signed_at = utcnow()
        run.signed_by_id = reviewer_id
        # Phase 4: generate and persist audit packet PDF (raises on failure → caught below)
        from shop.services.exports import generate_and_store_audit_packet
        generate_and_store_audit_packet(db, run)
        run.status = "signed_off"
        db.commit()
        return True
    except Exception:
        # Rollback to reviewable state — no signed-but-no-packet state allowed
        db.rollback()
        # Re-query run after rollback (object may be in detached state)
        run = db.query(Run).filter(Run.id == run.id).first()
        if run is not None:
            run.status = "reviewing"
            db.commit()
        return False
