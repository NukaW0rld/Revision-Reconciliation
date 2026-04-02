import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from sqlalchemy.orm import Session
from delta_preservation.types import DeltaItem
from shop.models import Run, ReviewItem, User
from shop.services.semantics import shape_semantic_contract
from shop.utils import utcnow

DEBUG_VERDICTS_FILENAME = "debug_verdicts.json"
VALID_DEBUG_VERDICTS = {"correct", "incorrect", "partially_correct"}


class DebugVerdictValidationError(ValueError):
    """Raised when a debug verdict payload is malformed."""


def _load_delta_packet(run: Run) -> dict:
    packet_path = Path(run.output_dir) / "delta_packet.json"
    return json.loads(packet_path.read_text())


def _debug_verdicts_path(run: Run) -> Path:
    if not run.output_dir:
        raise DebugVerdictValidationError("Run output directory is not configured.")
    return Path(run.output_dir) / DEBUG_VERDICTS_FILENAME


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
        if isinstance(payload, dict):
            verdicts[item_id] = payload
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
    if normalized_verdict not in VALID_DEBUG_VERDICTS:
        raise DebugVerdictValidationError("Unsupported debug verdict.")

    payload = {"verdict": normalized_verdict}
    optional_fields = {
        "corrected_classification": _normalize_optional_text(corrected_classification),
        "corrected_requirement_revA": _normalize_optional_text(corrected_requirement_revA),
        "corrected_requirement_revB": _normalize_optional_text(corrected_requirement_revB),
        "explanation": _normalize_optional_text(explanation),
    }

    if normalized_verdict != "correct":
        missing = [field for field, value in optional_fields.items() if value is None]
        if missing:
            raise DebugVerdictValidationError(
                "Corrected classification, corrected Rev A requirement, corrected Rev B requirement, and explanation are required for non-correct verdicts."
            )

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


def debug_verdict_state(items: list[ReviewItem], verdicts_by_item_id: dict[int, dict]) -> dict:
    """Return template-friendly per-item verdict map and submitted-progress counters."""
    item_ids = {item.id for item in items}
    filtered = {item_id: payload for item_id, payload in verdicts_by_item_id.items() if item_id in item_ids}
    return {
        "by_item_id": filtered,
        "submitted": len(filtered),
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

        revA = raw_item.get("revA") or {}
        revA_bbox = revA.get("bbox")
        revA_center = (
            ((revA_bbox[0] + revA_bbox[2]) / 2, (revA_bbox[1] + revA_bbox[3]) / 2)
            if revA_bbox
            else None
        )

        revB = raw_item.get("revB") or {}
        revB_bbox = revB.get("bbox")
        revB_center = (
            ((revB_bbox[0] + revB_bbox[2]) / 2, (revB_bbox[1] + revB_bbox[3]) / 2)
            if revB_bbox
            else None
        )

        result[char_no] = {
            "scores": scores,
            "reasons": reasons,
            "revA_center": revA_center,
            "revB_center": revB_center,
        }
    return result


def semantic_contract_for_item(run: Run, item: ReviewItem) -> dict | None:
    if item.char_no is None:
        return semantic_contracts_by_char(run).get(None)
    return semantic_contracts_by_char(run).get(item.char_no)


def open_review_queue(db: Session, run: Run) -> list[ReviewItem]:
    """Idempotent: create ReviewItems from delta_packet.json on first call.

    Returns all ReviewItems for this run ordered by char_no ascending.
    If ReviewItems already exist for this run, returns them without creating
    duplicates (idempotent for subsequent calls).
    """
    existing_count = db.query(ReviewItem).filter(ReviewItem.run_id == run.id).count()
    if existing_count > 0:
        return (
            db.query(ReviewItem)
            .filter(ReviewItem.run_id == run.id)
            .order_by(ReviewItem.char_no)
            .all()
        )

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

    items = []
    for delta in sorted(
        packet_data.get("items", []),
        key=lambda x: (x.get("char_no") is None, x.get("char_no") or 0),
    ):
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

    if run.status in ("completed", "warning"):
        run.status = "reviewing"
    db.commit()
    for item in items:
        db.refresh(item)
    return items


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
