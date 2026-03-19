import json
from pathlib import Path
from sqlalchemy.orm import Session
from shop.models import Run, ReviewItem, User
from shop.utils import utcnow


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
    packet_path = Path(run.output_dir) / "delta_packet.json"
    packet_data = json.loads(packet_path.read_text())

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
            requirement_revB=None,  # Rev B requirement text: Phase 4 concern
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
