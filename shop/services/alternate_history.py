import json
from pathlib import Path

from sqlalchemy.orm import Session

from delta_preservation.types import DeltaItem
from shop.models import AcceptedAlternateHistory, ReviewItem, Run
from shop.utils import utcnow


class AlternateHistorySyncError(ValueError):
    """Raised when a review item cannot be synchronized to alternate history."""


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _load_packet_row_for_item(db: Session, run: Run, item: ReviewItem) -> DeltaItem:
    if not run.output_dir:
        raise AlternateHistorySyncError("Run output directory is not configured.")

    packet_path = Path(run.output_dir) / "delta_packet.json"
    packet_data = json.loads(packet_path.read_text())
    raw_items = packet_data.get("items")
    if not isinstance(raw_items, list):
        raise AlternateHistorySyncError("delta_packet.json must contain an items array.")

    review_items = (
        db.query(ReviewItem)
        .filter(ReviewItem.run_id == run.id)
        .order_by(ReviewItem.id)
        .all()
    )
    if len(review_items) != len(raw_items):
        raise AlternateHistorySyncError(
            "Review queue and delta packet row counts do not align for alternate history."
        )

    for review_item, raw_item in zip(review_items, raw_items):
        if review_item.id == item.id:
            return DeltaItem.model_validate(raw_item)

    raise AlternateHistorySyncError(f"Review item {item.id} is not present in the packet order.")


def _ordered_mismatch_codes(delta_item: DeltaItem) -> list[str]:
    if delta_item.evaluation is None:
        return []
    return [
        mismatch.code
        for mismatch in delta_item.evaluation.mismatches
        if getattr(mismatch, "code", None)
    ]


def _deactivate_active_rows(db: Session, run_id: int, review_item_id: int) -> None:
    now = utcnow()
    active_rows = (
        db.query(AcceptedAlternateHistory)
        .filter(
            AcceptedAlternateHistory.run_id == run_id,
            AcceptedAlternateHistory.review_item_id == review_item_id,
            AcceptedAlternateHistory.is_active.is_(True),
        )
        .all()
    )
    for row in active_rows:
        row.is_active = False
        row.superseded_at = now


def sync_accepted_alternate_history(
    db: Session,
    run: Run,
    item: ReviewItem,
    payload: dict,
) -> AcceptedAlternateHistory | None:
    """Persist or deactivate accepted alternate history for one debug verdict."""
    verdict = payload.get("verdict")
    if verdict != "acceptable_alternate":
        _deactivate_active_rows(db, run.id, item.id)
        db.commit()
        return None

    delta_item = _load_packet_row_for_item(db, run, item)
    matched_truth_char_no = None
    if delta_item.evaluation is not None and delta_item.evaluation.matched_truth_char_no is not None:
        matched_truth_char_no = str(delta_item.evaluation.matched_truth_char_no)

    active_rows = (
        db.query(AcceptedAlternateHistory)
        .filter(
            AcceptedAlternateHistory.run_id == run.id,
            AcceptedAlternateHistory.review_item_id == item.id,
            AcceptedAlternateHistory.is_active.is_(True),
        )
        .order_by(AcceptedAlternateHistory.id.asc())
        .all()
    )
    history_row = active_rows[0] if active_rows else AcceptedAlternateHistory(
        run_id=run.id,
        review_item_id=item.id,
    )

    for stale_row in active_rows[1:]:
        stale_row.is_active = False
        stale_row.superseded_at = utcnow()

    history_row.reviewed_by_id = item.reviewed_by_id
    history_row.part_number = run.part_number
    history_row.char_no = item.char_no
    history_row.matched_truth_char_no = matched_truth_char_no
    history_row.reviewed_classification = (
        payload.get("corrected_classification")
        or item.pipeline_classification
    )
    history_row.reviewed_requirement_revB = _normalize_optional_text(
        payload.get("corrected_requirement_revB") or item.requirement_revB
    )
    history_row.mismatch_codes = _ordered_mismatch_codes(delta_item)
    history_row.rationale = payload["explanation"]
    history_row.is_active = True
    history_row.superseded_at = None

    db.add(history_row)
    db.commit()
    db.refresh(history_row)
    return history_row
