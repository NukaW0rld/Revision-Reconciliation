"""Amendment service: clone a signed-off run as an amendment.

Amendment runs share the parent's output_dir, revA/revB/form3 paths,
and pipeline outputs (delta_packet.json, snippets/). Only review decisions
can be changed in an amendment.

The amendment's packet_versions is initialized from the parent's list so
generate_and_store_audit_packet() computes the correct next version number.
"""
from sqlalchemy.orm import Session
from shop.models import Run, ReviewItem


def create_amendment(db: Session, parent_run: Run, initiator_id: int) -> Run:
    """Clone parent_run as an amendment. Returns the new amendment Run.

    The returned amendment has status='reviewing' and ReviewItems pre-filled
    from the parent's final decisions. open_review_queue() will short-circuit
    because existing_count > 0 immediately after creation.
    """
    # Copy parent's packet_versions so version numbering continues correctly
    # (amendment will produce v2, v3, etc. relative to the parent's history)
    inherited_versions = list(parent_run.packet_versions or [])

    amendment = Run(
        part_number=parent_run.part_number,
        rev_a_label=parent_run.rev_a_label,
        rev_b_label=parent_run.rev_b_label,
        customer=parent_run.customer,
        job_number=parent_run.job_number,
        status="reviewing",
        # Share parent's pipeline output and input files (AMEND-02: files locked)
        output_dir=parent_run.output_dir,
        revA_path=parent_run.revA_path,
        revB_path=parent_run.revB_path,
        form3_path=parent_run.form3_path,
        revA_page=parent_run.revA_page,
        revB_page=parent_run.revB_page,
        reviewer_id=initiator_id,
        parent_run_id=parent_run.id,
        # Inherit packet versions so v2 is computed at sign-off
        packet_versions=inherited_versions,
    )
    db.add(amendment)
    db.flush()  # get amendment.id before cloning items

    # Clone ReviewItems with existing decisions pre-filled (AMEND-01)
    parent_items = (
        db.query(ReviewItem)
        .filter(ReviewItem.run_id == parent_run.id)
        .order_by(ReviewItem.char_no)
        .all()
    )
    for item in parent_items:
        clone = ReviewItem(
            run_id=amendment.id,
            char_no=item.char_no,
            pipeline_classification=item.pipeline_classification,
            confidence=item.confidence,
            requirement_revA=item.requirement_revA,
            requirement_revB=item.requirement_revB,
            revA_snippet_path=item.revA_snippet_path,
            revB_snippet_path=item.revB_snippet_path,
            revA_bbox=item.revA_bbox,
            revB_bbox=item.revB_bbox,
            reviewer_decision=item.reviewer_decision,            # pre-filled
            override_classification=item.override_classification,
            override_note=item.override_note,
            reviewed_by_id=item.reviewed_by_id,
            reviewed_at=item.reviewed_at,
        )
        db.add(clone)

    db.commit()
    db.refresh(amendment)
    return amendment
