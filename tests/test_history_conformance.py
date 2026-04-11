from sqlalchemy.orm import sessionmaker

from delta_preservation.evaluation.conformance import apply_accepted_alternate_history
from delta_preservation.types import DeltaItem, ItemEvaluation
from shop.models import AcceptedAlternateHistory, ReviewItem, Run
from shop.services.alternate_history import load_active_accepted_alternates


def _seed_history_record(
    db_engine,
    *,
    part_number: str,
    char_no: int = 42,
    matched_truth_char_no: str | None = "truth-42",
    reviewed_classification: str = "changed",
    reviewed_requirement_revB: str | None = "REV B REQUIREMENT",
    mismatch_codes: list[str] | None = None,
) -> tuple[int, int]:
    session = sessionmaker(bind=db_engine)()
    try:
        run = Run(
            part_number=part_number,
            rev_a_label="A",
            rev_b_label="B",
            customer="Test",
            job_number="J-HIST",
            status="completed",
            output_dir="/tmp/history",
            revA_path="/tmp/a.pdf",
            revB_path="/tmp/b.pdf",
            form3_path="/tmp/f.pdf",
        )
        session.add(run)
        session.commit()
        session.refresh(run)

        item = ReviewItem(
            run_id=run.id,
            char_no=char_no,
            pipeline_classification=reviewed_classification,
            confidence=0.9,
            requirement_revB=reviewed_requirement_revB,
        )
        session.add(item)
        session.commit()
        session.refresh(item)

        history = AcceptedAlternateHistory(
            run_id=run.id,
            review_item_id=item.id,
            reviewed_by_id=None,
            part_number=part_number,
            char_no=char_no,
            matched_truth_char_no=matched_truth_char_no,
            reviewed_classification=reviewed_classification,
            reviewed_requirement_revB=reviewed_requirement_revB,
            mismatch_codes=mismatch_codes or ["classification_mismatch", "requirement_mismatch"],
            rationale="Approved alternate history",
            is_active=True,
        )
        session.add(history)
        session.commit()
        session.refresh(history)
        return history.id, run.id
    finally:
        session.close()


def _load_part_history(db_engine, part_number: str):
    session = sessionmaker(bind=db_engine)()
    try:
        return load_active_accepted_alternates(session, part_number)
    finally:
        session.close()


def _make_item(*, requirement_revB: str | None = "REV B REQUIREMENT") -> DeltaItem:
    return DeltaItem.model_validate(
        {
            "char_no": 42,
            "status": "changed",
            "confidence": 0.73,
            "reasons": ["classification differs", "requirement differs"],
            "scores": {"location": 0.3, "text": 0.8, "context": 0.5},
            "revA": None,
            "revB": None,
            "requirement_revB": requirement_revB,
        }
    )


def _make_evaluation(*, mismatch_codes: list[str] | None = None) -> ItemEvaluation:
    return ItemEvaluation.model_validate(
        {
            "status": "review_needed",
            "matched_truth_char_no": "truth-42",
            "classification_conforms": False,
            "requirement_conforms": False,
            "snippet_conforms": False,
            "mismatches": [
                {"code": code, "message": f"{code} detected"}
                for code in (mismatch_codes or ["classification_mismatch", "requirement_mismatch"])
            ],
        }
    )


def test_later_run_uses_active_history_record_to_auto_conform(db_engine):
    history_id, source_run_id = _seed_history_record(db_engine, part_number="PN-DBG")
    approved_alternates = _load_part_history(db_engine, "PN-DBG")

    [updated] = apply_accepted_alternate_history(
        [_make_item()],
        [_make_evaluation()],
        approved_alternates,
    )

    assert updated.status == "conforming"
    assert updated.conformance_source == "accepted_alternate"
    assert updated.history_reference is not None
    assert updated.history_reference.history_id == history_id
    assert updated.history_reference.source_run_id == source_run_id


def test_history_reuse_requires_matching_mismatch_fingerprint(db_engine):
    _seed_history_record(
        db_engine,
        part_number="PN-DBG",
        mismatch_codes=["classification_mismatch", "requirement_mismatch"],
    )
    approved_alternates = _load_part_history(db_engine, "PN-DBG")

    [updated] = apply_accepted_alternate_history(
        [_make_item()],
        [_make_evaluation(mismatch_codes=["classification_mismatch"])],
        approved_alternates,
    )

    assert updated.status == "review_needed"
    assert updated.conformance_source == "ground_truth"
    assert updated.history_reference is None


def test_cross_part_history_record_is_ignored(db_engine):
    _seed_history_record(db_engine, part_number="PN-OTHER")
    approved_alternates = _load_part_history(db_engine, "PN-DBG")

    [updated] = apply_accepted_alternate_history(
        [_make_item()],
        [_make_evaluation()],
        approved_alternates,
    )

    assert approved_alternates == []
    assert updated.status == "review_needed"
    assert updated.history_reference is None
