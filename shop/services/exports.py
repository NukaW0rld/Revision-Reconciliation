import csv
import io
import json
from pathlib import Path
from sqlalchemy.orm import Session
from delta_preservation.types import DeltaItem
from shop.models import Run, ReviewItem, ShopConfig, User
from shop.services.semantics import shape_semantic_contract
from shop.utils import utcnow


def _load_delta_packet_items(run: Run) -> list[dict]:
    if not run.output_dir:
        return []
    packet_path = Path(run.output_dir) / "delta_packet.json"
    if not packet_path.exists():
        return []
    packet_data = json.loads(packet_path.read_text())
    return packet_data.get("items", [])


def semantic_contracts_by_char(run: Run) -> dict[int | None, dict]:
    shaped: dict[int | None, dict] = {}
    for raw_item in _load_delta_packet_items(run):
        char_no = raw_item.get("char_no")
        shaped[char_no] = shape_semantic_contract(DeltaItem.model_validate(raw_item))
    return shaped


def generate_audit_packet_csv(db: Session, run: Run) -> io.StringIO:
    """Generate audit packet CSV. Returns StringIO ready for StreamingResponse.

    CRITICAL: caller must NOT seek(0) — already positioned at start on return.
    """
    items = (
        db.query(ReviewItem)
        .filter(ReviewItem.run_id == run.id)
        .order_by(ReviewItem.char_no)
        .all()
    )
    output = io.StringIO()
    semantic_by_char = semantic_contracts_by_char(run)
    fieldnames = [
        "char_no", "requirement_revA", "requirement_revB",
        "pipeline_classification", "reviewer_decision",
        "override_note", "reviewer_name", "reviewed_at",
        "semantic_family", "semantic_status", "semantic_summary", "semantic_reason_summary",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for item in items:
        reviewer_name = ""
        if item.reviewed_by_id:
            user = db.get(User, item.reviewed_by_id)
            reviewer_name = user.username if user else ""
        semantic = semantic_by_char.get(item.char_no)
        writer.writerow({
            "char_no": item.char_no if item.char_no is not None else "",
            "requirement_revA": item.requirement_revA or "",
            "requirement_revB": item.requirement_revB or "",
            "pipeline_classification": item.pipeline_classification,
            "reviewer_decision": item.reviewer_decision or "",
            "override_note": item.override_note or "",
            "reviewer_name": reviewer_name,
            "reviewed_at": item.reviewed_at.strftime("%Y-%m-%d %H:%M UTC") if item.reviewed_at else "",
            "semantic_family": semantic["family_label"] if semantic else "",
            "semantic_status": semantic["status_label"] if semantic else "",
            "semantic_summary": semantic["summary"] if semantic else "",
            "semantic_reason_summary": semantic["reason_summary"] if semantic else "",
        })
    output.seek(0)
    return output


def render_audit_packet_pdf(db: Session, run: Run, shop_config: ShopConfig) -> bytes:
    """Render audit packet to PDF bytes using WeasyPrint.

    base_url is set to run.output_dir/snippets/ so relative <img src="...">
    paths resolve to the snippet PNG files. Raises on failure.
    """
    from weasyprint import HTML
    from shop.app import templates  # Jinja2 environment

    items = (
        db.query(ReviewItem)
        .filter(ReviewItem.run_id == run.id)
        .order_by(ReviewItem.char_no)
        .all()
    )
    signed_by_name = ""
    semantic_by_char = semantic_contracts_by_char(run)
    if run.signed_by_id:
        user = db.get(User, run.signed_by_id)
        signed_by_name = user.username if user else ""

    html_string = templates.env.get_template("exports/audit_packet.html").render(
        run=run,
        items=items,
        semantic_by_char=semantic_by_char,
        shop_config=shop_config,
        signed_by_name=signed_by_name,
    )
    # base_url set to output_dir/snippets/ so basename img paths resolve
    snippets_dir = Path(run.output_dir) / "snippets" if run.output_dir else Path(".")
    return HTML(string=html_string, base_url=str(snippets_dir)).write_pdf()


def generate_and_store_audit_packet(db: Session, run: Run) -> None:
    """Generate audit packet PDF and persist path in run.packet_versions.

    Raises on failure — caller (attempt_sign_off) catches and rolls back.
    Stores packet at: output_dir/packets/v<version>.pdf
    """
    shop_config = db.query(ShopConfig).filter(ShopConfig.id == 1).first()

    # Determine version number (1-based; amendments add v2, v3, etc.)
    existing = run.packet_versions or []
    version = len(existing) + 1

    pdf_bytes = render_audit_packet_pdf(db, run, shop_config)

    # Write to output_dir/packets/v<N>.pdf
    packets_dir = Path(run.output_dir) / "packets"
    packets_dir.mkdir(parents=True, exist_ok=True)
    packet_path = packets_dir / f"v{version}.pdf"
    packet_path.write_bytes(pdf_bytes)

    # Persist version metadata in packet_versions JSON column
    new_entry = {
        "version": version,
        "type": "original" if version == 1 else "amendment",
        "path": str(packet_path),
        "signed_at": run.signed_at.isoformat() if run.signed_at else utcnow().isoformat(),
    }
    # Reassign list (SQLAlchemy detects JSON column mutation via assignment, not append)
    run.packet_versions = existing + [new_entry]
    db.add(run)
    # Do NOT commit here — attempt_sign_off controls the transaction


# ---------------------------------------------------------------------------
# Work order helpers
# ---------------------------------------------------------------------------

def _effective_classification(item: "ReviewItem") -> str:
    """Return the effective classification after reviewer decision."""
    if item.reviewer_decision == "overridden" and item.override_classification:
        return item.override_classification
    return item.pipeline_classification


def _work_order_rows(db: Session, run: Run) -> list[dict]:
    """Return work order data rows for changed and added characteristics.

    Each dict contains:
      char_no, priority, requirement_revA, requirement_revB,
      drawing_reference, confidence, override_note
    """
    items = (
        db.query(ReviewItem)
        .filter(ReviewItem.run_id == run.id)
        .order_by(ReviewItem.char_no)
        .all()
    )
    semantic_by_char = semantic_contracts_by_char(run)
    rows = []
    for item in items:
        eff = _effective_classification(item)
        if eff not in ("changed", "added"):
            continue
        priority = "RE-MEASURE" if eff == "changed" else "NEW"
        drawing_ref = f"Balloon {item.char_no}" if item.char_no is not None else "—"
        override_note = (
            item.override_note
            if item.reviewer_decision == "overridden" and item.override_note
            else ""
        )
        semantic = semantic_by_char.get(item.char_no)
        rows.append({
            "char_no": item.char_no,
            "priority": priority,
            "requirement_revA": item.requirement_revA or "",
            "requirement_revB": item.requirement_revB or "",
            "drawing_reference": drawing_ref,
            "confidence": f"{item.confidence:.2f}",
            "override_note": override_note,
            "semantic": semantic,
            "semantic_summary": semantic["summary"] if semantic else "",
            "semantic_reason_summary": semantic["reason_summary"] if semantic else "",
            "semantic_status": semantic["status_label"] if semantic else "",
            "semantic_family": semantic["family_label"] if semantic else "",
        })
    return rows


def generate_work_order_csv(db: Session, run: Run) -> io.StringIO:
    """Generate work order CSV. Returns StringIO seeked to 0."""
    rows = _work_order_rows(db, run)
    output = io.StringIO()
    fieldnames = [
        "char_no", "priority",
        "requirement_revA", "requirement_revB",
        "drawing_reference", "confidence", "override_note",
        "semantic_family", "semantic_status", "semantic_summary", "semantic_reason_summary",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(
        {
            key: value
            for key, value in row.items()
            if key in fieldnames
        }
        for row in rows
    )
    output.seek(0)
    return output


def generate_work_order_pdf(db: Session, run: Run) -> bytes:
    """Render work order to PDF bytes using WeasyPrint. Raises on failure."""
    from weasyprint import HTML
    from shop.app import templates
    rows = _work_order_rows(db, run)

    class _Row:
        def __init__(self, d):
            self.__dict__.update(d)
            self.char_no = d["char_no"]
            self.priority = d["priority"]
            self.requirement_revA = d["requirement_revA"]
            self.requirement_revB = d["requirement_revB"]
            self.drawing_reference = d["drawing_reference"]
            self.confidence = d["confidence"]

    all_rows = [_Row(r) for r in rows]
    remeasure_items = [r for r in all_rows if r.priority == "RE-MEASURE"]
    new_items = [r for r in all_rows if r.priority == "NEW"]

    html_string = templates.env.get_template("exports/work_order.html").render(
        run=run,
        remeasure_items=remeasure_items,
        new_items=new_items,
        generated_at=utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    )
    # No images in work order — base_url irrelevant
    return HTML(string=html_string, base_url=".").write_pdf()
