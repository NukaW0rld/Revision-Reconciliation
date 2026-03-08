import csv
import io
import json
from datetime import datetime
from pathlib import Path
from sqlalchemy.orm import Session
from shop.models import Run, ReviewItem, ShopConfig, User


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
    fieldnames = [
        "char_no", "requirement_revA", "requirement_revB",
        "pipeline_classification", "reviewer_decision",
        "override_note", "reviewer_name", "reviewed_at",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for item in items:
        reviewer_name = ""
        if item.reviewed_by_id:
            user = db.get(User, item.reviewed_by_id)
            reviewer_name = user.email if user else ""
        writer.writerow({
            "char_no": item.char_no if item.char_no is not None else "",
            "requirement_revA": item.requirement_revA or "",
            "requirement_revB": item.requirement_revB or "",
            "pipeline_classification": item.pipeline_classification,
            "reviewer_decision": item.reviewer_decision or "",
            "override_note": item.override_note or "",
            "reviewer_name": reviewer_name,
            "reviewed_at": item.reviewed_at.strftime("%Y-%m-%d %H:%M UTC") if item.reviewed_at else "",
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
    if run.signed_by_id:
        user = db.get(User, run.signed_by_id)
        signed_by_name = user.email if user else ""

    html_string = templates.env.get_template("exports/audit_packet.html").render(
        run=run,
        items=items,
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
        "signed_at": run.signed_at.isoformat() if run.signed_at else datetime.utcnow().isoformat(),
    }
    # Reassign list (SQLAlchemy detects JSON column mutation via assignment, not append)
    run.packet_versions = existing + [new_entry]
    db.add(run)
    # Do NOT commit here — attempt_sign_off controls the transaction
