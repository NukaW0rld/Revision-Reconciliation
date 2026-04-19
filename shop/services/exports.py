import csv
import io
import json
from pathlib import Path
from sqlalchemy.orm import Session
from delta_preservation.types import DeltaItem
from shop.models import Run, ReviewItem, ShopConfig, User
from shop.services.semantics import shape_semantic_contract
from shop.utils import utcnow


def _load_signed_debug_snapshot(run: Run, version: int | None = None) -> dict:
    """Load the signed debug snapshot captured at sign-off time.

    Resolves the matching packet_versions entry and reads the stored
    debug_snapshot_path JSON file.  Returns the captured payload plus
    version metadata.

    Raises ValueError if:
    - no packet_versions entries exist,
    - no matching entry for the requested version, or
    - the debug_snapshot_path is missing from the entry or the file is
      unreadable.
    """
    versions = run.packet_versions or []
    if not versions:
        raise ValueError("Run has no packet_versions entries")

    # Resolve target version: default to the latest (highest version number)
    if version is None:
        entry = max(versions, key=lambda v: v.get("version", 0))
    else:
        entry = next((v for v in versions if v.get("version") == version), None)
        if entry is None:
            raise ValueError(
                f"No packet_versions entry for version={version}"
            )

    snapshot_path_str = entry.get("debug_snapshot_path")
    if not snapshot_path_str:
        raise ValueError(
            f"packet_versions entry v{entry.get('version')} has no debug_snapshot_path"
        )

    snapshot_path = Path(snapshot_path_str)
    if not snapshot_path.exists():
        raise ValueError(
            f"Signed debug snapshot not found at {snapshot_path_str}"
        )

    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            f"Could not read signed debug snapshot: {exc}"
        ) from exc

    return {
        "version": entry.get("version"),
        "debug_snapshot_path": snapshot_path_str,
        "debug_total": entry.get("debug_total", payload.get("debug_total", 0)),
        "unresolved_exception_count": entry.get("unresolved_exception_count", 0),
        "payload": payload,
    }


def _snapshot_advisory_by_item_id(snapshot_payload: dict) -> dict[int, dict]:
    """Build a lookup from review_item_id -> advisory/debug state from a signed snapshot payload.

    Extracts confidence_flags and debug state (row_state, debug_verdict) for each
    snapshot row that has a non-None review_item_id.  Synthetic missing-added rows
    (review_item_id=None) are excluded — they belong in the debug summary only.
    """
    result: dict[int, dict] = {}
    for row in snapshot_payload.get("rows", []) or snapshot_payload.get("items", []):
        item_id = row.get("review_item_id")
        if item_id is None:
            continue
        packet_item = row.get("packet_item") or {}
        confidence_flags = list(
            packet_item.get("confidence_flags", None) or []
        )
        result[item_id] = {
            "confidence_flags": confidence_flags,
            "row_state": row.get("row_state", ""),
            "debug_verdict": row.get("debug_verdict", ""),
        }
    return result


def _load_delta_packet_items(run: Run) -> list[dict]:
    if not run.output_dir:
        return []
    packet_path = Path(run.output_dir) / "delta_packet.json"
    if not packet_path.exists():
        return []
    try:
        packet_data = json.loads(packet_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return packet_data.get("items", [])


def semantic_contracts_by_char(run: Run) -> dict[int | None, dict]:
    shaped: dict[int | None, dict] = {}
    for raw_item in _load_delta_packet_items(run):
        char_no = raw_item.get("char_no")
        shaped[char_no] = shape_semantic_contract(DeltaItem.model_validate(raw_item))
    return shaped


def generate_audit_packet_csv(db: Session, run: Run) -> io.StringIO:
    """Generate audit packet CSV. Returns StringIO ready for StreamingResponse.

    Merges signed debug snapshot state (confidence_flags, debug_row_state,
    debug_verdict) onto each review row by ReviewItem.id when a signed snapshot
    is available.  Falls back gracefully to empty advisory columns when the
    snapshot is absent (e.g. legacy runs or test fixtures without output_dir).

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

    # Load signed snapshot advisory state (best-effort; empty if unavailable)
    advisory_by_item_id: dict[int, dict] = {}
    signed_debug_summary: dict = {}
    try:
        snap = _load_signed_debug_snapshot(run)
        advisory_by_item_id = _snapshot_advisory_by_item_id(snap["payload"])
        signed_debug_summary = {
            "debug_total": snap["debug_total"],
            "unresolved_exception_count": snap["unresolved_exception_count"],
            "version": snap["version"],
        }
    except ValueError:
        pass

    fieldnames = [
        "char_no", "requirement_revA", "requirement_revB",
        "pipeline_classification", "reviewer_decision",
        "override_note", "reviewer_name", "reviewed_at",
        "semantic_family", "semantic_status", "semantic_summary", "semantic_reason_summary",
        "confidence_flags", "debug_row_state", "debug_verdict",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for item in items:
        reviewer_name = ""
        if item.reviewed_by_id:
            user = db.get(User, item.reviewed_by_id)
            reviewer_name = user.username if user else ""
        semantic = semantic_by_char.get(item.char_no)
        advisory = advisory_by_item_id.get(item.id, {})
        confidence_flags = advisory.get("confidence_flags", [])
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
            "confidence_flags": "; ".join(confidence_flags) if confidence_flags else "",
            "debug_row_state": advisory.get("row_state", ""),
            "debug_verdict": advisory.get("debug_verdict", "") or "",
        })
    output.seek(0)
    return output


def render_audit_packet_pdf(db: Session, run: Run, shop_config: ShopConfig) -> bytes:
    """Render audit packet to PDF bytes using WeasyPrint.

    Passes signed snapshot advisory state (confidence_flags, debug_row_state,
    debug_verdict) and a signed debug summary to the template so the PDF
    reflects the exact advisory/gating state cleared at sign-off time.

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

    # Load signed snapshot advisory state (best-effort)
    advisory_by_item_id: dict[int, dict] = {}
    signed_debug_summary: dict = {}
    try:
        snap = _load_signed_debug_snapshot(run)
        advisory_by_item_id = _snapshot_advisory_by_item_id(snap["payload"])
        signed_debug_summary = {
            "debug_total": snap["debug_total"],
            "unresolved_exception_count": snap["unresolved_exception_count"],
            "version": snap["version"],
        }
    except ValueError:
        pass

    html_string = templates.env.get_template("exports/audit_packet.html").render(
        run=run,
        items=items,
        semantic_by_char=semantic_by_char,
        shop_config=shop_config,
        signed_by_name=signed_by_name,
        advisory_by_item_id=advisory_by_item_id,
        signed_debug_summary=signed_debug_summary,
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


def _work_order_rows(db: Session, run: Run, advisory_by_item_id: dict | None = None) -> list[dict]:
    """Return work order data rows for changed and added characteristics.

    Each dict contains:
      char_no, priority, requirement_revA, requirement_revB,
      drawing_reference, confidence, override_note, confidence_flags

    Synthetic missing-added truth rows (review_item_id=None in the debug
    snapshot) are intentionally excluded — they belong in the audit/debug
    summary context, not the work-order action list.

    advisory_by_item_id: optional dict from _snapshot_advisory_by_item_id;
    when provided, confidence_flags from the signed snapshot are merged onto
    each actionable row keyed by ReviewItem.id.
    """
    items = (
        db.query(ReviewItem)
        .filter(ReviewItem.run_id == run.id)
        .order_by(ReviewItem.char_no)
        .all()
    )
    semantic_by_char = semantic_contracts_by_char(run)
    adv = advisory_by_item_id or {}
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
        item_advisory = adv.get(item.id, {})
        confidence_flags = item_advisory.get("confidence_flags", [])
        rows.append({
            "char_no": item.char_no,
            "priority": priority,
            "requirement_revA": item.requirement_revA or "",
            "requirement_revB": item.requirement_revB or "",
            "drawing_reference": drawing_ref,
            "confidence": f"{item.confidence:.2f}",
            "override_note": override_note,
            "confidence_flags": confidence_flags,
            "semantic": semantic,
            "semantic_summary": semantic["summary"] if semantic else "",
            "semantic_reason_summary": semantic["reason_summary"] if semantic else "",
            "semantic_status": semantic["status_label"] if semantic else "",
            "semantic_family": semantic["family_label"] if semantic else "",
        })
    return rows


def _load_signed_debug_summary(run: Run) -> dict:
    """Return signed debug summary dict for use in export context.

    Returns empty dict if snapshot is unavailable (legacy runs, test fixtures
    without output_dir, or runs signed before Phase 10).
    """
    try:
        snap = _load_signed_debug_snapshot(run)
        return {
            "debug_total": snap["debug_total"],
            "unresolved_exception_count": snap["unresolved_exception_count"],
            "version": snap["version"],
        }
    except ValueError:
        return {}


def generate_work_order_csv(db: Session, run: Run) -> io.StringIO:
    """Generate work order CSV. Returns StringIO seeked to 0.

    Merges signed snapshot confidence_flags onto each actionable row.
    Synthetic missing-added truth rows are excluded (they are not work-order tasks).
    """
    # Load signed snapshot advisory state (best-effort)
    advisory_by_item_id: dict[int, dict] = {}
    try:
        snap = _load_signed_debug_snapshot(run)
        advisory_by_item_id = _snapshot_advisory_by_item_id(snap["payload"])
    except ValueError:
        pass

    rows = _work_order_rows(db, run, advisory_by_item_id=advisory_by_item_id)
    output = io.StringIO()
    fieldnames = [
        "char_no", "priority",
        "requirement_revA", "requirement_revB",
        "drawing_reference", "confidence", "override_note",
        "confidence_flags",
        "semantic_family", "semantic_status", "semantic_summary", "semantic_reason_summary",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(
        {
            key: ("; ".join(value) if isinstance(value, list) else value)
            for key, value in row.items()
            if key in fieldnames
        }
        for row in rows
    )
    output.seek(0)
    return output


def generate_work_order_pdf(db: Session, run: Run) -> bytes:
    """Render work order to PDF bytes using WeasyPrint.

    Passes signed snapshot advisory flags and a compact signed debug summary
    to the work-order template.  Synthetic missing-added truth rows are kept
    out of the actionable row list (they are debug/audit summary context only).
    Raises on failure.
    """
    from weasyprint import HTML
    from shop.app import templates

    # Load signed snapshot advisory state (best-effort)
    advisory_by_item_id: dict[int, dict] = {}
    signed_debug_summary: dict = {}
    try:
        snap = _load_signed_debug_snapshot(run)
        advisory_by_item_id = _snapshot_advisory_by_item_id(snap["payload"])
        signed_debug_summary = {
            "debug_total": snap["debug_total"],
            "unresolved_exception_count": snap["unresolved_exception_count"],
            "version": snap["version"],
        }
    except ValueError:
        pass

    rows = _work_order_rows(db, run, advisory_by_item_id=advisory_by_item_id)

    class _Row:
        def __init__(self, d):
            self.__dict__.update(d)
            self.char_no = d["char_no"]
            self.priority = d["priority"]
            self.requirement_revA = d["requirement_revA"]
            self.requirement_revB = d["requirement_revB"]
            self.drawing_reference = d["drawing_reference"]
            self.confidence = d["confidence"]
            self.override_note = d.get("override_note", "")
            self.confidence_flags = d.get("confidence_flags", [])
            self.semantic = d.get("semantic")
            self.semantic_summary = d.get("semantic_summary", "")
            self.semantic_reason_summary = d.get("semantic_reason_summary", "")
            self.semantic_status = d.get("semantic_status", "")
            self.semantic_family = d.get("semantic_family", "")

    all_rows = [_Row(r) for r in rows]
    remeasure_items = [r for r in all_rows if r.priority == "RE-MEASURE"]
    new_items = [r for r in all_rows if r.priority == "NEW"]

    html_string = templates.env.get_template("exports/work_order.html").render(
        run=run,
        remeasure_items=remeasure_items,
        new_items=new_items,
        signed_debug_summary=signed_debug_summary,
        generated_at=utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    )
    # No images in work order — base_url irrelevant
    return HTML(string=html_string, base_url=".").write_pdf()
