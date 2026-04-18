"""Phase 4: Audit packet and work order export tests."""
import csv
import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from shop.services.exports import semantic_contracts_by_char


# conftest provides: client, admin_user, engineer_user fixtures


def _login_engineer(client, db_engine, engineer_user):
    """Seed a session for the engineer user and set cookie on client."""
    from shop.services.auth import create_session
    Session = sessionmaker(bind=db_engine)
    db = Session()
    token = create_session(db, engineer_user)
    db.close()
    client.cookies.set("session_token", token)
    return token


SEMANTIC_FIXTURE_ITEMS = [
    {
        "char_no": 21,
        "status": "changed",
        "confidence": 0.83,
        "reasons": ["semantic GD&T changed: tolerance ⌀0.10 → ⌀0.20"],
        "scores": {"location": 0.88, "text": 0.71, "context": 0.84},
        "revA": None,
        "revB": None,
        "semantic_callout": {
            "provenance": {
                "authority": "pdf",
                "source_type": "drawing_pdf",
                "source_ref": "page:1/span:4",
                "notes": [],
            },
            "status": {
                "state": "parsed",
                "parser_family": "gdt",
                "reason_code": None,
                "detail": "parsed feature control frame from PDF spans",
            },
            "raw_text": "⌖ ⌀0.20 M A B C",
            "normalized_text": "⌖ ⌀0.20 M A B C",
            "gdt": {
                "frame_text": "⌖ | ⌀0.20 | M | A | B | C",
                "control_type": "position",
                "tolerance_text": "⌀0.20",
                "datum_refs": ["A", "B", "C"],
                "modifiers": ["MMC"],
            },
            "metadata": {},
        },
    },
    {
        "char_no": 22,
        "status": "changed",
        "confidence": 0.79,
        "reasons": ["semantic weld changed: fillet size 1/4 → 3/8"],
        "scores": {"location": 0.82, "text": 0.76, "context": 0.74},
        "revA": None,
        "revB": None,
        "semantic_callout": {
            "provenance": {
                "authority": "pdf",
                "source_type": "drawing_pdf",
                "source_ref": "page:1/span:7",
                "notes": [],
            },
            "status": {
                "state": "parsed",
                "parser_family": "weld",
                "reason_code": None,
                "detail": "parsed weld symbol from callout text",
            },
            "raw_text": "FILLET 3/8 BOTH SIDES",
            "normalized_text": "FILLET 3/8 BOTH SIDES",
            "weld": {
                "process": "fillet",
                "size": "3/8",
                "side": "both sides",
                "all_around": False,
                "length": None,
                "pitch": None,
                "contour": None,
                "tail": None,
            },
            "metadata": {},
        },
    },
    {
        "char_no": 23,
        "status": "unchanged",
        "confidence": 0.91,
        "reasons": ["semantic surface finish matched: Ra 63"],
        "scores": {"location": 0.89, "text": 0.87, "context": 0.9},
        "revA": None,
        "revB": None,
        "semantic_callout": {
            "provenance": {
                "authority": "pdf",
                "source_type": "drawing_pdf",
                "source_ref": "page:2/span:2",
                "notes": [],
            },
            "status": {
                "state": "parsed",
                "parser_family": "surface_finish",
                "reason_code": None,
                "detail": "parsed surface finish requirement",
            },
            "raw_text": "⌯ Ra 63",
            "normalized_text": "⌯ Ra 63",
            "surface_finish": {
                "canonical_text": "Ra 63",
                "parameter": "Ra",
                "value": "63",
                "units": None,
            },
            "metadata": {},
        },
    },
    {
        "char_no": 24,
        "status": "added",
        "confidence": 0.77,
        "reasons": ["semantic fit added: H7/g6 hole-basis fit"],
        "scores": {"location": 0.8, "text": 0.73, "context": 0.75},
        "revA": None,
        "revB": None,
        "semantic_callout": {
            "provenance": {
                "authority": "pdf",
                "source_type": "drawing_pdf",
                "source_ref": "page:2/span:6",
                "notes": [],
            },
            "status": {
                "state": "parsed",
                "parser_family": "fit",
                "reason_code": None,
                "detail": "parsed ISO fit notation",
            },
            "raw_text": "H7/g6",
            "normalized_text": "H7/g6",
            "fit": {
                "canonical_text": "H7/g6",
                "fit_class": "H7/g6",
                "basis": "hole basis",
            },
            "metadata": {},
        },
    },
    {
        "char_no": 25,
        "status": "unchanged",
        "confidence": 0.98,
        "reasons": ["Location and text matched"],
        "scores": {"location": 0.98, "text": 0.98, "context": 0.97},
        "revA": None,
        "revB": None,
    },
    {
        "char_no": 26,
        "status": "added",
        "confidence": 0.72,
        "reasons": [
            "semantic comparison fallback: left semantic state empty/surface_finish_no_match",
            "New requirement detected in Rev B",
        ],
        "scores": {"location": 0.7, "text": 0.69, "context": 0.68},
        "revA": None,
        "revB": None,
        "semantic_callout": {
            "provenance": {
                "authority": "pdf",
                "source_type": "drawing_pdf",
                "source_ref": "page:3/span:1",
                "notes": ["pdf span selected"],
            },
            "status": {
                "state": "empty",
                "parser_family": "surface_finish",
                "reason_code": "surface_finish_no_match",
                "detail": "text did not match the bounded GD&T, weld, surface finish, or fit grammar",
            },
            "raw_text": "FLAG NOTE 12",
            "normalized_text": "FLAG NOTE 12",
            "metadata": {"authority_source": "pdf"},
        },
    },
]


def _write_minimal_debug_snapshot(output_dir: str, version: int = 1) -> str:
    """Write a minimal signed debug snapshot JSON and return its path string."""
    packets_dir = Path(output_dir) / "packets"
    packets_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = packets_dir / f"v{version}-debug-report.json"
    payload = {
        "debug_total": 0,
        "unresolved_exception_count": 0,
        "exception_items": [],
        "missing_added_truth_items": [],
        "rows": [],
    }
    snapshot_path.write_text(json.dumps(payload), encoding="utf-8")
    return str(snapshot_path)


def _make_signed_run(db, engineer_id, tmp_path=None):
    """Create a minimal signed-off Run with 2 ReviewItems for testing.

    Writes a real debug_snapshot_path so _get_signed_run passes the
    signed-snapshot contract check introduced in Phase 10-03.
    """
    from shop.models import Run, ReviewItem
    # Use a real temp dir so the snapshot file can be written
    out_dir = Path(tmp_path) / "signed-run-base" if tmp_path else Path(tempfile.mkdtemp()) / "signed-run-base"
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = _write_minimal_debug_snapshot(str(out_dir), version=1)
    run = Run(
        part_number="PN-TEST",
        rev_a_label="A",
        rev_b_label="B",
        customer="Acme",
        job_number="JOB-001",
        status="signed_off",
        output_dir=str(out_dir),
        revA_path="/tmp/revA.pdf",
        revB_path="/tmp/revB.pdf",
        form3_path="/tmp/form3.xlsx",
        reviewer_id=engineer_id,
        signed_at=datetime(2026, 3, 8, 12, 0, 0),
        signed_by_id=engineer_id,
        packet_versions=[{
            "version": 1,
            "type": "original",
            "path": str(out_dir / "packets" / "v1.pdf"),
            "signed_at": "2026-03-08T12:00:00",
            "debug_snapshot_path": snapshot_path,
            "debug_total": 0,
            "unresolved_exception_count": 0,
        }],
    )
    db.add(run)
    db.flush()
    for i in (1, 2):
        item = ReviewItem(
            run_id=run.id,
            char_no=i,
            pipeline_classification="unchanged",
            confidence=0.95,
            requirement_revA=f"Req A {i}",
            requirement_revB=f"Req B {i}",
            reviewer_decision="approved",
            reviewed_by_id=engineer_id,
            reviewed_at=datetime(2026, 3, 8, 12, 0, 0),
        )
        db.add(item)
    db.commit()
    return run


def _make_semantic_run(tmp_path, db, engineer_id, *, output_slug="semantic-export-001"):
    from shop.models import Run, ReviewItem

    out_dir = tmp_path / "out" / output_slug
    out_dir.mkdir(parents=True)
    packet = {"run_id": output_slug, "inputs": {}, "items": SEMANTIC_FIXTURE_ITEMS}
    (out_dir / "delta_packet.json").write_text(json.dumps(packet))

    run = Run(
        part_number="PN-SEM",
        rev_a_label="A",
        rev_b_label="B",
        customer="Acme",
        job_number="JOB-SEM",
        status="signed_off",
        output_dir=str(out_dir),
        revA_path="/tmp/a.pdf",
        revB_path="/tmp/b.pdf",
        form3_path="/tmp/form3.xlsx",
        reviewer_id=engineer_id,
        signed_at=datetime(2026, 3, 8, 12, 0, 0),
        signed_by_id=engineer_id,
    )
    db.add(run)
    db.flush()
    db.add_all([
        ReviewItem(
            run_id=run.id,
            char_no=21,
            pipeline_classification="changed",
            confidence=0.83,
            requirement_revA="⌖ ⌀0.10 M A B C",
            requirement_revB="⌖ ⌀0.20 M A B C",
            reviewer_decision="approved",
            reviewed_by_id=engineer_id,
            reviewed_at=datetime(2026, 3, 8, 12, 0, 0),
        ),
        ReviewItem(
            run_id=run.id,
            char_no=22,
            pipeline_classification="changed",
            confidence=0.79,
            requirement_revA="FILLET 1/4 BOTH SIDES",
            requirement_revB="FILLET 3/8 BOTH SIDES",
            reviewer_decision="approved",
            reviewed_by_id=engineer_id,
            reviewed_at=datetime(2026, 3, 8, 12, 0, 0),
        ),
        ReviewItem(
            run_id=run.id,
            char_no=23,
            pipeline_classification="unchanged",
            confidence=0.91,
            requirement_revA="⌯ Ra 63",
            requirement_revB="⌯ Ra 63",
            reviewer_decision="approved",
            reviewed_by_id=engineer_id,
            reviewed_at=datetime(2026, 3, 8, 12, 0, 0),
        ),
        ReviewItem(
            run_id=run.id,
            char_no=24,
            pipeline_classification="added",
            confidence=0.77,
            requirement_revA=None,
            requirement_revB="H7/g6",
            reviewer_decision="approved",
            reviewed_by_id=engineer_id,
            reviewed_at=datetime(2026, 3, 8, 12, 0, 0),
        ),
        ReviewItem(
            run_id=run.id,
            char_no=25,
            pipeline_classification="unchanged",
            confidence=0.98,
            requirement_revA="Stable requirement",
            requirement_revB="Stable requirement",
            reviewer_decision="approved",
            reviewed_by_id=engineer_id,
            reviewed_at=datetime(2026, 3, 8, 12, 0, 0),
        ),
        ReviewItem(
            run_id=run.id,
            char_no=26,
            pipeline_classification="added",
            confidence=0.72,
            requirement_revA=None,
            requirement_revB="FLAG NOTE 12",
            reviewer_decision="approved",
            reviewed_by_id=engineer_id,
            reviewed_at=datetime(2026, 3, 8, 12, 0, 0),
        ),
    ])
    db.commit()
    db.refresh(run)
    return run


def test_exports_semantic_contract_shapes_csv_and_work_order_rows(tmp_path, client: TestClient, engineer_user):
    """semantic_contract: export service carries parsed and fallback semantic summaries."""
    from shop.dependencies import get_db
    from shop.services.exports import generate_audit_packet_csv, _work_order_rows

    db_gen = client.app.dependency_overrides.get(get_db)
    db = next(db_gen()) if db_gen else None
    if db is None:
        pytest.skip("no DB override available")

    run = _make_semantic_run(tmp_path, db, engineer_user.id)

    contracts = semantic_contracts_by_char(run)
    assert contracts[21]["family_label"] == "GD&T"
    assert contracts[21]["summary"] == "position ⌀0.20 datums A, B, C modifiers MMC"
    assert contracts[21]["detail"] == "parsed feature control frame from PDF spans"
    assert contracts[21]["provenance"] == {
        "authority": "pdf",
        "source_type": "drawing_pdf",
        "source_ref": "page:1/span:4",
        "notes": [],
    }
    assert contracts[22]["family_label"] == "Weld"
    assert contracts[22]["summary"] == "fillet size 3/8 both sides"
    assert contracts[22]["detail"] == "parsed weld symbol from callout text"
    assert contracts[23]["family_label"] == "Surface Finish"
    assert contracts[23]["summary"] == "Ra 63"
    assert contracts[23]["detail"] == "parsed surface finish requirement"
    assert contracts[24]["family_label"] == "Fit"
    assert contracts[24]["summary"] == "H7/g6 (hole basis)"
    assert contracts[24]["detail"] == "parsed ISO fit notation"
    assert contracts[25] is None
    assert contracts[26]["status"] == "empty"
    assert contracts[26]["block_label"] == "Surface Finish empty"
    assert contracts[26]["detail"] == "text did not match the bounded GD&T, weld, surface finish, or fit grammar"
    assert contracts[26]["provenance"] == {
        "authority": "pdf",
        "source_type": "drawing_pdf",
        "source_ref": "page:3/span:1",
        "notes": ["pdf span selected"],
    }

    csv_rows = {row["char_no"]: row for row in csv.DictReader(generate_audit_packet_csv(db, run))}
    assert csv_rows["21"]["semantic_family"] == "GD&T"
    assert csv_rows["21"]["semantic_status"] == "parsed"
    assert csv_rows["21"]["semantic_summary"] == contracts[21]["summary"]
    assert csv_rows["21"]["semantic_reason_summary"] == "semantic GD&T changed: tolerance ⌀0.10 → ⌀0.20"
    assert csv_rows["22"]["semantic_summary"] == contracts[22]["summary"]
    assert csv_rows["23"]["semantic_summary"] == "Ra 63"
    assert csv_rows["24"]["semantic_summary"] == "H7/g6 (hole basis)"
    assert csv_rows["25"]["semantic_family"] == ""
    assert csv_rows["25"]["semantic_summary"] == ""
    assert csv_rows["26"]["semantic_family"] == "Surface Finish"
    assert csv_rows["26"]["semantic_status"] == "empty"
    assert csv_rows["26"]["semantic_summary"] == "FLAG NOTE 12"
    assert csv_rows["26"]["semantic_reason_summary"] == (
        "semantic comparison fallback: left semantic state empty/surface_finish_no_match"
    )

    work_rows = {str(row["char_no"]): row for row in _work_order_rows(db, run)}
    assert set(work_rows.keys()) == {"21", "22", "24", "26"}
    assert work_rows["21"]["semantic_family"] == "GD&T"
    assert work_rows["21"]["semantic_status"] == "parsed"
    assert work_rows["21"]["semantic_summary"] == contracts[21]["summary"]
    assert work_rows["21"]["semantic_reason_summary"] == contracts[21]["reason_summary"]
    assert work_rows["22"]["semantic_summary"] == contracts[22]["summary"]
    assert work_rows["24"]["semantic_reason_summary"] == "semantic fit added: H7/g6 hole-basis fit"
    assert work_rows["26"]["semantic_status"] == "empty"
    assert work_rows["26"]["semantic_family"] == "Surface Finish"
    assert work_rows["26"]["semantic_summary"] == contracts[26]["summary"]
    assert work_rows["26"]["semantic_reason_summary"] == contracts[26]["reason_summary"]


def test_exports_semantic_and_omission_render_semantic_evidence_and_omit_empty_sections(
    tmp_path, client: TestClient, engineer_user
):
    """semantic and omission: export templates show compact evidence and omit dead scaffolding."""
    from shop.dependencies import get_db
    from shop.models import ShopConfig
    from shop.services.exports import generate_work_order_pdf, render_audit_packet_pdf
    from shop.app import templates

    db_gen = client.app.dependency_overrides.get(get_db)
    db = next(db_gen()) if db_gen else None
    if db is None:
        pytest.skip("no DB override available")

    run = _make_semantic_run(tmp_path, db, engineer_user.id, output_slug="semantic-export-render-001")
    semantic_by_char = semantic_contracts_by_char(run)
    items = (
        db.query(__import__("shop.models", fromlist=["ReviewItem"]).ReviewItem)
        .filter_by(run_id=run.id)
        .order_by(__import__("shop.models", fromlist=["ReviewItem"]).ReviewItem.char_no)
        .all()
    )

    audit_html = templates.env.get_template("exports/audit_packet.html").render(
        run=run,
        items=items,
        semantic_by_char=semantic_by_char,
        shop_config=ShopConfig(shop_name="Delta Shop"),
        signed_by_name=engineer_user.username,
    )
    assert "Semantic Read" in audit_html
    assert "GD&amp;T parsed" in audit_html
    assert "position ⌀0.20 datums A, B, C modifiers MMC" in audit_html
    assert "semantic weld changed: fillet size 1/4 → 3/8" in audit_html
    assert "Surface Finish empty" in audit_html
    assert "FLAG NOTE 12" in audit_html
    no_semantic_audit = audit_html.split("Characteristic #25", 1)[1].split("Characteristic #26", 1)[0]
    assert "Semantic Read" not in no_semantic_audit

    assert render_audit_packet_pdf(db, run, ShopConfig(shop_name="Delta Shop"))
    assert generate_work_order_pdf(db, run)

    work_rows = __import__("shop.services.exports", fromlist=["_work_order_rows"])._work_order_rows(db, run)

    class _Row:
        def __init__(self, payload):
            self.__dict__.update(payload)

    remeasure_items = [_Row(r) for r in work_rows if r["priority"] == "RE-MEASURE"]
    new_items = [_Row(r) for r in work_rows if r["priority"] == "NEW"]
    work_html = templates.env.get_template("exports/work_order.html").render(
        run=run,
        remeasure_items=remeasure_items,
        new_items=new_items,
        generated_at="2026-03-08 12:00 UTC",
    )
    assert "Semantic Evidence" in work_html
    assert "GD&amp;T parsed" in work_html
    assert "fillet size 3/8 both sides" in work_html
    assert "H7/g6 (hole basis)" in work_html
    assert "Surface Finish empty" in work_html
    assert "semantic comparison fallback: left semantic state empty/surface_finish_no_match" in work_html
    assert "No changed or added characteristics" not in work_html


def test_semantic_parity_review_and_exports(client: TestClient, engineer_user, db_engine, tmp_path):
    """semantic parity: representative items expose matching semantic meaning in review and export surfaces."""
    from shop.dependencies import get_db
    from shop.services.exports import generate_audit_packet_csv, _work_order_rows
    from shop.app import templates
    from shop.models import ShopConfig

    _login_engineer(client, db_engine, engineer_user)

    db_gen = client.app.dependency_overrides.get(get_db)
    db = next(db_gen()) if db_gen else None
    if db is None:
        pytest.skip("no DB override available")

    run = _make_semantic_run(tmp_path, db, engineer_user.id, output_slug="semantic-parity-001")

    review_resp = client.get(f"/review/{run.id}", follow_redirects=False)
    assert review_resp.status_code == 200, review_resp.text
    review_html = review_resp.text

    contracts = semantic_contracts_by_char(run)
    audit_rows = {row["char_no"]: row for row in csv.DictReader(generate_audit_packet_csv(db, run))}
    work_rows = {str(row["char_no"]): row for row in _work_order_rows(db, run)}

    items = (
        db.query(__import__("shop.models", fromlist=["ReviewItem"]).ReviewItem)
        .filter_by(run_id=run.id)
        .order_by(__import__("shop.models", fromlist=["ReviewItem"]).ReviewItem.char_no)
        .all()
    )
    audit_html = templates.env.get_template("exports/audit_packet.html").render(
        run=run,
        items=items,
        semantic_by_char=contracts,
        shop_config=ShopConfig(shop_name="Delta Shop"),
        signed_by_name=engineer_user.username,
    )

    class _Row:
        def __init__(self, payload):
            self.__dict__.update(payload)

    render_rows = [_Row(r) for r in _work_order_rows(db, run)]
    work_html = templates.env.get_template("exports/work_order.html").render(
        run=run,
        remeasure_items=[r for r in render_rows if r.priority == "RE-MEASURE"],
        new_items=[r for r in render_rows if r.priority == "NEW"],
        generated_at="2026-03-08 12:00 UTC",
    )

    review_text = review_resp.text.replace("&amp;", "&")
    audit_text = audit_html.replace("&amp;", "&")
    work_text = work_html.replace("&amp;", "&")

    # Review queue no longer surfaces semantic read/rationale — only exports do.
    assert "Semantic Read" not in review_text
    assert "Rationale" not in review_text

    for char_no in (21, 22, 23, 24, 26):
        contract = contracts[char_no]
        assert contract is not None
        if contract["reason_fragments"]:
            for fragment in contract["reason_fragments"]:
                assert fragment in audit_text
                if str(char_no) in work_rows:
                    assert fragment in work_text
        assert contract["block_label"] in audit_text
        assert contract["detail"] in audit_text
        assert audit_rows[str(char_no)]["semantic_summary"] == contract["summary"]
        assert audit_rows[str(char_no)]["semantic_reason_summary"] == (contract["reason_summary"] or "")
        if str(char_no) in work_rows:
            assert contract["block_label"] in work_text
            assert contract["detail"] in work_text
            assert work_rows[str(char_no)]["semantic_summary"] == contract["summary"]
            assert work_rows[str(char_no)]["semantic_reason_summary"] == (contract["reason_summary"] or "")

    assert audit_rows["25"]["semantic_summary"] == ""
    assert "25" not in work_rows


def test_audit_packet_csv_rows(client: TestClient, engineer_user):
    """PACKET-02: CSV has correct rows for each ReviewItem."""
    from shop.dependencies import get_db
    db_gen = client.app.dependency_overrides.get(get_db)
    db = next(db_gen()) if db_gen else None
    if db is None:
        pytest.skip("no DB override available")
    run = _make_signed_run(db, engineer_user.id)
    from shop.services.exports import generate_audit_packet_csv
    buf = generate_audit_packet_csv(db, run)
    reader = csv.DictReader(buf)
    rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["char_no"] == "1"
    assert rows[0]["requirement_revA"] == "Req A 1"
    assert rows[0]["reviewer_decision"] == "approved"


def test_audit_packet_redownload(client: TestClient, db_engine, engineer_user):
    """PACKET-03: GET /exports/{id}/audit-packet.csv returns 200 with attachment header."""
    _login_engineer(client, db_engine, engineer_user)
    from shop.dependencies import get_db
    db_gen = client.app.dependency_overrides.get(get_db)
    db = next(db_gen()) if db_gen else None
    if db is None:
        pytest.skip("no DB override available")
    run = _make_signed_run(db, engineer_user.id)
    resp = client.get(f"/exports/{run.id}/audit-packet.csv")
    assert resp.status_code == 200
    assert "attachment" in resp.headers.get("content-disposition", "")


def _make_run_with_mixed_items(db, engineer_id, tmp_path=None):
    """Create signed-off run with changed, added, and unchanged items."""
    from shop.models import Run, ReviewItem
    out_dir = Path(tmp_path) / "mixed-items" if tmp_path else Path(tempfile.mkdtemp()) / "mixed-items"
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = _write_minimal_debug_snapshot(str(out_dir), version=1)
    run = Run(
        part_number="PN-WO",
        rev_a_label="A",
        rev_b_label="B",
        customer="Test",
        job_number="WO-001",
        status="signed_off",
        output_dir=str(out_dir),
        revA_path="/tmp/a.pdf",
        revB_path="/tmp/b.pdf",
        form3_path="/tmp/f.xlsx",
        reviewer_id=engineer_id,
        signed_at=datetime(2026, 3, 8),
        signed_by_id=engineer_id,
        packet_versions=[{
            "version": 1,
            "type": "original",
            "path": str(out_dir / "packets" / "v1.pdf"),
            "signed_at": "2026-03-08T12:00:00",
            "debug_snapshot_path": snapshot_path,
            "debug_total": 0,
            "unresolved_exception_count": 0,
        }],
    )
    db.add(run)
    db.flush()
    for char_no, classification in [(1, "changed"), (2, "added"), (3, "unchanged")]:
        item = ReviewItem(
            run_id=run.id,
            char_no=char_no,
            pipeline_classification=classification,
            confidence=0.9,
            requirement_revB=f"Req B {char_no}",
            reviewer_decision="approved",
        )
        db.add(item)
    db.commit()
    return run


def test_work_order_button_visible(client: TestClient, db_engine, engineer_user):
    """WORK-01: work-order links present on signed-off run status page."""
    _login_engineer(client, db_engine, engineer_user)
    from shop.dependencies import get_db
    db_gen = client.app.dependency_overrides.get(get_db)
    db = next(db_gen()) if db_gen else None
    if db is None:
        pytest.skip("no DB override available")
    run = _make_run_with_mixed_items(db, engineer_user.id)
    resp = client.get(f"/runs/{run.id}")
    assert resp.status_code == 200
    assert "work-order.pdf" in resp.text


def test_work_order_filters_status(client: TestClient, engineer_user):
    """WORK-02: work order CSV includes only changed and added characteristics."""
    from shop.dependencies import get_db
    db_gen = client.app.dependency_overrides.get(get_db)
    db = next(db_gen()) if db_gen else None
    if db is None:
        pytest.skip("no DB override available")
    run = _make_run_with_mixed_items(db, engineer_user.id)
    from shop.services.exports import generate_work_order_csv
    buf = generate_work_order_csv(db, run)
    reader = csv.DictReader(buf)
    rows = list(reader)
    assert len(rows) == 2  # only changed (char 1) and added (char 2)
    char_nos = {r["char_no"] for r in rows}
    assert "1" in char_nos
    assert "2" in char_nos
    assert "3" not in char_nos  # unchanged excluded
    # New columns present in header
    assert "requirement_revA" in rows[0]
    assert "confidence" in rows[0]
    assert "override_note" in rows[0]


def test_work_order_priority_labels(client: TestClient, engineer_user):
    """WORK-03: RE-MEASURE for changed, NEW for added; requirement_revA column present."""
    from shop.dependencies import get_db
    db_gen = client.app.dependency_overrides.get(get_db)
    db = next(db_gen()) if db_gen else None
    if db is None:
        pytest.skip("no DB override available")
    run = _make_run_with_mixed_items(db, engineer_user.id)
    from shop.services.exports import generate_work_order_csv
    buf = generate_work_order_csv(db, run)
    reader = csv.DictReader(buf)
    rows = {r["char_no"]: r for r in reader}
    assert rows["1"]["priority"] == "RE-MEASURE"
    assert rows["2"]["priority"] == "NEW"
    assert "requirement_revA" in rows["1"]


def test_work_order_pdf_csv(client: TestClient, db_engine, engineer_user):
    """WORK-04: Both work-order.pdf and .csv routes return 200 with attachment headers."""
    _login_engineer(client, db_engine, engineer_user)
    from shop.dependencies import get_db
    db_gen = client.app.dependency_overrides.get(get_db)
    db = next(db_gen()) if db_gen else None
    if db is None:
        pytest.skip("no DB override available")
    run = _make_run_with_mixed_items(db, engineer_user.id)
    for ext in ("csv",):  # csv only in tests (PDF needs WeasyPrint system deps)
        resp = client.get(f"/exports/{run.id}/work-order.{ext}")
        assert resp.status_code == 200, f"work-order.{ext} returned {resp.status_code}"
        assert "attachment" in resp.headers.get("content-disposition", "")


def _make_run_with_override(db, engineer_id, tmp_path=None):
    """Create signed-off run: char 1 changed+overridden with note, char 2 added (no note)."""
    from shop.models import Run, ReviewItem
    out_dir = Path(tmp_path) / "override-run" if tmp_path else Path(tempfile.mkdtemp()) / "override-run"
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = _write_minimal_debug_snapshot(str(out_dir), version=1)
    run = Run(
        part_number="PN-OVR",
        rev_a_label="A",
        rev_b_label="B",
        customer="Test",
        job_number="OVR-001",
        status="signed_off",
        output_dir=str(out_dir),
        revA_path="/tmp/a.pdf",
        revB_path="/tmp/b.pdf",
        form3_path="/tmp/f.xlsx",
        reviewer_id=engineer_id,
        signed_at=datetime(2026, 3, 8),
        signed_by_id=engineer_id,
        packet_versions=[{
            "version": 1,
            "type": "original",
            "path": str(out_dir / "packets" / "v1.pdf"),
            "signed_at": "2026-03-08T12:00:00",
            "debug_snapshot_path": snapshot_path,
            "debug_total": 0,
            "unresolved_exception_count": 0,
        }],
    )
    db.add(run)
    db.flush()
    # char 1: pipeline says changed, reviewer overrides and adds a note
    item1 = ReviewItem(
        run_id=run.id,
        char_no=1,
        pipeline_classification="changed",
        confidence=0.65,
        requirement_revA="Ø 6.0 ± 0.1",
        requirement_revB="Ø 6.0 ± 0.05",
        reviewer_decision="overridden",
        override_classification="changed",
        override_note="Tolerance tightened — confirmed with design authority",
        reviewed_by_id=engineer_id,
        reviewed_at=datetime(2026, 3, 8, 12, 0, 0),
    )
    # char 2: added, no override
    item2 = ReviewItem(
        run_id=run.id,
        char_no=2,
        pipeline_classification="added",
        confidence=0.92,
        requirement_revA=None,
        requirement_revB="R3.5 mm",
        reviewer_decision="approved",
        reviewed_by_id=engineer_id,
        reviewed_at=datetime(2026, 3, 8, 12, 0, 0),
    )
    db.add_all([item1, item2])
    db.commit()
    return run


def test_work_order_override_note_in_csv(client: TestClient, engineer_user):
    """WORK-05: override_note present for overridden items, empty for others."""
    from shop.dependencies import get_db
    db_gen = client.app.dependency_overrides.get(get_db)
    db = next(db_gen()) if db_gen else None
    if db is None:
        pytest.skip("no DB override available")
    run = _make_run_with_override(db, engineer_user.id)
    from shop.services.exports import generate_work_order_csv
    buf = generate_work_order_csv(db, run)
    reader = csv.DictReader(buf)
    rows = {r["char_no"]: r for r in reader}
    assert rows["1"]["override_note"] == "Tolerance tightened — confirmed with design authority"
    assert rows["2"]["override_note"] == ""


def test_work_order_confidence_in_csv(client: TestClient, engineer_user):
    """WORK-06: confidence formatted to 2 decimal places in CSV."""
    from shop.dependencies import get_db
    db_gen = client.app.dependency_overrides.get(get_db)
    db = next(db_gen()) if db_gen else None
    if db is None:
        pytest.skip("no DB override available")
    run = _make_run_with_override(db, engineer_user.id)
    from shop.services.exports import generate_work_order_csv
    buf = generate_work_order_csv(db, run)
    reader = csv.DictReader(buf)
    rows = {r["char_no"]: r for r in reader}
    # char 1 confidence=0.65 → "0.65"
    assert rows["1"]["confidence"] == "0.65"
    # char 2 confidence=0.92 → "0.92"
    assert rows["2"]["confidence"] == "0.92"
    # values are 2 dp formatted strings
    for char_no, row in rows.items():
        parts = row["confidence"].split(".")
        assert len(parts) == 2 and len(parts[1]) == 2, f"char {char_no}: bad confidence format {row['confidence']!r}"


def test_export_routes_require_signed_debug_snapshot_metadata(
    tmp_path, client: TestClient, db_engine, engineer_user
):
    """Phase 10-03 T1: Export routes must fail 409 when signed debug snapshot is missing.

    A signed-off run without a debug_snapshot_path in its packet_versions entry
    must be rejected by the export routes with 409 Conflict rather than silently
    falling back to mutable current state.

    A signed-off run WITH a valid debug_snapshot_path must be served normally (200).
    """
    from shop.models import Run, ReviewItem
    from shop.dependencies import get_db

    _login_engineer(client, db_engine, engineer_user)
    db_gen = client.app.dependency_overrides.get(get_db)
    db = next(db_gen()) if db_gen else None
    if db is None:
        pytest.skip("no DB override available")

    # --- Run A: signed_off but NO debug_snapshot_path ---
    run_no_snap = Run(
        part_number="PN-NOSNAP",
        rev_a_label="A",
        rev_b_label="B",
        customer="Test",
        job_number="NO-SNAP-001",
        status="signed_off",
        output_dir=None,
        revA_path="/tmp/a.pdf",
        revB_path="/tmp/b.pdf",
        form3_path="/tmp/f.xlsx",
        reviewer_id=engineer_user.id,
        signed_at=datetime(2026, 3, 8),
        signed_by_id=engineer_user.id,
        # packet_versions entry exists but has no debug_snapshot_path key
        packet_versions=[{
            "version": 1,
            "type": "original",
            "path": "/nonexistent/v1.pdf",
            "signed_at": "2026-03-08T12:00:00",
        }],
    )
    db.add(run_no_snap)
    db.flush()
    db.add(ReviewItem(
        run_id=run_no_snap.id,
        char_no=1,
        pipeline_classification="unchanged",
        confidence=0.9,
        requirement_revA="Req A",
        requirement_revB="Req A",
        reviewer_decision="approved",
    ))
    db.commit()

    # All signed export routes must reject with 409 when snapshot is absent
    for route_suffix in ("audit-packet.csv", "work-order.csv"):
        resp = client.get(f"/exports/{run_no_snap.id}/{route_suffix}")
        assert resp.status_code == 409, (
            f"Expected 409 for /{route_suffix} without snapshot, got {resp.status_code}: {resp.text}"
        )
        assert "snapshot" in resp.json().get("detail", "").lower(), (
            f"Expected 'snapshot' in 409 detail, got: {resp.json()}"
        )

    # --- Run B: signed_off WITH a valid debug_snapshot_path ---
    run_with_snap = _make_signed_run(db, engineer_user.id, tmp_path)
    for route_suffix in ("audit-packet.csv", "work-order.csv"):
        resp = client.get(f"/exports/{run_with_snap.id}/{route_suffix}")
        assert resp.status_code == 200, (
            f"Expected 200 for /{route_suffix} with valid snapshot, got {resp.status_code}: {resp.text}"
        )
