import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Delta preservation pipeline for engineering drawings"
    )
    parser.add_argument("--revA_pdf", required=True, help="Path to Revision A PDF")
    parser.add_argument("--revB_pdf", required=True, help="Path to Revision B PDF")
    parser.add_argument("--form3_xlsx", required=True, help="Path to Form 3 XLSX")
    parser.add_argument("--out_dir", default="./out", help="Output directory (default: ./out)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for rendering (default: 300)")
    parser.add_argument("--part_name", default="part", help="Part name (default: part)")
    
    args = parser.parse_args()
    
    # Validate input files
    revA_path = Path(args.revA_pdf)
    revB_path = Path(args.revB_pdf)
    form3_path = Path(args.form3_xlsx)
    
    if not revA_path.exists():
        raise FileNotFoundError(f"Revision A PDF not found: {revA_path}")
    if not revA_path.suffix.lower() == ".pdf":
        raise ValueError(f"Revision A must be a PDF file, got: {revA_path.suffix}")
    
    if not revB_path.exists():
        raise FileNotFoundError(f"Revision B PDF not found: {revB_path}")
    if not revB_path.suffix.lower() == ".pdf":
        raise ValueError(f"Revision B must be a PDF file, got: {revB_path.suffix}")
    
    if not form3_path.exists():
        raise FileNotFoundError(f"Form 3 XLSX not found: {form3_path}")
    if not form3_path.suffix.lower() == ".xlsx":
        raise ValueError(f"Form 3 must be an XLSX file, got: {form3_path.suffix}")
    
    # Generate deterministic run_id
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    hash_input = f"{revA_path.absolute()}{revB_path.absolute()}{form3_path.absolute()}"
    short_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
    run_id = f"{args.part_name}_{timestamp}_{short_hash}"
    
    # Create output directory structure
    run_dir = Path(args.out_dir) / run_id
    snippets_dir = run_dir / "snippets"
    intermediate_dir = run_dir / "intermediate"
    
    run_dir.mkdir(parents=True, exist_ok=False)
    snippets_dir.mkdir(parents=True, exist_ok=False)
    intermediate_dir.mkdir(parents=True, exist_ok=False)
    
    # Write stub delta_packet.json
    delta_packet = {
        "run_id": run_id,
        "inputs": {
            "revA_pdf": str(revA_path.absolute()),
            "revB_pdf": str(revB_path.absolute()),
            "form3_xlsx": str(form3_path.absolute()),
            "dpi": args.dpi
        },
        "items": []
    }
    
    packet_path = run_dir / "delta_packet.json"
    with open(packet_path, "w") as f:
        json.dump(delta_packet, f, indent=2)
    
    # Print summary
    print(f"Run created: {run_id}")
    print(f"Output folder: {run_dir.absolute()}")


if __name__ == "__main__":
    main()
