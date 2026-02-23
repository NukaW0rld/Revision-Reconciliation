"""
Convenience entry point: uv run python run.py <part_name> [--dpi N] [--out_dir PATH]
Resolves assets/{part_name}/ relative to this file's directory (always the repo root).
"""
import argparse
from pathlib import Path
from delta_preservation.cli import run_pipeline

REPO_ROOT = Path(__file__).parent.resolve()


def main():
    parser = argparse.ArgumentParser(
        description="Run delta preservation pipeline for a named part"
    )
    parser.add_argument(
        "part_name",
        help="Part name matching assets/ subdirectory (e.g. part1, part2)"
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--out_dir", default=str(REPO_ROOT / "out"))
    args = parser.parse_args()

    asset_dir = REPO_ROOT / "assets" / args.part_name
    if not asset_dir.is_dir():
        parser.error(f"No asset directory for part '{args.part_name}': {asset_dir}")

    run_pipeline(
        revA_pdf=asset_dir / "revA.pdf",
        revB_pdf=asset_dir / "revB.pdf",
        form3_xlsx=asset_dir / "FAIR.xlsx",
        out_dir=args.out_dir,
        dpi=args.dpi,
        part_name=args.part_name,
    )


if __name__ == "__main__":
    main()
