# delta-preservation

Prototype for preserving inspection characteristic identity across drawing revisions.

This repository contains an early, Python-first prototype focused on a single hard problem:

> Given Rev A (ballooned), Rev B (unballooned), and an existing AS9102 Form 3, create a run artifact that will later reconcile which inspection characteristics are unchanged, changed, removed, or uncertain across revisions.

At the current stage, the system:
- validates inputs,
- creates a uniquely identified run directory,
- and writes a minimal `delta_packet.json` stub that later pipeline stages will populate.

---

## Requirements

- Python 3.10+
- No external system dependencies at this stage

---

## Repository structure (current)

```
delta-preservation/
delta-preservation/
cli.py
out/
<run_id>/
delta_packet.json
snippets/
intermediate/
README.md

```

---

## Running the CLI (Step 1)

From the **repository root**, run:

```bash
python -m delta_preservation.cli \
  --revA_pdf path/to/revA.pdf \
  --revB_pdf path/to/revB.pdf \
  --form3_xlsx path/to/form3.xlsx \
  --part_name Part1
````

### Required arguments

* `--revA_pdf`
  Path to the Revision A PDF (must exist, must be `.pdf`)

* `--revB_pdf`
  Path to the Revision B PDF (must exist, must be `.pdf`)

* `--form3_xlsx`
  Path to the AS9102 Form 3 Excel file (must exist, must be `.xlsx`)

### Optional arguments

* `--out_dir`
  Output directory root (default: `./out`)

* `--dpi`
  Rendering DPI to be used by later pipeline stages (default: `300`)

* `--part_name`
  Human-readable part identifier used in the run ID (default: `part`)

---

## Output

Each invocation creates a **new run directory**:

```
out/<part_name>_<timestamp>_<hash>/
  delta_packet.json
  snippets/
  intermediate/
```

### `delta_packet.json` (stub)

At this stage, the file contains:

```json
{
  "run_id": "...",
  "inputs": {
    "revA_pdf": "...",
    "revB_pdf": "...",
    "form3_xlsx": "...",
    "dpi": 300
  },
  "items": []
}
```

* `items` is intentionally empty and will be populated by later reconciliation steps.
* `snippets/` and `intermediate/` are created up front to avoid ad-hoc filesystem writes later.

---

## Design notes

* Each run is immutable: if a run directory already exists, execution fails.
* Run IDs include:

  * part name,
  * timestamp,
  * short hash of input paths
    to ensure traceability during experimentation.
* This stage performs **no drawing analysis** yet. It establishes the execution container and input record only.

---

## Next steps (not yet implemented)

* Form 3 parsing
* Balloon detection on Rev A
* Rev A → Rev B alignment
* Characteristic matching and confidence scoring
* Evidence snippet generation
* Population of `items` in `delta_packet.json`

These will be layered incrementally on top of the current run structure.