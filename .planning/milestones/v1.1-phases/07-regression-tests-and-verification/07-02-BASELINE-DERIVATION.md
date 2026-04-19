# Phase 7 TST-02 Baseline Derivation

**Derived at:** 2026-04-17
**Git SHA:** c5c19f0
**Source:** `assets/debug_report_part{1..9}.json`

## Derivation command

```python
python3 - <<'PY'
import json, collections
from pathlib import Path
for i in range(1, 10):
    path = Path(f"assets/debug_report_part{i}.json")
    d = json.load(path.open())
    items = d.get("items", [])
    statuses = collections.Counter(
        (it.get("evaluation") or {}).get("status", "(no_eval)") for it in items
    )
    conforming = statuses.get("conforming", 0)
    review = statuses.get("review_needed", 0)
    virtual_missing_rows = statuses.get("(no_eval)", 0)
    missing_idx = d.get("missing_added_truth_indexes", [])
    print(
        f"part{i}: items={len(items)} conforming={conforming} "
        f"review_needed={review} virtual_missing_rows={virtual_missing_rows} "
        f"missing_added_truth_indexes={missing_idx}"
    )
PY
```

## Derived counts

```
part1: items=39 conforming=23 review_needed=16 virtual_missing_rows=0 missing_added_truth_indexes=[]
part2: items=23 conforming=18 review_needed=5 virtual_missing_rows=0 missing_added_truth_indexes=[]
part3: items=22 conforming=12 review_needed=10 virtual_missing_rows=0 missing_added_truth_indexes=[]
part4: items=17 conforming=7 review_needed=10 virtual_missing_rows=0 missing_added_truth_indexes=[]
part5: items=17 conforming=9 review_needed=8 virtual_missing_rows=0 missing_added_truth_indexes=[]
part6: items=20 conforming=13 review_needed=7 virtual_missing_rows=0 missing_added_truth_indexes=[]
part7: items=17 conforming=7 review_needed=10 virtual_missing_rows=0 missing_added_truth_indexes=[]
part8: items=13 conforming=7 review_needed=5 virtual_missing_rows=1 missing_added_truth_indexes=[10]
part9: items=42 conforming=7 review_needed=27 virtual_missing_rows=8 missing_added_truth_indexes=[35, 36, 37, 38, 39, 40, 41, 42]
```

## Baseline dict for test_phase7_benchmark.py

```python
BASELINE_COUNTS = {
    "part1": {"min_conforming": 23, "max_missing_added": 0},
    "part2": {"min_conforming": 18, "max_missing_added": 0},
    "part3": {"min_conforming": 12, "max_missing_added": 0},
    "part4": {"min_conforming": 7,  "max_missing_added": 0},
    "part5": {"min_conforming": 9,  "max_missing_added": 0},
    "part6": {"min_conforming": 13, "max_missing_added": 0},
    "part7": {"min_conforming": 7,  "max_missing_added": 0},
    "part8": {"min_conforming": 7,  "max_missing_added": 1},
    "part9": {"min_conforming": 7,  "max_missing_added": 8},
}
```
