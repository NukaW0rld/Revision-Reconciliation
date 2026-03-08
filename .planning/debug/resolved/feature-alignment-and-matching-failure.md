---
status: resolved
trigger: "feature-alignment-and-matching-failure"
created: 2026-02-26T00:00:00Z
updated: 2026-02-26T04:00:00Z
---

## Current Focus

hypothesis: CONFIRMED AND FIXED. All residual issues resolved. Part1 char 31 now "unchanged". Part2 detects exactly 5 added features including .900 and .635/.615.
test: Both pipelines run to completion with correct results.
expecting: Human confirms the new results are correct.
next_action: Archive session after human verification.

## Symptoms

expected: Features that exist in both rev A and rev B (but shifted position) should be classified as "unchanged" or "changed" based on annotation text comparison, not position alone.
actual:
  - Features 2, 12, 15, 16: classified as "removed" with "No candidate found within search window"
  - Features 3, 4, 8, 10, 11: classified as "changed" with 0% text overlap and near-zero context scores (wrong candidate matched)
  - Features 5, 7, 9, 14: "uncertain" with near-zero confidence
  - Features 18-22: revA bbox used for revB snippet (identical bboxes)
  - Nearly all text/context scores are 0.0 across all matches
errors: No crash errors. Output produced but with incorrect classifications.
reproduction: uv run python run.py part2
started: Fails for part2 where rev B introduces a new projected view causing all features to shift significantly rightward. Works for part1.

## Eliminated

- hypothesis: ORB homography quality threshold too low (min 40 inliers, 15% ratio)
  evidence: Homography passes with 262 inliers, 0.873 ratio — quality is fine. The problem is the homography is WRONG, not low quality.
  timestamp: 2026-02-26T00:01:00Z

- hypothesis: Search radius (144 pts) too small
  evidence: True shift is ~362 pts. Root cause is wrong homography producing identity transform, not just radius.
  timestamp: 2026-02-26T00:01:00Z

## Evidence (continued from checkpoint)

- timestamp: 2026-02-26T03:00:00Z
  checked: Part1 balloon 31 anchor and candidate generation
  found: Balloon 31 (countersink depth=10) matched to span '10 x 90°' (score 0.391) but char 32 (angle 90°) matched the same span at higher score (0.451). Bipartite assignment gave '10 x 90°' to char 32 first, leaving char 31 unmatched → "removed". The span contains both values (10 and 90) representing a combined annotation: depth=10 AND angle=90° for the same countersink feature.
  implication: Combined annotation spans need shared-span fallback in bipartite assignment.

- timestamp: 2026-02-26T03:00:00Z
  checked: detect_added_characteristics filter for .900 and .635/.615
  found: .900 blocked at filter "not (symbol_tokens or count_tokens or angle_dimension)" because it has no symbol/count. .635 and .615 individually blocked by same filter. The stacked pair .635/.615 represents 0.625 ± 0.010 in limits form.
  implication: Need leading-decimal detection and stacked limits-pair detection in added-feature identification.

- timestamp: 2026-02-26T03:00:00Z
  checked: False positives from stacked-pair detection (initial implementation)
  found: Three .140 spans (orphaned lower limits of matched .160 spans for chars 3,4,8) detected as individual added features. Note list numbers '1.'...'5.' excluded by trailing_dot_re pattern. '4.'/'3.' pair (ratio=0.286) excluded by 20% threshold. .140 spans suppressed by proximity check (within 12 pts of matched .160 span).
  implication: Proximity-to-matched-span filter needed for leading-decimal Pass 2 detection; tighter 20% tolerance ratio threshold prevents numeric-pair false positives.

## Evidence


- timestamp: 2026-02-26T00:01:00Z
  checked: estimate_transform() on part2 assets at dpi=150
  found: Homography is identity matrix. Inliers=262, ratio=0.873.
  implication: ORB matched title block / border / zone letters (identical in both revisions at same positions). These static elements dominate feature matching and produce identity transform.

- timestamp: 2026-02-26T00:01:00Z
  checked: Text span positions in revA vs revB (common text spans)
  found: Title block text has 0 shift. Annotation features shift ~362 PDF pts rightward (.733: +362, 1.250±.010: +362, 2 x R.750: +362, Ø.300: +362, 2.900: +362). Side view annotations shift ~212 pts.
  implication: Two groups of features with different shifts. New projected view inserted in rev B caused existing views to shift right.

- timestamp: 2026-02-26T00:02:00Z
  checked: parse_requirement('.750', '0.750', '.160', '0.160')
  found: Leading-decimal numbers like '.750' parsed as 750.0, not 0.750. '0.750' parsed as 0.750. These compare as not equal, causing 0% text overlap for all dimension annotations.
  implication: normalize.py regex r'\d+\.?\d*' misses leading-decimal numbers.

- timestamp: 2026-02-26T00:02:00Z
  checked: context score computation in match.py
  found: block_id filter in context scoring excludes all neighbor spans (each annotation is its own block in PDF structure). Context score is always 0.0.
  implication: Context score component was non-functional.

- timestamp: 2026-02-26T00:02:00Z
  checked: Added characteristics snippet logic in cli.py
  found: revA_bbox_pdf = revB_bbox_pdf for added items (line 370). Both snippets show same revB location.
  implication: Added item revA snippets showed wrong location (revB content instead of empty/different revA area).

## Resolution

root_cause: Seven separate issues (5 from initial investigation + 2 residual):

  1. ORB alignment fails: Both PDFs share identical title block features that dominate ORB matching, producing identity homography when annotation content actually shifted ~362 pts rightward.

  2. Leading-decimal number parsing: normalize.py regex r'\d+\.?\d*' misses numbers like '.750', '.160', '.010' (no leading digit), parsing them as 750.0, 160.0, 10.0 instead of 0.750, 0.160, 0.010. This caused 0% text overlap for all annotations since Form 3 uses full decimal (0.750) while PDF uses leading-decimal (.750).

  3. Context score always 0: match.py context scoring filtered candidate neighbor spans by block_id, but each annotation in an engineering PDF is its own block. This made context score structurally zero.

  4. Wrong classification for limits-form annotations: PDF annotations use limits form (.160/.140) for toleranced dimensions (0.150 ± 0.010). The system had no understanding of this notation, causing limits-form matches to be classified as "changed" or to beat correct candidates with primary-mismatch penalties.

  5. Added item snippets: cli.py used revB bbox for both revA and revB snippets of added characteristics. Fixed to use inverse homography to map revB location back to revA space.

  6. Shared-annotation misclassification (part1 char 31): PDF encodes countersink depth (10) and angle (90°) in a single span '10 x 90°'. Bipartite assignment gave the span to char 32 (angle=90°, score 0.451), leaving char 31 (depth=10, score 0.391) unmatched → "removed". Root cause: strict bipartite constraint doesn't handle combined annotation spans.

  7. Added feature detection misses leading-decimal and stacked limits (part2 features .900, .635/.615): detect_added_characteristics filter required symbol_tokens or count_tokens. Leading-decimal values like ".900" and stacked pairs like ".635"/".615" have neither → filtered out. Root cause: filter too strict for plain dimension annotations without prefixes.

fix:
  - alignment.py: Added estimate_transform_from_text_spans() that matches dimension-like text spans between revisions to compute a pure translation transform. Added _is_dimension_like(), _homography_is_near_identity() helpers.
  - cli.py: Added text-span alignment as a validation/override for ORB when ORB is near-identity but text spans indicate significant shift. Fixed added-item revA bbox to use inverse homography.
  - normalize.py: Changed regex to r'\d+\.?\d*|\.\d+' to also match leading-decimal numbers.
  - match.py: Removed block_id filter in context scoring (use spatial distance only). Added non-numeric span penalty for dimension anchors. Added reduced penalty for close primary values (limits-form). Added notes-anchor candidate filter (only header spans). Added MIN_MATCH_SCORE=0.02 threshold. Increased SEARCH_RADIUS from 144 to 288 pts. Added notes-type identity transform (notes don't move between revisions).
  - classify.py: Added limits-form detection in classification (primary within tolerance band → unchanged). Added primary-mismatch override for tolerance-overlap false positives. Added relative-tolerance threshold (< 15%) for limits-form.
  - match.py (assign_matches): Added shared-span fallback pass after main greedy assignment. Unmatched anchors whose primary value appears in an already-used span are allowed to share that span. Handles combined annotation spans that encode multiple characteristics.
  - classify.py (detect_added_characteristics): Added Pass 1 for stacked tolerance-limit pair detection (two numeric-only spans same x-column, adjacent y, difference ≤ 20% of mean). Added leading-decimal single value detection (regex "^\.\d+$"). Added proximity-to-matched-span filter to suppress orphaned tolerance limit spans. Added trailing-dot filter to exclude note list numbers.

verification: Part1: 37 unchanged, 1 changed, 0 removed, 1 added (char 31 now correctly "unchanged"). Part2: 11 unchanged, 3 changed, 3 removed, 5 added (includes .900 and .635/.615). All existing classifications preserved.

files_changed:
  - delta_preservation/vision/alignment.py
  - delta_preservation/cli.py
  - delta_preservation/reconcile/normalize.py
  - delta_preservation/reconcile/match.py
  - delta_preservation/reconcile/classify.py

files_changed:
  - delta_preservation/vision/alignment.py
  - delta_preservation/cli.py
  - delta_preservation/reconcile/normalize.py
  - delta_preservation/reconcile/match.py
  - delta_preservation/reconcile/classify.py
