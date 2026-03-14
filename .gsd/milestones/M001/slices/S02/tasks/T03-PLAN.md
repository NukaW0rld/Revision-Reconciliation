# T03: 02-pipeline-bridge 03

**Slice:** S02 — **Milestone:** M001

## Description

Connect the Huey worker to the existing pipeline by adding stage progress callbacks and implementing full failure/warning classification in the task.

Purpose: The pipeline already works standalone. This plan adds the minimal wiring needed for the web app to observe progress and outcomes: a `stage_callback` parameter in `run_pipeline()` and a complete `run_pipeline_task()` implementation in `shop/tasks.py`.
Output: cli.py with stage_callback support, shop/tasks.py with full pipeline task (stage updates, failure handling, warning detection, alert creation).

## Must-Haves

- [ ] "run_pipeline() accepts optional stage_callback and calls it before each of the 8 stages"
- [ ] "run_pipeline_task Huey task calls run_pipeline() with stage_callback that updates Run DB row"
- [ ] "RevA balloon failure sets Run.status=failed, Run.failure_stage, Run.failure_message"
- [ ] "RevB balloon failure sets Run.status=warning, Run.warning_type=revB_balloon"
- [ ] "Alignment inlier_ratio below MIN_ALIGNMENT_RATIO sets Run.status=warning, Run.warning_type=low_confidence with JSON confidence_summary"
- [ ] "Successful completion sets Run.status=completed and Run.output_dir to the pipeline output path"
- [ ] "On failure, RunAlert is created for the reviewer"

## Files

- `delta_preservation/cli.py`
- `shop/tasks.py`
