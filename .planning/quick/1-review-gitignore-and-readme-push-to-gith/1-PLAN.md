---
phase: quick
plan: 1
type: execute
wave: 1
depends_on: []
files_modified:
  - .gitignore
  - README.md
autonomous: false
requirements: []
must_haves:
  truths:
    - ".gitignore prevents sensitive or noisy runtime artifacts from reaching the public repo"
    - "README.md accurately reflects the current project including the web interface built in Phase 2"
    - "All commits are pushed to the GitHub remote"
    - "A v0.2 release message is drafted and output in conversation"
  artifacts:
    - path: ".gitignore"
      provides: "Exclusion rules appropriate for a public GitHub repo"
    - path: "README.md"
      provides: "Accurate project description including web app"
  key_links:
    - from: "local repo"
      to: "https://github.com/NukaW0rld/Revision-Reconciliation.git"
      via: "git push origin main"
---

<objective>
Review .gitignore and README.md for public GitHub repo readiness, commit any changes, push to GitHub, and draft a v0.2 release message in conversation.

Purpose: Prepare the repository for public visibility — ensure no sensitive files are exposed, the README reflects the current state of the project (including the Phase 2 web interface), and the repo is up to date on GitHub.
Output: Updated .gitignore (if needed), updated README.md (if needed), pushed commits, and v0.2 release message printed in conversation.
</objective>

<execution_context>
@/home/khoa2/.claude/get-shit-done/workflows/execute-plan.md
</execution_context>

<context>
@/home/khoa2/delta-preservation/.planning/STATE.md
@/home/khoa2/delta-preservation/README.md
@/home/khoa2/delta-preservation/.gitignore
@/home/khoa2/delta-preservation/pyproject.toml
</context>

<tasks>

<task type="auto">
  <name>Task 1: Review and fix .gitignore for public repo</name>
  <files>/home/khoa2/delta-preservation/.gitignore</files>
  <action>
Explore the full codebase to determine what should be excluded from the public GitHub repo. Run these checks:

```bash
# See all tracked files
git -C /home/khoa2/delta-preservation ls-files

# See what's currently untracked (already excluded)
git -C /home/khoa2/delta-preservation ls-files --others --exclude-standard

# See what runtime artifacts exist on disk but might slip through
ls /home/khoa2/delta-preservation/data/
ls /home/khoa2/delta-preservation/uploads/ 2>/dev/null
ls /home/khoa2/delta-preservation/out/ 2>/dev/null
```

Key things to verify and fix:

1. **`uv.lock`** — Currently in .gitignore. For a public repo with `uv`, uv.lock SHOULD be committed for reproducible installs. Remove `uv.lock` from .gitignore and stage it for commit.

2. **`.planning/` files** — They are listed in .gitignore but are already tracked by git (gitignore does not untrack already-committed files). Determine: does the user want to keep planning artifacts in the public repo? Since there is no env-var secret or PII in those files (they are project planning docs), leave them tracked — no change needed. Do NOT run `git rm --cached` on .planning/ without user confirmation.

3. **`data/` directory** — Contains runtime artifacts (huey.db, shop.db, uploads/, out/). Verify `data/` is in .gitignore (it is). Verify none of its contents are tracked. If `git ls-files data/` returns anything, add it to .gitignore and untrack with `git rm --cached`.

4. **`shop.db`, `huey.db`** at root — Verify these are in .gitignore (they are) and not tracked.

5. **Secrets scan** — Check for any .env files, API keys, credentials:
```bash
git -C /home/khoa2/delta-preservation ls-files | grep -iE '\.env|secret|credential|key|password|token'
```
If any found and they contain real secrets, add to .gitignore and untrack.

6. **`package.json` / `static/dist/`** — These are tracked. The built CSS (`static/dist/output.css`) is tracked. For a public repo this is fine — it allows zero-build deployment. No change needed.

After assessment, if changes are needed:
- Edit .gitignore to add/remove rules
- Run `git rm --cached <path>` for any newly-ignored tracked files
- Stage the .gitignore changes

If no changes are needed, note that in the output and move on.
  </action>
  <verify>
    <automated>git -C /home/khoa2/delta-preservation ls-files | grep -E 'shop\.db|huey\.db|uploads/|out/' || echo "No sensitive runtime artifacts tracked — PASS"</automated>
  </verify>
  <done>No sensitive runtime artifacts or secrets tracked. .gitignore is appropriate for a public GitHub repo. Any necessary changes staged.</done>
</task>

<task type="auto">
  <name>Task 2: Update README.md to reflect current project state</name>
  <files>/home/khoa2/delta-preservation/README.md</files>
  <action>
The README.md currently has an inaccurate "Planned Improvements" section. Phase 2 of the project is now complete — the web interface for quality team review and approval workflows has been built (the `shop/` package). Update the README to reflect this.

Specific changes required:

1. **"Planned Improvements" / "Future Work" section** — Remove or update the bullet "Web interface for quality team review and approval workflows" since it is now built. Replace with accurate description of what is built vs what is still planned.

2. **Repository Structure section** — The current structure only shows the CLI pipeline. Add `shop/` and `docker/` to the structure:
```
delta-preservation/
├── run.py                       # Convenience entry point (uv run python run.py part1)
├── run_web.py                   # Web application entry point
├── delta_preservation/          # Core pipeline package
│   ├── io/                      # PDF and Excel I/O modules
│   ├── vision/                  # Computer vision (balloons, alignment)
│   ├── reconcile/               # Core matching and classification logic
│   ├── cli.py                   # Pipeline orchestration and CLI
│   ├── types.py                 # Pydantic data models
│   └── config.py                # Configuration constants
├── shop/                        # Web application (FastAPI + Jinja2)
│   ├── routers/                 # Auth, runs, admin, setup routes
│   ├── services/                # Business logic layer
│   ├── templates/               # Jinja2 HTML templates
│   └── tasks.py                 # Huey async pipeline task
├── docker/                      # Docker deployment files
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── supervisord.conf
├── tests/                       # Test suite
├── assets/                      # Test fixtures and sample data
└── static/                      # Built CSS, bundled JS (htmx, htmx-sse)
```

3. **Current Limitations** — Update "Single-page support" — verify if this is still accurate by checking cli.py. If it is still a limitation, keep it. Remove "Web interface" from planned items since it's done.

4. **Add a brief "Web Interface" section** after the pipeline description explaining:
- FastAPI + Jinja2 web app in `shop/`
- Upload Rev A PDF, Rev B PDF, Form 3 Excel via browser form
- Real-time stage progress via SSE
- Async pipeline execution via Huey task queue (SQLite-backed, no external services)
- Docker deployment with supervisord managing uvicorn + huey_consumer
- Role-based access: admin + engineer roles

5. **Docker / Deployment section** — Add a brief section showing how to run via Docker:
```bash
cd docker/
docker compose up --build
```

6. **Version** — The README currently has no version. Do NOT add one — version tracking is handled by git tags and the release message.

Keep all existing content that is still accurate. Do not rewrite sections that don't need updating. Make minimal targeted edits.

After editing, stage the file: `git -C /home/khoa2/delta-preservation add README.md`
  </action>
  <verify>
    <automated>grep -n "shop/" /home/khoa2/delta-preservation/README.md | head -5</automated>
  </verify>
  <done>README.md accurately describes both the CLI pipeline and the web application. Repository structure includes shop/ and docker/. Planned improvements no longer lists web interface as future work. File staged.</done>
</task>

<task type="auto">
  <name>Task 3: Commit staged changes, push to GitHub, draft release message</name>
  <files></files>
  <action>
1. Check what is staged:
```bash
git -C /home/khoa2/delta-preservation status
git -C /home/khoa2/delta-preservation diff --staged
```

2. Also stage the two currently-modified tracked files (STATE.md and config.json) — they reflect project completion state and should be in the repo:
```bash
git -C /home/khoa2/delta-preservation add .planning/STATE.md .planning/config.json
```

3. If uv.lock was removed from .gitignore in Task 1, also add it:
```bash
git -C /home/khoa2/delta-preservation add uv.lock 2>/dev/null || true
```

4. Commit all staged changes. If only STATE.md / config.json changed (no .gitignore or README changes were needed), commit with an appropriate message. If README was updated, include that. Use this format:
```bash
git -C /home/khoa2/delta-preservation commit -m "$(cat <<'EOF'
docs: update README for Phase 2 web app and fix gitignore for public repo

- Add shop/, docker/, static/ to repository structure section
- Add web interface section describing FastAPI app and Docker deployment
- Update planned improvements to reflect web interface is now built
- Stage planning state reflecting Phase 2 completion

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

Adjust commit message to accurately reflect only what actually changed (if README was not changed, remove that line, etc.).

5. Push to origin:
```bash
git -C /home/khoa2/delta-preservation push origin main
```

6. Get the final commit hash for the release message:
```bash
git -C /home/khoa2/delta-preservation log --oneline -3
```

7. Output the v0.2 release message directly in the conversation (do NOT write it to a file). The release message should be:

---

**v0.2 — Web Application + Docker Deployment**

This release ships the complete web interface for the delta preservation pipeline, transforming the CLI prototype into a deployable quality control application.

**What's new in v0.2:**

- **Web application** (`shop/`) — FastAPI + Jinja2 UI for uploading Rev A/B PDFs and Form 3 Excel, triggering the pipeline, and monitoring results
- **Real-time pipeline progress** — Server-sent events (SSE) stream stage-by-stage progress to the browser during analysis
- **Async task execution** — Huey task queue with SQLite storage runs the pipeline in the background; no Redis or external services required
- **Alert system** — Run completion alerts with inline dismissal via HTMX
- **Role-based access** — Admin and engineer roles with bcrypt authentication and session tokens
- **Setup wizard** — First-run wizard for shop name, admin credentials, engineer account, and Form 3 column mapping
- **Docker deployment** — Single-container deployment with supervisord managing uvicorn + huey_consumer; Tailwind CSS built at image build time
- **Audit trail foundation** — Run records linked to reviewers; sign-off and review queue planned for v0.3

**Upgrade from v0.1:**
v0.1 was CLI-only. v0.2 adds the web layer on top of the unchanged pipeline. CLI usage (`uv run python run.py part1`) continues to work.

**Docker quick start:**
```bash
cd docker/
docker compose up --build
```

---

Print this release message verbatim in the conversation response. Do NOT write it to any file.
  </action>
  <verify>
    <automated>git -C /home/khoa2/delta-preservation log --oneline -1 && git -C /home/khoa2/delta-preservation status</automated>
  </verify>
  <done>All changes committed and pushed to https://github.com/NukaW0rld/Revision-Reconciliation.git. v0.2 release message printed in conversation.</done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <what-built>Reviewed .gitignore, updated README.md to reflect the Phase 2 web app, committed all changes including STATE.md and config.json, and pushed to GitHub. v0.2 release message printed in conversation.</what-built>
  <how-to-verify>
    1. Visit https://github.com/NukaW0rld/Revision-Reconciliation to confirm the push succeeded
    2. Confirm README.md on GitHub shows shop/ in the structure and a web interface section
    3. Confirm no sensitive files (shop.db, uploads/, out/) appear in the repo
    4. Confirm the v0.2 release message was printed in conversation (not written to a file)
  </how-to-verify>
  <resume-signal>Type "approved" if everything looks good, or describe any issues</resume-signal>
</task>

</tasks>

<verification>
- No runtime artifacts (shop.db, huey.db, uploads/, out/, data/) tracked in git
- No secrets (.env, API keys, passwords) tracked in git
- README.md includes shop/ in structure and describes the web interface
- All commits pushed to https://github.com/NukaW0rld/Revision-Reconciliation.git
- v0.2 release message output in conversation only (no file created)
</verification>

<success_criteria>
- Public GitHub repo at https://github.com/NukaW0rld/Revision-Reconciliation.git is up to date
- README.md accurately reflects v0.2: CLI pipeline + web application + Docker deployment
- .gitignore is appropriate for a public repo (no sensitive files exposed)
- v0.2 release message is visible in the conversation
</success_criteria>

<output>
No SUMMARY.md required for quick tasks.
</output>
