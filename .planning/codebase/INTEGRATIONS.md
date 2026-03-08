# External Integrations

**Analysis Date:** 2026-02-25

## APIs & External Services

**None detected.**

This is a standalone command-line application with no external API dependencies.

## Data Storage

**Databases:**
- None - All data is in-memory or file-based

**File Storage:**
- Local filesystem only
  - Input: PDF files (Rev A and Rev B) + Excel Form 3 workbook
  - Output: JSON delta packet + PNG evidence snippet images
  - Location: `out/<part_name>_<timestamp>_<hash>/`
  - No cloud storage (S3, GCS, Azure Blob) detected

**Caching:**
- None - Pipeline executes fresh for each run; no Redis or in-process cache framework

## Authentication & Identity

**Auth Provider:**
- None - No authentication required
- All processing is local

**Implementation:**
- N/A

## Monitoring & Observability

**Error Tracking:**
- None - No Sentry, DataDog, or error tracking service

**Logs:**
- Console output only
  - Progress messages printed to stdout (pipeline stages [1/8]–[8/8])
  - JSON debug artifacts in `out/<part_name>_<timestamp>_<hash>/debug/` directory:
    - `form3_chars.json` - Parsed Form 3 characteristics
    - `tolerance_parsing_tests.json` - Tolerance validation debug data

## CI/CD & Deployment

**Hosting:**
- Not applicable - CLI tool, runs locally or in custom orchestration

**CI Pipeline:**
- Not detected
- Test suite exists (`pytest`) but no GitHub Actions, GitLab CI, or other CI configuration in repo

## Environment Configuration

**Required env vars:**
- None - All configuration via command-line arguments

**Secrets location:**
- No secrets management
- All configuration is explicit (file paths, DPI, output directory)

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

---

*Integration audit: 2026-02-25*
