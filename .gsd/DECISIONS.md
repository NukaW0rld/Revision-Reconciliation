# Decisions Register

<!-- Append-only. Never edit or remove existing rows.
     To reverse a decision, add a new row that supersedes it.
     Read this file at the start of any planning or research phase. -->

| # | When | Scope | Decision | Choice | Rationale | Revisable? |
|---|------|-------|----------|--------|-----------|------------|
| D001 | M003 | arch | UI aesthetic direction | Industrial dark-mode — dark bg, amber/green status, monospaced accents | User selected "industrial precision" — aerospace tooling software feel, not generic SaaS | Yes — if customer feedback demands light mode |
| D002 | M003 | arch | Navigation layout | Sidebar + top bar | User selected sidebar for section nav + top bar for context; current single top nav too flat for multi-section tool | Yes — if responsive/mobile layout needed |
| D003 | M003 | arch | Review queue interaction model | Full-screen focused mode — one item at a time, keyboard shortcuts A/O/J/K | User selected full-screen; keyboard nav for processing 50-100 items efficiently | Yes — if team prefers dense list |
| D004 | M003 | convention | HTMX swap IDs | Preserved unchanged from M001/M002 | Backend routers and HTMX attributes reference specific IDs; changing them requires backend changes which are out of scope | No — would require coordinated backend change |
| D005 | M003 | scope | Print/export templates | Not redesigned in M003 | audit_packet.html and work_order.html are WeasyPrint PDF templates — print aesthetics are separate from screen UI | Yes — future print redesign milestone |
| D006 | M003 | scope | Interaction patterns | Open to change where better UX exists | User confirmed interactions can change as long as backend can support; not locked to current HTMX patterns | Yes |
| D007 | M003/S01 | arch | Standalone page layout | `_base_standalone.html` — separate template for login/wizard, not extending `base.html` | Login and wizard must not render sidebar chrome; overriding `{% block nav %}` won't hide sidebar since it's outside that block; separate standalone base is the clean solution | No — login/wizard must stay decoupled from sidebar layout |
| D008 | M003/S01 | convention | DaisyUI v5 custom theme definition | Raw `[data-theme="name"]{...}` CSS block in `input.css`, placed after `@plugin "daisyui"` directive, not inside it | DaisyUI v5 @plugin handles semantic class generation; the raw CSS block handles custom color values via CSS custom properties. Attempting to nest theme vars inside @plugin config caused build errors. | No — this is the DaisyUI v5 API contract |
| D009 | M003/S01 | convention | Wizard step progress indicator | Custom flex divs instead of DaisyUI `ul.steps` component | DaisyUI steps uses CSS pseudo-element counter backgrounds that don't render cleanly against `oklch(13% 0.005 260)` (near-black base-100). Raw flex divs with explicit border/background classes give identical visual semantics with full control over theme vars. | Yes — if DaisyUI steps gains dark-mode support or a workaround is found |
