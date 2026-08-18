# Python Lab Phase 1 Deployment Test Report

## Verdict

**PASS — acceptable for private staging**, after the mandatory operator credential-rotation check below. Public multi-user production is not approved because Jupyter kernels still execute on the API host without container or microVM isolation and enforced resource controls.

## Automated results

- Backend: `163 passed`, 9 dependency deprecation warnings.
- Focused backend Phase 1 matrix: `89 passed` across the release and configuration/export suites.
- Frontend: `18 passed`.
- TypeScript: passed.
- Production frontend build: passed with Next.js 16.3.1.
- Backend startup import: passed.
- Phase 1 frontend lint scope: passed with no warnings or errors.
- Full frontend lint: existing unrelated backlog of 55 errors and 14 warnings; no changed Phase 1 file is implicated.
- Backend formatting/import order for changed files: passed.
- Full backend Black baseline: existing repository-wide formatting backlog; not introduced by this release.
- Secret scan: both repositories' tracked files passed after removing a credential-shaped MongoDB URI from tracked `.env.example`.
- Frontend dependency audit: `0 vulnerabilities` after compatible, non-forced upgrades and overrides.
- Backend environment dependency check: global environment reports three LangGraph packages missing `langchain-core`; these packages are not declared by this repository. `pip-audit` is not installed/configured.
- `.ipynb`: the real TypeScript builder output passed `nbformat.validate`.

## Security matrix

| Area | Result | Evidence |
|---|---|---|
| Authentication | Pass | All eight execution/lifecycle endpoints reject absent tokens; malformed, invalid, expired, empty, and unsupported schemes are covered. |
| Ownership | Pass | User B is denied across 16 notebook, cell, execution, clear, and lifecycle operations with the same safe `404` policy as a missing notebook. |
| Interrupt independence | Pass | Synchronization-based fake verifies interrupt does not wait on execution locking and the API remains responsive. |
| Event-loop safety | Pass | Controlled blocking Jupyter work is delegated and an unrelated request completes first. |
| Error disclosure | Pass | Forced repository, service, kernel, and persistence failures return controlled responses without secrets, paths, tracebacks, or environment data. |
| HTML output | Pass | Behavioral DOMPurify test removes scripts, handlers, JavaScript URLs, iframes, and inline CSS; iframe remains scriptless with restrictive CSP. |
| Route protection | Pass | Loading, redirect, authenticated, API `401`, and non-disclosing `404` states are tested. |
| Secrets/config | Pass with operator action | Tracked scan passes; production rejects debug mode, weak/missing auth, missing/local DB, insecure object storage, insecure origins, and debug logging. |

## Functional matrix

| Feature | Result | Evidence |
|---|---|---|
| Notebook CRUD/update | Pass | Owner paths plus nullable description, partial/multi-field updates, strict unknown-field handling, invalid types/limits, and atomic rejection. |
| Cell CRUD/reorder | Pass | Code/markdown create/update/delete and complete three-cell reorder; invalid permutations are atomic. |
| Execute cell/all | Pass | Ordered code execution, markdown skip, empty-cell behavior, shared kernel state, stop-on-failure, counts, and rich outputs. |
| Output persistence/clear | Pass | stdout, stderr, text, PNG, HTML, error, clear-one, clear-all, already-empty, missing, foreign, and other-notebook isolation. |
| Lifecycle/status | Pass | Before/after creation, interrupt, restart, shutdown, repeated shutdown, and controlled manager failures. |
| Latest drafts/autosave | Pass | Per-cell serialization preserves newest content; Run and Shift+Enter persist first; Run All flushes all drafts and save failure blocks execution. |
| Status/error UI | Pass | Real polling, unmount cleanup, state-based controls, save state, and accessible visible errors are covered. |
| Export | Pass | HTML metadata/source/output escaping and Jupyter stdout/stderr/display/result/error/image/HTML/markdown mapping; nbformat validation succeeds. |
| Cosmetic controls | Pass | Nonfunctional Save/More controls are absent; visible notebook controls invoke real actions. |

## Genuine Phase 1 defects fixed

- Added authenticated Execute All and Clear All backend contracts.
- Preserved explicit nullable descriptions and rejected unknown request fields.
- Preserved stderr stream identity.
- Prevented non-serializable validation details from turning intended `422` responses into `500` responses.
- Removed exception-message/traceback logging that could expose secrets.
- Made draft-save failures reject and block stale execution.
- Added real kernel polling, lifecycle-aware controls, clear/shutdown actions, and visible save/error states.
- Hardened HTML output and exports; added valid Jupyter cell IDs.
- Replaced remote build-time Google font fetching with local/system font styling.
- Added production fail-fast configuration and exact configured CORS origins.

## Deployment blockers

- No code/test blocker remains for private staging.
- **Mandatory operator gate:** the MongoDB credential formerly present in tracked `.env.example` must be treated as compromised. Revoke/rotate it, review repository history and access logs, and record completion before deploying. Removing it from the current tree is not sufficient.

## Non-blocking known issues

- Full frontend lint has a pre-existing unrelated backlog (55 errors, 14 warnings); the Phase 1 changed-file lint is clean.
- Full backend formatting has a pre-existing repository-wide backlog; changed files are clean.
- FastAPI and python-jose emit nine deprecation warnings in the passing backend suite.
- The local/global Python environment has unrelated undeclared LangGraph packages with an unsatisfied dependency; use an isolated deployment environment installed only from the repository lock/requirements.
- Cookie security is not applicable: notebook authentication uses bearer tokens.
- The private-staging smoke checklist is documented but requires a deployed staging environment and two disposable users to execute.

## Approved deployment scope

- Local development: approved.
- Private staging with trusted users: approved after the credential operator gate and successful smoke checklist.
- Public multi-user production: **not approved**.

## Commands used

```text
py -m pytest -q
py -m pytest -q tests/test_python_lab_phase1_release.py
py -m pytest -q tests/test_phase1_configuration_and_exports.py
npm.cmd run test
npm.cmd run typecheck
npm.cmd run build
npx.cmd eslint <Phase-1 changed frontend files>
npm.cmd run lint
py -m black <Phase-1 changed backend files>
py -m isort --check-only <Phase-1 changed backend files>
py -c "import app.main; print('backend import ok')"
npm.cmd audit --json
npm.cmd audit fix
npm.cmd dedupe
py -m pip check
py -m pip_audit -r requirements.txt
git diff --check
```
