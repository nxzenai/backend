# Python Lab architecture and delivery plan

## Current boundary

The existing public notebook API remains under `/api/v1/notebooks` for
compatibility. MongoDB stores notebook metadata and lightweight embedded cell
state. A development-only Jupyter manager currently provides one in-memory
kernel per notebook in a single API process.

Host Jupyter execution is not a production sandbox. It must not be enabled for
untrusted users in production: it has no container boundary, shared runtime
registry, filesystem isolation, network policy, or enforced resource quotas.

## Target modules

- **Python projects** own workspace metadata, members, environment selection,
  quotas, lifecycle state, and storage roots.
- **Project files** store normalized project-relative paths, revisions, media
  type, size, checksum, and an object-storage reference. Client-supplied paths
  are never passed directly to the host filesystem.
- **Notebooks** store notebook metadata and independently revisioned cells.
  Large outputs and artifacts are stored outside MongoDB.
- **Runtimes** store durable runtime identity, provider identity, state, lease,
  heartbeat, expiry, environment, and resource policy.
- **Executions** store queued job state, source revision, timestamps, duration,
  bounded logs/output references, and failure details.
- **Environments** are immutable, versioned runtime profiles. Package changes
  create a user/project-scoped environment rather than modifying a shared one.
- **Artifacts** point to datasets, images, reports, models, and checkpoints in
  object storage.

## Provider and coordination design

`RuntimeProvider` is the application boundary for create, execute, interrupt,
restart, status, and shutdown operations. A `LocalDevelopmentRuntimeProvider`
may wrap Jupyter for local development and must refuse to start when the
application environment is production. A production provider must launch a
non-root isolated container, pod, or microVM with explicit CPU, memory, PID,
disk, time, output, filesystem, and network limits.

Redis-compatible shared coordination should own runtime leases, per-runtime
queues and locks, heartbeats, and event fan-out. MongoDB remains the durable job
and runtime record. SSE is sufficient for the first authenticated event stream;
WebSocket support can follow where bidirectional runtime features require it.

Runtime states are `starting`, `ready`, `busy`, `interrupting`, `restarting`,
`stopping`, `stopped`, `failed`, and `expired`. Execution states are `queued`,
`running`, `succeeded`, `failed`, `cancelled`, and `timed_out`.

## Compatibility and migration

Existing notebook IDs and routes remain valid during migration. The migration
creates one default Python project per owner, records the project ID on each
legacy notebook, converts embedded cells into independently revisioned records,
and moves oversized outputs to object storage. It is idempotent and records a
migration version and old-to-new identifiers. Reads support unmigrated documents
until verification is complete; existing notebooks never disappear solely
because a migration has not run.

## Delivery phases

1. **Security and correctness:** authenticate and authorize execution, align
   schemas, return stable errors, protect frontend routes, isolate untrusted
   HTML, correct notebook export, and add contract tests.
2. **Runtime abstraction:** introduce providers and durable jobs, move execution
   outside the API process, enforce limits, add shared coordination, cleanup,
   shutdown, recovery, and authenticated event streaming.
3. **Projects and files:** add safe project-relative storage APIs, object
   storage, uploads/downloads, project explorer, tabs, Monaco `.py` editing,
   Run File, and an execution console.
4. **Notebook reliability:** add cell revisions, optimistic concurrency,
   flush-before-run, complete cell commands, valid `nbformat` import/export,
   runtime state, and streaming output.
5. **Versioned environments:** add reproducible base, data-science, ML, DL CPU,
   and NLP images plus policy-controlled package workflows. GPU is enabled only
   with real scheduling, drivers, quotas, and availability checks.
6. **Production hardening:** isolation and abuse testing, quotas, observability,
   load/failure tests, migration rollout, backups, and operations documentation.

## Phase 1 status

The current security slice authenticates all execution endpoints, verifies
ownership without revealing foreign private notebook IDs, removes duplicate API
schemas, aligns notebook update fields, persists notebook execution counts,
makes interrupt independent of the execution lock, moves blocking Jupyter calls
off the FastAPI event loop, protects the editor route, and renders sanitized HTML
inside a script-disabled iframe. Runtime isolation, shared coordination, full
`nbformat` import, revision-safe autosave, output limits, and project files remain
future work and are not represented as production-ready.
