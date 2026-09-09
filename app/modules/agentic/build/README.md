# Agentic local build worker

Agentic build requests are durable MongoDB jobs. The API never executes them. Start one
separate worker process from `backend-main`:

```powershell
py -m app.modules.agentic.build.worker
```

The worker requires Docker Desktop and the configured `AGENTIC_BUILD_IMAGE` to already be
available locally. The default can be prepared with:

```powershell
docker pull nikolaik/python-nodejs:python3.12-nodejs22-bookworm
```

Use `--once` to claim at most one job. Local P3 enforces one running build through a MongoDB
unique worker-slot lease. It never falls back to host execution.
