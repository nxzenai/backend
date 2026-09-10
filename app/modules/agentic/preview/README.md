# Agentic preview worker

Start the local lifecycle worker separately from the FastAPI process:

```powershell
py -m app.modules.agentic.preview.worker
```

The worker atomically claims `starting` preview records, prepares and runs the immutable source in Docker, performs localhost health checks, and cleans expired runtimes. Use `--once` for maintenance or tests.
