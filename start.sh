#!/bin/sh
set -eu

echo "[Startup] Starting NxZenAI backend..."

echo "[Startup] Starting local Llama service in background..."
python /app/scripts/start_llama.py &

echo "[Startup] Starting FastAPI on port ${PORT:-8080}..."
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8080}"