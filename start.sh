#!/bin/sh

set -e

echo "[Startup] Starting NxZenAI backend..."

echo "[Startup] Starting local Llama service in background..."
python /app/scripts/start_llama.py &

echo "[Startup] Starting FastAPI..."

exec uvicorn main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8080}"
