#!/bin/sh
# API container startup: migrate the database, build the knowledge base index on first
# start, then serve FastAPI. Postgres readiness is handled by the compose healthcheck
# (depends_on: condition: service_healthy); the migration retries a few times anyway.
set -eu

if [ -n "${DATABASE_URL:-}" ]; then
  attempt=1
  until alembic upgrade head; do
    if [ "$attempt" -ge 5 ]; then
      echo "Database migration failed after $attempt attempts" >&2
      exit 1
    fi
    echo "Migration failed (attempt $attempt), retrying in 3s..." >&2
    attempt=$((attempt + 1))
    sleep 3
  done
else
  echo "DATABASE_URL is not set: running without persistence" >&2
fi

index_dir="${CHROMA_PERSIST_DIR:-data/chroma}"
if [ -z "$(ls -A "$index_dir" 2>/dev/null)" ]; then
  echo "Building the knowledge base index in $index_dir"
  python -m qa_agent.rag.ingestion \
    || echo "WARNING: knowledge base indexing failed; retrieval will be empty until it succeeds" >&2
fi

exec uvicorn qa_agent.api.main:app --app-dir src \
  --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips="*" --no-server-header
