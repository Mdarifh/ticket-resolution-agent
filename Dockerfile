# syntax=docker/dockerfile:1
#
# One image definition, three runtime targets (see docker-compose.yml):
#   api   FastAPI + LangGraph workflow + Playwright/Chromium for UI tests
#   ui    Streamlit dashboard
#   demo  Demo Shop (the system under test) and its auth API
#
# No secrets are baked in: .env is excluded by .dockerignore and passed at runtime
# through docker compose (env_file / environment).

ARG PYTHON_VERSION=3.14

# --- dependencies (shared, cached until requirements.txt changes) --------------
FROM python:${PYTHON_VERSION}-slim AS deps
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt \
    && useradd --create-home --uid 10001 app

# --- application source ------------------------------------------------------
FROM deps AS source
COPY alembic.ini ./
COPY migrations ./migrations
COPY src ./src
COPY knowledge_base ./knowledge_base
COPY web ./web
COPY demo_app ./demo_app
COPY .streamlit ./.streamlit
COPY --chmod=755 docker/api-entrypoint.sh /usr/local/bin/api-entrypoint.sh
# Writable state (vector index, demo users) lives in a volume mounted here.
RUN mkdir -p /app/data && chown app:app /app/data

# --- api: FastAPI + Chromium for Playwright UI tests -------------------------
FROM deps AS api
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN python -m playwright install --with-deps chromium \
    && chmod -R a+rX /ms-playwright \
    && rm -rf /var/lib/apt/lists/*
COPY --from=source /app /app
COPY --from=source /usr/local/bin/api-entrypoint.sh /usr/local/bin/api-entrypoint.sh
# COPY --from resets ownership to root; the data volume must stay writable for "app".
RUN chown app:app /app/data
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status != 200)"
ENTRYPOINT ["api-entrypoint.sh"]

# --- ui: Streamlit dashboard -------------------------------------------------
FROM source AS ui
USER app
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status != 200)"
CMD ["streamlit", "run", "src/qa_agent/ui/Home.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", \
     "--browser.gatherUsageStats=false"]

# --- demo: the system under test ---------------------------------------------
FROM source AS demo
USER app
EXPOSE 8765
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8765/login.html', timeout=4).status != 200)"
CMD ["python", "-m", "qa_agent.tools.demo_server", "--host", "0.0.0.0", "--port", "8765", \
     "--db", "/app/data/demo_users.db", "--public-url", "http://127.0.0.1:8765"]
