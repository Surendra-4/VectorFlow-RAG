# Dockerfile — VectorFlow-RAG backend (FastAPI + hybrid-RAG pipeline).
#
# The Next.js frontend deploys separately (Vercel); this image is the API only.
# Build:  docker build -t vectorflow-rag .
# Run:    docker run -p 8000:8000 --env-file .env vectorflow-rag
#         (SQLite + local Ollama work out of the box; set DATABASE_URL +
#          provider keys via --env-file for Postgres / hosted models.)

FROM python:3.11-slim AS base

# System deps:
#   tesseract-ocr — the image-OCR document loader (pytesseract needs the binary)
#   libgomp1      — OpenMP runtime used by faiss-cpu
#   curl          — container HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so the (slow) ML-stack layer is cached across code changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code (backend only — no tests/docs/frontend in the runtime image).
COPY src ./src
COPY .env.example ./.env.example

# Run as a non-root user; pre-create the writable runtime dirs.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/var /app/indices \
    && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/var/hf \
    VFR_API__HOST=0.0.0.0 \
    VFR_API__PORT=8000

EXPOSE 8000

# Generous start-period: the first boot downloads the embedding model.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Factory entrypoint (the app is factory-only by design — no import-time model load).
CMD ["uvicorn", "src.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
