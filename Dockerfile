# ─────────────────────────────────────────────────────────────────────────────
# Mirage Verification API — Production Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
#
# Multi-stage build:
#   Stage 1 (builder): Install ALL Python dependencies including CPU-only torch.
#   Stage 2 (runtime): Copy installed packages + source. Pre-download spaCy
#                      model and NLI model so ZERO downloads happen at runtime.
#
# Base image: python:3.11-slim  (~200 MB vs ~900 MB for full python:3.11)
# Final image size (with pre-baked models): ~1.4–1.6 GB
#
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System dependencies (required by spaCy and sentence-transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip --no-cache-dir

# ── Install CPU-only torch FIRST (must precede sentence-transformers) ────────
# Without --index-url the default wheel is GPU-enabled (~2.5 GB).
# CPU wheel is ~250 MB — a 10x reduction.
RUN pip install --no-cache-dir \
    torch==2.3.1 \
    --index-url https://download.pytorch.org/whl/cpu

# ── Install remaining Python dependencies ───────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Download spaCy model (bundled into image — no runtime download) ──────────
RUN python -m spacy download en_core_web_sm

# ── Pre-download NLI model (bundled into image — no runtime download) ────────
# This makes cold starts nearly instant. HF_HOME controls the cache location.
ENV HF_HOME=/root/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/root/.cache/sentence-transformers
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

RUN python -c "\
from sentence_transformers import CrossEncoder; \
m = CrossEncoder('cross-encoder/nli-deberta-v3-base', device='cpu'); \
print('NLI model pre-downloaded successfully.') \
"

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system libs (libgomp1 needed by PyTorch CPU)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy pre-downloaded model caches from builder
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Copy application source
COPY backend/ ./backend/
COPY verification/ ./verification/
COPY .env.example .env.example

# Environment — point to baked-in model cache
ENV HF_HOME=/root/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/root/.cache/sentence-transformers
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Railway injects $PORT at runtime
EXPOSE 8080

# Health check — Railway uses this to determine readiness


# Use 1 worker — DeBERTa is memory-hungry; multiple workers would OOM.
# --timeout 120 matches VERIFY_TIMEOUT env var.
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 120 --log-level info"]
