# ---- Base runtime
FROM python:3.12-slim

# LightGBM runtime dep; add others if you know you need them
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# App dir
WORKDIR /app

# (A) If you have pyproject.toml (recommended for caching deps)
# COPY pyproject.toml poetry.lock* ./
# RUN pip install --no-cache-dir .

# (B) If you use requirements.txt
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt


# Copy source
COPY src/ ./src/

# Copy model artifacts into the image
# (keeps relative path inside the container as /app/storage/artifacts/latest)
COPY storage/artifacts/latest ./storage/artifacts/latest

# Env for src-layout import
ENV PYTHONPATH=/app/src
ENV PORT=8000

# Non-root
RUN useradd -m appuser
USER appuser

# Healthcheck (hits /health)
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:' + str(__import__('os').environ.get('PORT',8000)) + '/health')" >/dev/null 2>&1 || exit 1

EXPOSE 8000
CMD ["uvicorn", "--app-dir", "src", "cpml.predictive.api:app", "--host", "0.0.0.0", "--port", "8000"]
