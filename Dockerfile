

FROM python:3.11-slim

#Metadata
LABEL maintainer="smart-meeting-scheduler"
LABEL description="Smart Meeting Scheduler OpenEnv Environment"

#Env vars
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

#System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

#To create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 appuser

WORKDIR /app

#To install Python dependencies first (layer-cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

#To copy application code
COPY --chown=appuser:appuser . .

#To switch to non-root user
USER appuser

#To expose HF Spaces port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

#To launch FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
