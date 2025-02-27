# =======================
#  1) Builder Stage
# =======================
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv==0.6.3

WORKDIR /app

COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY backend/src/ ./src/
RUN uv sync --frozen --no-dev


# =======================
#  2) Final Stage
# =======================
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    libssl-dev \
    openjdk-17-jre \
    procps \
    gcc \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="$JAVA_HOME/bin:$PATH:/app/.venv/bin"

WORKDIR /app

COPY --from=builder /app /app

RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["uvicorn", "magi.main:app", "--host", "0.0.0.0", "--port", "8000"]
