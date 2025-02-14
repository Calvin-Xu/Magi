# syntax=docker/dockerfile:1.9

###
# 1. Dev Stage
###
FROM ubuntu:noble AS dev
SHELL ["sh", "-exc"]

# 1a. System-level deps
RUN apt-get update -qy && \
    apt-get install -qyy \
    software-properties-common \
    build-essential \
    ca-certificates \
    curl \
    git \
    python3-setuptools \
    cmake pkg-config libssl-dev libffi-dev libpq-dev \
    libblas-dev liblapack-dev libopenblas-dev gfortran \
    zlib1g-dev libbz2-dev liblzma-dev \
    openjdk-17-jdk

# Install Python 3.11
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update -qy && \
    apt-get install -qyy python3.11 python3.11-dev

# 1b. Bring in uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 1c. Environment setup
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3.11 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONPATH=/app/src:/app/.venv/lib/python3.11/site-packages \
    PATH=/app/.venv/bin:$PATH

# Create app directory and set as workdir
RUN mkdir -p /app/src
WORKDIR /app

# 1d. Copy project files and install dependencies
COPY pyproject.toml uv.lock ./

# Create venv and install all dependencies
RUN --mount=type=cache,target=/root/.cache \
    uv venv && \
    # uv pip install -e ".[dev]"
    uv sync --locked

EXPOSE 1998

# 1f. Start the development server
CMD ["uv", "run", "python", "-m", "magi.main"]


###
# 2. Production Build Stage
###
FROM ubuntu:noble AS build
SHELL ["sh", "-exc"]

RUN apt-get update -qy && \
    apt-get install -qyy \
    software-properties-common \
    build-essential \
    ca-certificates \
    curl \
    python3-setuptools \
    git \
    cmake \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    gfortran \
    openjdk-17-jdk \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev

RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update -qy && \
    apt-get install -qyy python3.11-dev

# Bring in uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3.11 \
    UV_PROJECT_ENVIRONMENT=/app

# 2a. Synchronize (prod) dependencies from pyproject.toml & uv.lock
RUN --mount=type=cache,target=/root/.cache \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --locked --no-dev --no-install-project

# 2b. Copy in the actual source code for the production build
COPY . /src
WORKDIR /src

RUN --mount=type=cache,target=/root/.cache \
    uv sync --locked --no-dev --no-editable


###
# 3. Production Runtime Stage
###
FROM ubuntu:noble
SHELL ["sh", "-exc"]

ENV PATH=/app/bin:$PATH

# Don't run your app as root
RUN groupadd -r app && useradd -r -d /app -g app -N app

# See https://hynek.me/articles/docker-signals/
STOPSIGNAL SIGINT

# Minimal runtime dependencies
RUN apt-get update -qy && \
    apt-get install -qyy software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update -qy && \
    apt-get install -qyy \
    python3.11 \
    libpython3.11 \
    libssl3 \
    libpq5 \
    libopenblas0 \
    liblapack3 \
    gfortran \
    zlib1g \
    libbz2-1.0 \
    liblzma5 \
    openjdk-17-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy entrypoint and make it runnable
COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]

# Bring the pre-built /app from build stage
COPY --from=build --chown=app:app /app /app

EXPOSE 8080
USER app
WORKDIR /app

# Optionally, smoke test
RUN python -V && \
    python -Im site && \
    python -Ic 'import magi'

CMD ["python", "-m", "magi.main"]
