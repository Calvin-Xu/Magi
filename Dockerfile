# https://hynek.me/articles/docker-uv/
# syntax=docker/dockerfile:1.9

###
# 1. Build Stage
###
FROM ubuntu:noble AS build

# The following does not work in Podman unless you build in Docker
# compatibility mode: <https://github.com/containers/podman/issues/8477>
# You can manually prepend every RUN script with `set -ex` too.
SHELL ["sh", "-exc"]

### Start build prep.
### This should be a separate build container for better reuse.

RUN apt-get update -qy && \
    apt-get install -qyy \
      -o APT::Install-Recommends=false \
      -o APT::Install-Suggests=false \
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
    apt-get update -qy

RUN apt-get install -qyy python3.11-dev

# Copy uv from Astral's GHCR into /usr/local/bin/
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Environment variables for uv
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3.11 \
    UV_PROJECT_ENVIRONMENT=/app

### End build prep -- this is where your app Dockerfile should start.

# Synchronize DEPENDENCIES without the application itself.
# This layer is cached until uv.lock or pyproject.toml change, which are
# only temporarily mounted into the build container since we don't need
# them in the production one.
# You can create `/app` using `uv venv` in a separate `RUN`
# step to have it cached, but with uv it's so fast, it's not worth
# it, so we let `uv sync` create it for us automagically.
RUN --mount=type=cache,target=/root/.cache \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync \
        --locked \
        --no-dev \
        --no-install-project

# RUN ls -l /app && ls -l /app/bin || true && ls -l /app/.venv/bin || true

# Now install the rest from `/src`: The APPLICATION w/o dependencies.
# `/src` will NOT be copied into the runtime container.
# LEAVE THIS OUT if your application is NOT a proper Python package.
COPY . /src
WORKDIR /src
RUN --mount=type=cache,target=/root/.cache \
    uv sync \
        --locked \
        --no-dev \
        --no-editable

###
# 2. Runtime Stage
###
FROM ubuntu:noble
SHELL ["sh", "-exc"]

# Optional: add the application virtualenv to search path.
ENV PATH=/app/bin:$PATH
# ENV PATH="/app/bin:$PATH" \
#    PYTHONPATH="/app/lib/python3.11/site-packages:/app/src" \
#    UV_PYTHON=python3.11

# Don't run your app as root.
RUN <<EOT
groupadd -r app
useradd -r -d /app -g app -N app
EOT

ENTRYPOINT ["/docker-entrypoint.sh"]
# See <https://hynek.me/articles/docker-signals/>.
STOPSIGNAL SIGINT

# Note how the runtime dependencies differ from build-time ones.
# Notably, there is no uv either!
RUN <<EOT
apt-get update -qy
apt-get install -qyy software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update -qy
apt-get install -qyy \
    -o APT::Install-Recommends=false \
    -o APT::Install-Suggests=false \
    python3.11 \
    libpython3.11 \
    libssl3 \
    libpq5 \
    openjdk-17-jdk \
    libopenblas0 \
    liblapack3 \
    gfortran \
    zlib1g \
    libbz2-1.0 \
    liblzma5

apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
EOT

COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh

# Copy the pre-built `/app` directory to the runtime container
# and change the ownership to user app and group app in one step.
COPY --from=build --chown=app:app /app /app

EXPOSE 8080

# RUN chmod +x /usr/bin/python3.11

USER app
WORKDIR /app

# run a smoke test that the application can, in fact, be imported.
RUN <<EOT
python -V
python -Im site
python -Ic 'import magi'
EOT