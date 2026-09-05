#FROM python:3.11-slim AS builder
#WORKDIR /build
#COPY pyproject.toml .
#COPY src/ src/
#RUN pip install --no-cache-dir --prefix=/install .
#
#FROM python:3.11-slim
#WORKDIR /app
#COPY --from=builder /install /usr/local
#COPY . .
#ENTRYPOINT ["train"]
#FROM python:3.11-slim AS builder
#WORKDIR /build
#COPY pyproject.toml .
#COPY src/ src/
#RUN pip install --no-cache-dir --prefix=/install .
#
#
#FROM python:3.11-slim
#WORKDIR /app
#COPY --from=builder /install /usr/local
#COPY configs/ configs/
#ENTRYPOINT ["train"]

# Multi-stage build: keep the final image free of build-only tooling.
FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir --prefix=/install .


FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /install /usr/local
# Only configs/ is copied here — src/ is already installed into
# site-packages via the builder stage's `pip install .` (non-editable),
# and data/results/logs are volume-mounted at runtime by
# docker-compose.yaml, not baked into the image.
COPY configs/ configs/
# NOTE: no CONFIG_PATH env var needed — src/training/run.py's main()
# resolves the config directory from the current working directory
# (WORKDIR /app here) at runtime, not from an env var. An earlier fix
# attempt set CONFIG_PATH here; left as dead config until superseded by
# the current cwd-based approach, then removed.

ENTRYPOINT ["train"]