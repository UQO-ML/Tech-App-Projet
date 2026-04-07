# syntax=docker/dockerfile:1.7
ARG CUDA_IMAGE_TAG=12.6.3-cudnn-runtime-ubuntu24.04
FROM nvidia/cuda:${CUDA_IMAGE_TAG}

ARG ENABLE_RAPIDS=0
ARG REQUIREMENTS_MODE=freeze
ARG USERNAME=app
ARG UID=1000
ARG GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    VENV_PATH=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    git \
    curl \
    ca-certificates \
    tini \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && python3 -m venv "${VENV_PATH}" \
    && python -m pip install --upgrade pip setuptools wheel uv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/build

COPY docker/requirements.base.txt /tmp/build/requirements.base.txt
COPY docker/requirements.rapids.txt /tmp/build/requirements.rapids.txt
COPY docker/requirements.base.lock.txt /tmp/build/requirements.base.lock.txt
COPY docker/requirements.rapids.lock.txt /tmp/build/requirements.rapids.lock.txt

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    if [ "${REQUIREMENTS_MODE}" = "freeze" ]; then \
      BASE_REQ="/tmp/build/requirements.base.lock.txt"; \
      RAPIDS_REQ="/tmp/build/requirements.rapids.lock.txt"; \
    else \
      BASE_REQ="/tmp/build/requirements.base.txt"; \
      RAPIDS_REQ="/tmp/build/requirements.rapids.txt"; \
    fi && \
    python -m pip install --prefer-binary -r "${BASE_REQ}" && \
    if [ "${ENABLE_RAPIDS}" = "1" ]; then \
      python -m pip install --prefer-binary -r "${RAPIDS_REQ}"; \
    fi

RUN if ! getent group "${GID}" >/dev/null; then \
      groupadd --gid "${GID}" "${USERNAME}"; \
    fi && \
    if ! id -u "${UID}" >/dev/null 2>&1; then \
      useradd --uid "${UID}" --gid "${GID}" -m "${USERNAME}"; \
    fi

WORKDIR /workspace
RUN chown -R "${UID}:${GID}" /workspace

USER ${UID}:${GID}

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["bash"]
