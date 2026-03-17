FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PATH="/root/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    libportaudio2 \
    libsndfile1 \
    lsof \
    portaudio19-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app
COPY . /app

RUN chmod +x /app/deploy/container_entrypoint.sh \
    /app/scripts/run_heavy.sh \
    /app/scripts/run_hot.sh \
    /app/scripts/run_local.sh \
    /app/services/vision/start.sh

RUN uv sync --frozen
RUN uv sync --project services/vision
RUN uv tool install vllm==0.16.0

EXPOSE 7862 7870 7860 8000 8003

ENTRYPOINT ["/app/deploy/container_entrypoint.sh"]
