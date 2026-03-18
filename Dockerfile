FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    CHARACTER_ENG_RUNTIME_ROOT=/workspace/character-eng-runtime \
    CHARACTER_ENG_BUNDLED_VISION_ENV=/opt/character-eng/vision-env \
    CHARACTER_ENG_CACHE_ROOT=/workspace/character-eng-runtime/.cache \
    HF_HOME=/workspace/character-eng-runtime/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/character-eng-runtime/.cache/huggingface/hub \
    TRANSFORMERS_CACHE=/workspace/character-eng-runtime/.cache/huggingface/transformers \
    TORCH_HOME=/workspace/character-eng-runtime/.cache/torch \
    XDG_CACHE_HOME=/workspace/character-eng-runtime/.cache/xdg \
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

RUN mkdir -p /workspace/character-eng-runtime/.cache /opt/character-eng
RUN UV_PROJECT_ENVIRONMENT=/opt/character-eng/vision-env \
    UV_CACHE_DIR=/tmp/uv-cache/vision \
    uv sync --project services/vision --frozen \
    && rm -rf /tmp/uv-cache/vision
RUN UV_CACHE_DIR=/tmp/uv-cache/app uv sync --frozen --no-dev --extra pocket-server \
    && rm -rf /tmp/uv-cache/app

EXPOSE 7862 7870 7860 8000 8003

ENTRYPOINT ["/app/deploy/container_entrypoint.sh"]
