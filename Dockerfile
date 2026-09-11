# Custom Chatterbox Multilingual TTS worker for RunPod Serverless.
#
# Built by GitHub Actions (.github/workflows/build.yml) and published to GHCR:
#   ghcr.io/kirushanetypi/tts-serverless-worker:latest
# The VPS cannot build it locally (~2 GB free disk), so CI does the build.
#
# Base image carries torch 2.6.0 + CUDA 12.4 runtime, which is exactly what
# chatterbox-tts 0.1.7 pins (torch==2.6.0, torchaudio==2.6.0), so pip only has
# to add torchaudio + the TTS dependency tree instead of a second torch.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg: mp3 encoding of the response. libsndfile1: soundfile/flac decoding of
# a caller-supplied reference clip. git: some dependency specs still use git+.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install -r /app/requirements.txt \
    && python3 -c "import torch, torchaudio, chatterbox.mtl_tts; print('torch', torch.__version__, 'torchaudio', torchaudio.__version__)"

COPY model_store.py tts_utils.py handler.py /app/

# HF_HOME is resolved at runtime by model_store.resolve_hf_home():
#   1. an HF_HOME RunPod already exported (cached-model hosts),
#   2. /runpod-volume/huggingface-cache when that mount exists,
#   3. /tmp/hf-home otherwise (weights live on the container disk).
# DEFAULT_LANGUAGE selects the language_id used when a job omits it.
ENV DEFAULT_LANGUAGE=ru \
    CHATTERBOX_T3_MODEL=v3 \
    CHATTERBOX_REPO=ResembleAI/chatterbox \
    ENABLE_WATERMARK=1 \
    FALLBACK_HF_HOME=/tmp/hf-home

CMD ["python3", "-u", "/app/handler.py"]
