# CPU image for CI and for `/healthz` without a GPU.
# Serving the real ViT-L on a GPU uses the same package with a CUDA torch;
# do not bake the 1.3 GB checkpoint into a layer (see weights.lock).

FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_HOME=/opt/cache/torch \
    HF_HOME=/opt/cache/hf \
    SPUR_SKIP_WEIGHTS=1 \
    SPUR_HOST=0.0.0.0 \
    SPUR_PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md model_card.json weights.lock ./
COPY spur_depth ./spur_depth
COPY samples ./samples
COPY scripts ./scripts

# CPU torch first so pip does not pull a multi-GB CUDA wheel into CI.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e ".[serve]"

EXPOSE 8000
# One worker. Four processes each hold a ViT-L and the GPU serialises them.
CMD ["python", "-m", "spur_depth.serve"]
