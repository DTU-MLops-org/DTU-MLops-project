FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY LICENSE LICENSE
COPY src/ src/
COPY models/model.pth models/model.pth

ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync

# Set Hugging Face cache directory for runtime model downloads
ENV HF_HOME=/tmp/huggingface_cache

ENTRYPOINT ["sh", "-c", "uv run uvicorn --host 0.0.0.0 --port ${PORT:-8003} --app-dir src mlops.monitoring_api:app"]
# EXPOSE 8003
# ENTRYPOINT ["uv", "run", "uvicorn", "--port", "8003", "--host", "0.0.0.0", "--app-dir", "src", "mlops.monitoring_api:app"]
