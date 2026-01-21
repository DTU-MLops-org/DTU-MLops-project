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

EXPOSE 8002
ENTRYPOINT ["uv", "run", "uvicorn", "--port", "8002", "--host", "0.0.0.0", "--app-dir", "src", "mlops.backend:app"]