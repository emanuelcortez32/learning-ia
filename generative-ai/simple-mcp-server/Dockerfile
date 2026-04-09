# ---------- Builder ----------
FROM python:3.12-slim AS builder

WORKDIR /server

RUN pip install uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

# ---------- Runtime ----------
FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash --uid 1001 appuser

WORKDIR /server

COPY --from=builder /server/.venv /server/.venv
COPY src/ .

COPY pyproject.toml uv.lock ./

RUN chown -R appuser:appuser /server

USER appuser

ENV PATH="/server/.venv/bin:$PATH"
ENV PYTHONPATH="/server"

EXPOSE 8088

CMD ["python", "server.py"]