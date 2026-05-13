# ---------- Builder ----------
FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

# ---------- Runtime ----------
FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash --uid 1001 appuser

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ .

COPY pyproject.toml uv.lock ./

RUN chown -R appuser:appuser /app

USER appuser

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

EXPOSE 8088

CMD ["python", "main.py"]