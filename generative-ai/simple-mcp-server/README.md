# simple-mcp-server

Minimal MCP server built with **FastMCP** using **streamable-http** transport.

## What this project includes

- MCP server entrypoint: `src/main.py`
- Health endpoint: `GET /health` (returns `OK`)
- Example MCP tool: `say_hello(name: str)`
- Local run with `uv`
- Docker Compose stack with:
  - `simple-mcp-server`
  - `mcp-inspector`
  - `mcpo`
  - `ollama`
  - `open-webui`

## Requirements

- Python `>=3.10`
- [`uv`](https://docs.astral.sh/uv/)
- (Optional) Docker + Docker Compose

## Local development

```bash
make setup
make dev
```

The server runs on:

- `http://localhost:8088/health`
- MCP endpoint: `http://localhost:8088/mcp`

## Environment variables

The app reads `.env` with these keys (defaults shown):

```env
HOST=0.0.0.0
PORT=8088
```

## Make commands

```bash
make setup        # install dependencies with uv
make dev          # run app locally
make test         # run pytest + coverage
make docker-up    # start app + mcp-inspector
make docker-up-all# start full stack
make docker-down  # stop containers
make docker-clean # cleanup docker resources
make clean        # remove local temporary artifacts
```

## Docker

Start minimal MCP stack:

```bash
make docker-up
```

Start full stack:

```bash
make docker-up-all
```

Useful ports:

- `8088` MCP server
- `6789` MCP Inspector client
- `6790` MCP Inspector server
- `8000` MCPO
- `11434` Ollama
- `3000` Open WebUI

## Project layout

```text
src/
  main.py
  config/
  controllers/
  tools/
  lib/
  utils/
scripts/
tests/
```
