# Simple AI Agent

FastAPI API that exposes a LangChain agent backed by Ollama. The default agent is `WikipediaAgent`, with standard and streaming chat endpoints.

## Stack

- Python 3.10+
- FastAPI
- LangChain + LangGraph
- Ollama (`llama3.2:3b` by default)
- `uv` for dependency management

## Quick start

```bash
make setup
make docker-up-all
make dev
```

API URLs:

- API root: `http://localhost:8000/`
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Make targets

```bash
make info          # Show project name/version
make setup         # Install dependencies with uv
make dev           # Run API (uv run python src/main.py)
make test          # Run pytest + coverage
make docker-up-all # Build and start Docker services
make docker-down   # Stop Docker services
make docker-clean  # Remove containers/volumes and prune
make clean         # Remove local build/cache artifacts
```

`make docker-up` exists but is currently a placeholder (`Pendiente de implementacion`) in `scripts/run-docker.sh`.

## API endpoints

### `GET /`

Returns:

```json
{ "message": "AI Agent API is running" }
```

### `GET /health/`

Returns:

```json
{ "status": "healthy" }
```

### `POST /chat/`

Request:

```json
{ "query": "Who is Ada Lovelace?" }
```

Response shape:

```json
{
  "content": "...",
  "message": "Success",
  "status": true,
  "error": null
}
```

### `POST /chat/stream`

Server-Sent Events (SSE) stream. Event payloads include:

- `{"type":"content","content":"..."}`
- `{"type":"tool_call","tools":["..."]}`
- `[DONE]` terminator

Example:

```bash
curl -N -X POST "http://localhost:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"query":"Tell me about Alan Turing"}'
```

## Configuration

The app reads `.env` values using `pydantic-settings`.

Application settings:

```env
HOST=0.0.0.0
PORT=8000
```

Ollama/docker settings:

```env
LLM_MODEL=llama3.2:3b
OLLAMA_NUM_PARALLEL=2
OLLAMA_MAX_LOADED_MODELS=1
```

Notes:

- Docker compose starts an Ollama container on port `11434`.
- On startup, Ollama pulls `${LLM_MODEL}` automatically.
- `src/agents/WikipediaAgent.py` currently hardcodes `LLM_MODEL = "llama3.2:3b"` in the Python agent configuration.

## Project layout

```text
simple-ai-agent/
├── src/
│   ├── main.py
│   ├── agents/
│   │   └── WikipediaAgent.py
│   ├── controllers/
│   │   ├── chat_controller.py
│   │   └── health_controller.py
│   ├── models/
│   │   ├── AgentChat.py
│   │   ├── ChatRequestModel.py
│   │   └── ChatResponseModel.py
│   ├── config/
│   │   └── config.py
│   └── lib/
│       └── response_wrapper.py
├── scripts/
│   ├── setup.sh
│   ├── run-app.sh
│   ├── run-tests.sh
│   └── run-docker.sh
├── docker-compose.yml
├── pyproject.toml
├── Makefile
└── AGENTS.md
```

## Agent customization

See [AGENTS.md](./AGENTS.md) for architecture and extension guidance.
