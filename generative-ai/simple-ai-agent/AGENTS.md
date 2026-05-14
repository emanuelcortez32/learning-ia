# AGENTS.md - Simple AI Agent

Guidance for contributors and coding agents working in `generative-ai/simple-ai-agent`.

## Scope

- Keep changes local to this subproject.
- Do not refactor other folders in the monorepo unless explicitly requested.

## Tech and entrypoints

- Runtime: Python + FastAPI (`src/main.py`)
- Agent wiring: `src/agents/WikipediaAgent.py`
- Agent implementation: `src/models/AgentChat.py`
- API routers:
  - `src/controllers/health_controller.py`
  - `src/controllers/chat_controller.py`

## Local workflow

Use Make targets from this folder:

```bash
make setup
make dev
make test
make docker-up-all
```

Notes:

- `make docker-up` is currently a placeholder in `scripts/run-docker.sh`.
- `make test` currently runs pytest with coverage; there are no committed tests yet.

## Coding conventions

1. Keep edits surgical and behavior-focused.
2. Preserve current folder/module structure.
3. Reuse existing patterns in `controllers`, `models`, and `agents`.
4. Avoid broad exception swallowing; surface meaningful errors.
5. Update `README.md` when API behavior, commands, or setup changes.

## Adding a new agent

1. Create a module in `src/agents/` (e.g., `MyAgent.py`) with prompt, model, and exported instance.
2. Reuse `AgentChat` from `src/models/AgentChat.py`.
3. Add/adjust endpoints in `src/controllers/chat_controller.py`.
4. If response shape changes, update:
   - `src/models/ChatResponseModel.py`
   - `src/lib/response_wrapper.py`
   - `README.md`

## Streaming contract

`POST /chat/stream` returns SSE messages in this format:

- `data: {"type":"content","content":"..."}`
- `data: {"type":"tool_call","tools":["tool_name"]}`
- `data: [DONE]`

Keep this contract backward compatible unless a breaking change is intentional and documented.

## Config and environment

App config lives in `src/config/config.py` and reads `.env` via `pydantic-settings`.

Primary variables:

- `HOST`
- `PORT`
- `LLM_MODEL`
- `OLLAMA_NUM_PARALLEL`
- `OLLAMA_MAX_LOADED_MODELS`
