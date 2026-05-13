# AGENTS.md

Guidance for coding agents working in `generative-ai/simple-mcp-server`.

## Scope

- Keep changes focused on this service.
- Do not modify unrelated projects under the monorepo root.

## Tech stack

- Python + FastMCP
- Dependency management and execution via `uv`
- Docker Compose for local integration stack

## Key entry points

- App entrypoint: `src/main.py`
- Config: `src/config/config.py`
- Health controller: `src/controllers/health_controller.py`
- Tool registration:
  - `src/tools/hello/register.py`
  - `src/utils/tools_loader.py`
- Tool safety wrapper: `src/lib/tool_wrapper.py`

## Development workflow

1. Install dependencies: `make setup`
2. Run locally: `make dev`
3. Run tests: `make test`

For container workflows:

- `make docker-up` (MCP server + inspector)
- `make docker-up-all` (full stack)
- `make docker-down`

## Conventions

- Register MCP tools through `load_tools(...)` instead of ad hoc registration.
- Wrap tool functions with `@safe_tool` to preserve the expected response shape:
  - success: `{"success": true, "data": ...}`
  - error: `{"success": false, "error": {...}}`
- Keep async tool signatures (`async def`) consistent with existing tools.
- Preserve streamable HTTP setup in `main.py` unless the task explicitly requires changing transport.

## Documentation expectations

When behavior changes, update:

- `README.md` for setup/run/use changes
- `mcpo.config.json` only if MCP endpoint wiring changes
