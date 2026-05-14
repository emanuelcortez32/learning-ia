# AGENTS.md

Guidance for coding agents working at the `learning-ia` repository root.

## Scope and ownership

- Treat this repository as a monorepo of independent projects.
- Keep changes scoped to the target project folder.
- Do not make cross-project refactors unless explicitly requested.

## Directory map

- `generative-ai/simple-ai-agent`: FastAPI + LangChain agent API.
- `generative-ai/simple-mcp-server`: FastMCP server implementation.
- `machine-learning/*`: isolated ML and notebook exercises.

## Execution rules

1. Run commands from the specific project directory, not from repo root.
2. Use each project's `Makefile` targets when available (`setup`, `info`, and project-specific targets like `dev`/`test`).
3. For Docker workflows, only use compose files inside the project being changed.

## Documentation rules

- If behavior, setup, or commands change in a subproject, update that subproject's `README.md`.
- Keep this root README focused on navigation and high-level usage.
- Respect nested agent instructions:
  - `generative-ai/simple-mcp-server/AGENTS.md`
  - `generative-ai/simple-ai-agent/AGENT.md`

## Change quality expectations

- Prefer surgical edits; avoid unrelated cleanup.
- Preserve existing project conventions and file structure.
- Do not introduce new top-level tooling unless explicitly requested.
