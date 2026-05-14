# learning-ia

Monorepo with practical AI/ML study projects, split into **Generative AI services** and **Machine Learning notebooks**.

## Repository layout

| Path | Purpose |
| --- | --- |
| `generative-ai/simple-ai-agent` | FastAPI + LangChain + Ollama conversational agent API |
| `generative-ai/simple-mcp-server` | Minimal FastMCP server over streamable HTTP |
| `machine-learning/ml-*` | Focused ML algorithm exercises (one project per model) |
| `machine-learning/prompt-injector-*` | Prompt-injection related notebook experiments |

## Requirements

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/)
- `make`
- Docker + Docker Compose (only for projects that use containers)

## Working with a project

Each folder is self-contained. Run commands inside the target project directory:

```bash
cd <project-path>
make setup
make info
```

For service projects under `generative-ai/`, common commands include:

```bash
make dev
make test
```

For `machine-learning/` projects, the main artifact is `main.ipynb` and the default workflow is environment setup plus notebook execution.

## Project index

### Generative AI

- `simple-ai-agent` — AI Agent API with FastAPI and LangChain
- `simple-mcp-server` — MCP with MCPFast

### Machine Learning

- `ml-knn` — KNeighborsClassifier
- `ml-linear-regression` — LinearRegression
- `ml-linear-sgd` — SGDClassifier
- `ml-logistic-regression` — LogisticRegression
- `ml-naive-bayes` — MultinomialNB
- `ml-polynomial-regression` — Polynomial LinearRegression
- `ml-svc-classifier` — SVC
- `ml-svr-regression` — SVR
- `ml-tree-classifier` — TreeClassifier
- `ml-tree-regression` — Tree Regression
- `prompt-injector-defender` — Prompt Injector Defender
- `prompt-injector-translator` — Prompt Injector Translator

## Notes

- Some subprojects define their own `README.md` and/or `AGENTS.md` with project-specific details; follow local documentation first when working inside those folders.
