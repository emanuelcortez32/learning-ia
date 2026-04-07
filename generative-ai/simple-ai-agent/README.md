# Simple AI Agent

A production-ready AI Agent API built with FastAPI, LangChain, and Ollama. This project demonstrates how to create an intelligent conversational agent with streaming support, designed for e-commerce sales assistance.

## 🌟 Features

- **FastAPI Framework**: High-performance REST API with automatic OpenAPI documentation
- **LangChain Integration**: Flexible agent architecture with prompt templates and chains
- **Ollama Support**: Local LLM inference with llama3.2:3b model
- **Streaming Responses**: Real-time Server-Sent Events (SSE) for chat streaming
- **Docker Compose**: Easy setup with Ollama and Open WebUI containers
- **Modular Architecture**: Clean separation of concerns with agents, models, and services
- **Type Safety**: Full Pydantic model validation
- **Sales Assistant Agent**: Pre-configured agent for clothing sales assistance

## 📋 Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Docker and Docker Compose (for Ollama)
- 8GB+ RAM recommended for local LLM inference

## 🚀 Quick Start

### 1. Install Dependencies

First, install uv if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or via pip
pip install uv
```

Then run the setup script:

```bash
./setup.sh
```

### 2. Start Ollama Services

Start the Ollama and Open WebUI containers:

```bash
docker-compose up -d
```

This will:
- Start Ollama server on port 11434
- Pull the llama3.2:3b model
- Start Open WebUI on port 3000

### 3. Run the Application

```bash
./run-app.sh
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Open WebUI**: http://localhost:3000

## 📡 API Endpoints

### Health Check

```bash
GET /
GET /health
```

### Standard Chat

```bash
POST /chat
Content-Type: application/json

{
  "query": "I'm looking for a summer dress"
}
```

Response:
```json
{
  "response": "I'd be happy to help you find the perfect summer dress! ..."
}
```

### Streaming Chat

```bash
POST /chat/stream
Content-Type: application/json

{
  "query": "What colors are popular this season?"
}
```

Returns Server-Sent Events (SSE) stream:
```
data: {"content": "This "}
data: {"content": "season, "}
data: {"content": "popular "}
...
data: [DONE]
```

## 🏗️ Project Structure

```
simple-ai-agent/
├── src/
│   ├── agents/
│   │   └── SalesAssistantAgent.py    # Sales assistant agent configuration
│   ├── lib/
│   │   └── ai/
│   │       └── agent/
│   │           ├── BaseAgent.py       # Base agent class
│   │           ├── AgentChat.py       # Chat agent implementation
│   │           └── AgentLlm.py        # LLM utilities
│   ├── models/
│   │   ├── ChatRequestModel.py       # Request schemas
│   │   └── ChatResponseModel.py      # Response schemas
│   ├── controllers/                   # API controllers (empty)
│   ├── repositories/                  # Data repositories (empty)
│   ├── services/                      # Business logic (empty)
│   └── app.py                         # FastAPI application
├── tests/                             # Test suite
├── docker-compose.yml                 # Docker services configuration
├── pyproject.toml                     # Project dependencies
├── setup.sh                           # Setup script
├── run-app.sh                         # Run script
└── .env                              # Environment variables
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Ollama Configuration
OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_NUM_PARALLEL=4
LLM_MODEL=llama3.2:3b
```

### Model Configuration

The default configuration uses `llama3.2:3b` model. To use a different model:

1. Update `.env` file:
```env
LLM_MODEL=llama3.2:1b  # or any other Ollama model
```

2. Update `src/agents/SalesAssistantAgent.py`:
```python
LLM_MODEL = "llama3.2:1b"  # Match your .env
```

3. Restart the services:
```bash
docker-compose down
docker-compose up -d
```

## 🧪 Testing

Run the test suite:

```bash
./run-preintegration.sh
```

Or manually:

```bash
uv run pytest tests/
```

## 📚 Usage Examples

### Python Client

```python
import requests
import json

# Standard chat
response = requests.post(
    "http://localhost:8000/chat",
    json={"query": "I need a formal shirt for an interview"}
)
print(response.json()["response"])

# Streaming chat
response = requests.post(
    "http://localhost:8000/chat/stream",
    json={"query": "What's your return policy?"},
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = line[6:]
            if data != '[DONE]':
                chunk = json.loads(data)
                print(chunk['content'], end='', flush=True)
```

### cURL

```bash
# Standard chat
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me casual shirts"}'

# Streaming chat
curl -X POST "http://localhost:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"query": "What sizes do you have?"}' \
  --no-buffer
```

## 🎯 Creating Custom Agents

See [AGENT.md](./AGENT.md) for detailed information on creating and customizing agents.

## 🐳 Docker Configuration

### GPU Support (NVIDIA)

To enable GPU acceleration, uncomment the deploy section in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

### Resource Limits

Adjust Ollama settings in `.env`:

```env
OLLAMA_MAX_LOADED_MODELS=2     # Number of models to keep in memory
OLLAMA_NUM_PARALLEL=8          # Parallel requests
```

## 🔍 Troubleshooting

### Port Already in Use

If port 8000 is busy, modify `src/app.py`:

```python
uvicorn.run(app, host="0.0.0.0", port=8001)
```

### Ollama Connection Error

Ensure Ollama is running:

```bash
docker-compose ps
curl http://localhost:11434/api/tags
```

### Model Not Found

Pull the model manually:

```bash
docker exec -it ollama ollama pull llama3.2:3b
```

## 📈 Performance Considerations

- **Response Time**: 2-5 seconds for standard chat (CPU)
- **Streaming**: ~50-100 tokens/second (CPU)
- **Memory**: ~4GB for llama3.2:3b model
- **Concurrent Requests**: Configured via `OLLAMA_NUM_PARALLEL`

## 🛠️ Development

### Adding New Endpoints

1. Define models in `src/models/`
2. Add endpoint to `src/app.py`
3. Update tests in `tests/`

### Creating New Agents

1. Create agent class in `src/agents/`
2. Extend `BaseAgent` or use `AgentChat`
3. Define system prompt and model configuration
4. Import and use in `app.py`

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🔗 Related Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Models](https://ollama.ai/library)
- [uv Package Manager](https://github.com/astral-sh/uv)

## 📧 Support

For issues and questions, please open an issue on GitHub.
