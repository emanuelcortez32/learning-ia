# Test Suite Documentation

## Overview
Comprehensive pytest test suite for the simple-ai-agent project with **86% code coverage** and **30 passing tests**.

## Test Structure

### Test Files Created

1. **tests/conftest.py** - Pytest fixtures for mocking
   - `mock_chat_llm`: Mock ChatOllama for testing
   - `mock_llm`: Mock base LLM for testing
   - `mock_ai_message`: Mock AIMessage responses

2. **tests/test_models.py** - Pydantic model validation tests
   - `TestChatRequestModel`: 4 tests for request validation
   - `TestChatResponseModel`: 3 tests for response validation

3. **tests/test_utils.py** - Utility function tests
   - `TestResponseWrapper`: 3 tests for API response formatting

4. **tests/test_agents.py** - Agent implementation tests
   - `TestAgentChatStructure`: Tests for ChatPromptTemplate structure
   - `TestAgentLlmStructure`: Tests for PromptTemplate structure

5. **tests/test_controllers.py** - FastAPI controller tests
   - `TestHealthController`: 1 test for health endpoint
   - `TestChatController`: 5 tests for chat endpoints (standard + streaming)

6. **tests/test_app.py** - FastAPI application tests
   - `TestApp`: 6 tests for app configuration and routes

7. **tests/test_sales_assistant_agent.py** - Sales assistant specific tests
   - `TestSalesAssistantAgent`: 6 tests for agent configuration and behavior

## Running Tests

### Run all tests:
```bash
pytest tests/ -v
```

### Run with coverage report:
```bash
pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
```

### Run specific test file:
```bash
pytest tests/test_models.py -v
```

### Run specific test class:
```bash
pytest tests/test_models.py::TestChatRequestModel -v
```

## Test Coverage

**Overall: 86% coverage**

| Module | Coverage |
|--------|----------|
| models/ | 100% |
| utils/ | 100% |
| controllers/health_controller.py | 100% |
| controllers/chat_controller.py | 92% |
| ai/agents/ | 100% |
| lib/ai/agent/AgentChat.py | 100% |
| lib/ai/agent/BaseAgent.py | 92% |
| app.py | 86% |

## Test Categories

### Unit Tests (24 tests)
- Model validation (7 tests)
- Utility functions (3 tests)
- Agent structure (2 tests)
- Sales assistant configuration (6 tests)
- Application setup (6 tests)

### Integration Tests (6 tests)
- Health endpoint
- Chat endpoint (standard)
- Chat endpoint (streaming)
- Error handling
- Request validation

## Key Features Tested

✅ Pydantic model validation  
✅ API response formatting  
✅ FastAPI endpoints (sync and streaming)  
✅ Error handling and exception cases  
✅ Agent configuration and methods  
✅ Request/response models  
✅ Health checks  
✅ OpenAPI schema generation  

## CI/CD Integration

The test suite is configured for easy CI/CD integration:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pytest tests/ -v --cov=src --cov-report=xml
```

## Configuration

**pytest.ini** includes:
- Test discovery patterns
- Coverage reporting
- Python path configuration
- Custom markers for test categorization

## Notes

- Tests use mocking to avoid external dependencies (Ollama LLM)
- Coverage HTML report available in `htmlcov/index.html`
- Tests are isolated and can run in any order
- All dependencies managed through pyproject.toml dev-dependencies
