# Agent Architecture Guide

This guide explains how to create, customize, and extend AI agents in the Simple AI Agent framework.

## 📐 Architecture Overview

The agent system is built with a modular, extensible architecture:

```
BaseAgent (Abstract)
    ├── AgentChat (Chat-based interactions)
    ├── AgentLlm (Completion-based interactions)
    └── Custom Agents (Your implementations)
```

### Core Components

1. **BaseAgent**: Abstract base class providing common functionality
2. **AgentChat**: Chat-based agent with conversation history support
3. **AgentLlm**: Completion-based agent for single-turn interactions
4. **SalesAssistantAgent**: Example implementation using AgentChat

## 🏗️ BaseAgent Class

The foundation of all agents. Located in `src/lib/ai/agent/BaseAgent.py`.

### Key Features

- **System Prompt Management**: Define agent personality and behavior
- **Model Integration**: Support for any LangChain-compatible LLM
- **Structured Output**: Optional schema-based output formatting
- **Chain Composition**: LangChain LCEL (LangChain Expression Language) support

### Class Definition

```python
from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.runnables import RunnableSequence
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts.base import BasePromptTemplate
from abc import ABC, abstractmethod

class BaseAgent(BaseModel, ABC):
    system_prompt: str = Field(
        description="The system message", 
        default=""
    )
    model: BaseLanguageModel = Field(
        description="LLM Model", 
        default=None
    )
    structured_output: dict | type | None = Field(
        description="Model wrapper that returns outputs formatted to match the given schema.", 
        default=None
    )

    _llm: BaseLanguageModel = PrivateAttr()
    _prompt: BasePromptTemplate = PrivateAttr()
    _chain: RunnableSequence = PrivateAttr()

    @abstractmethod
    def _create_prompt(self) -> BasePromptTemplate:
        """Each subclass must define its own prompt template"""
        pass
```

## 💬 AgentChat Class

Chat-based agent for conversational interactions. Located in `src/lib/ai/agent/AgentChat.py`.

### Features

- **Chat History**: Maintains conversation context
- **System Messages**: Persistent agent personality
- **Streaming Support**: Real-time response generation
- **Simple Interface**: Easy-to-use chat() and stream() methods

### Implementation

```python
from .BaseAgent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage 
from langchain_core.language_models.chat_models import BaseChatModel

class AgentChat(BaseAgent):
    model: BaseChatModel

    def _create_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            ("human", "{query}")
        ])
    
    def chat(self, query: str):
        """Synchronous chat - returns complete response"""
        return self._chain.invoke({"query": query})
    
    def stream(self, query: str):
        """Streaming chat - yields response chunks"""
        return self._chain.stream({"query": query})
```

### Usage Example

```python
from lib.ai.agent.AgentChat import AgentChat
from langchain_ollama import ChatOllama

# Define system prompt
SYSTEM_PROMPT = """
You are a helpful assistant specialized in Python programming.
Provide clear, concise answers with code examples when appropriate.
"""

# Initialize model
model = ChatOllama(
    model="llama3.2:3b",
    base_url="http://localhost:11434",
    temperature=0.7
)

# Create agent
assistant = AgentChat(
    system_prompt=SYSTEM_PROMPT, 
    model=model
)

# Standard chat
response = assistant.chat("How do I sort a list in Python?")
print(response.content)

# Streaming chat
for chunk in assistant.stream("Explain list comprehensions"):
    print(chunk.content, end='', flush=True)
```

## 🎯 Creating Custom Agents

### Step 1: Define Your Agent Class

Create a new file in `src/agents/`:

```python
# src/agents/CustomerSupportAgent.py
from lib.ai.agent.AgentChat import AgentChat
from langchain_ollama import ChatOllama

SYSTEM_PROMPT = """
You are a customer support specialist for an e-commerce platform.
You help customers with:
- Order tracking and status
- Returns and refunds
- Product questions
- Account issues

Be empathetic, professional, and solution-oriented.
Always verify customer information before providing sensitive details.
"""

LLM_MODEL = "llama3.2:3b"

model = ChatOllama(
    model=LLM_MODEL,
    base_url="http://localhost:11434",
    temperature=0.3
)

customer_support = AgentChat(
    system_prompt=SYSTEM_PROMPT, 
    model=model
)
```

### Step 2: Integrate with FastAPI

Update `src/app.py`:

```python
from agents.CustomerSupportAgent import customer_support

@app.post("/support/chat")
async def support_chat(request: ChatRequestModel):
    try:
        response = customer_support.chat(request.query)
        return ChatResponseModel(response=response.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/support/stream")
async def support_stream(request: ChatRequestModel):
    async def generate():
        for chunk in customer_support.stream(request.query):
            if hasattr(chunk, 'content'):
                yield f"data: {json.dumps({'content': chunk.content})}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

## 🔧 Advanced Agent Configurations

### 1. Multi-Turn Conversation Agent

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

class ConversationAgent(BaseAgent):
    def _create_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{query}")
        ])
    
    def chat_with_history(self, query: str, history: list):
        return self._chain.invoke({
            "query": query,
            "history": history
        })
```

### 2. Structured Output Agent

```python
from pydantic import BaseModel, Field
from lib.ai.agent.AgentChat import AgentChat

class ProductRecommendation(BaseModel):
    product_name: str = Field(description="Name of the product")
    price: float = Field(description="Price in USD")
    reason: str = Field(description="Why this product is recommended")
    confidence: float = Field(description="Confidence score 0-1")

SYSTEM_PROMPT = """
Analyze customer queries and recommend products.
Always provide structured recommendations with confidence scores.
"""

structured_agent = AgentChat(
    system_prompt=SYSTEM_PROMPT,
    model=model,
    structured_output=ProductRecommendation
)

# Usage
result = structured_agent.chat("I need affordable running shoes")
# result is a ProductRecommendation object
print(f"Product: {result.product_name}")
print(f"Price: ${result.price}")
print(f"Reason: {result.reason}")
```

### 3. Tool-Using Agent (RAG)

```python
from langchain.agents import create_tool_calling_agent
from langchain.tools import tool

@tool
def search_products(query: str) -> str:
    """Search product database"""
    # Your search logic
    return "Search results..."

@tool
def check_inventory(product_id: str) -> str:
    """Check product inventory"""
    # Your inventory logic
    return "Stock: 50 units"

tools = [search_products, check_inventory]

# Create tool-calling agent
from langchain.agents import AgentExecutor

agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

result = agent_executor.invoke({
    "input": "Do you have product ABC123 in stock?"
})
```

### 4. Temperature and Parameter Tuning

```python
# Creative writing agent (high temperature)
creative_agent = AgentChat(
    system_prompt="You are a creative storyteller...",
    model=ChatOllama(
        model="llama3.2:3b",
        temperature=0.9,  # More creative
        top_p=0.95,
        top_k=50
    )
)

# Factual assistant (low temperature)
factual_agent = AgentChat(
    system_prompt="You provide accurate, factual information...",
    model=ChatOllama(
        model="llama3.2:3b",
        temperature=0.1,  # More deterministic
        top_p=0.9
    )
)
```

## 📊 Agent Prompt Engineering

### Best Practices

1. **Be Specific**: Clearly define agent's role and capabilities
2. **Set Boundaries**: Explicitly state what the agent should NOT do
3. **Provide Context**: Include relevant background information
4. **Use Examples**: Few-shot learning with example interactions
5. **Define Output Format**: Specify how responses should be structured

### Example: Well-Structured Prompt

```python
SYSTEM_PROMPT = """
# Role
You are a technical support specialist for SoftwareX product.

# Capabilities
- Troubleshoot installation issues
- Explain features and configuration
- Provide code examples for API integration
- Guide users through common workflows

# Constraints
- Only answer questions about SoftwareX
- Do not provide information about competitor products
- Do not share internal company information
- Redirect billing questions to finance team

# Response Format
1. Acknowledge the user's question
2. Provide clear, step-by-step solution
3. Offer additional resources if available
4. Ask if they need further clarification

# Tone
Professional, patient, and encouraging. Use technical terms when appropriate
but explain them clearly. Always prioritize user understanding.
"""
```

## 🧪 Testing Agents

### Unit Testing

```python
# tests/test_agents.py
import pytest
from agents.SalesAssistantAgent import sales_assistant

def test_sales_assistant_response():
    response = sales_assistant.chat("Hello")
    assert response.content is not None
    assert len(response.content) > 0

def test_sales_assistant_clothing_context():
    response = sales_assistant.chat("I need a jacket")
    assert "jacket" in response.content.lower() or "clothing" in response.content.lower()

def test_sales_assistant_off_topic():
    response = sales_assistant.chat("What is quantum physics?")
    assert "can't answer" in response.content.lower() or "unrelated" in response.content.lower()
```

### Integration Testing

```python
import requests

def test_chat_endpoint():
    response = requests.post(
        "http://localhost:8000/chat",
        json={"query": "Show me shirts"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert len(data["response"]) > 0

def test_streaming_endpoint():
    response = requests.post(
        "http://localhost:8000/chat/stream",
        json={"query": "Hello"},
        stream=True
    )
    assert response.status_code == 200
    
    chunks = []
    for line in response.iter_lines():
        if line:
            chunks.append(line)
    
    assert len(chunks) > 0
```

## 🚀 Performance Optimization

### 1. Model Selection

```python
# Faster inference
model = ChatOllama(model="llama3.2:1b")  # Smaller, faster

# Better quality
model = ChatOllama(model="llama3.2:3b")  # Larger, more accurate

# Best of both
model = ChatOllama(model="llama3.2:3b", num_gpu=1)  # GPU acceleration
```

### 2. Caching

```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

set_llm_cache(InMemoryCache())
```

### 3. Batch Processing

```python
queries = ["Query 1", "Query 2", "Query 3"]
responses = [agent.chat(q) for q in queries]
```

### 4. Async Operations

```python
import asyncio

async def async_chat(agent, query: str):
    return await agent._chain.ainvoke({"query": query})

async def process_multiple():
    queries = ["Q1", "Q2", "Q3"]
    tasks = [async_chat(agent, q) for q in queries]
    results = await asyncio.gather(*tasks)
    return results
```

## 📚 Real-World Examples

### Example 1: Technical Documentation Assistant

```python
# src/agents/DocsAssistant.py
SYSTEM_PROMPT = """
You are a technical documentation assistant for API developers.
You help users understand API endpoints, authentication, and integration patterns.

When users ask questions:
1. Reference specific API endpoints and methods
2. Provide code examples in Python, JavaScript, or cURL
3. Explain authentication and authorization requirements
4. Link to relevant documentation sections
"""

docs_assistant = AgentChat(
    system_prompt=SYSTEM_PROMPT,
    model=ChatOllama(model="llama3.2:3b", temperature=0.2)
)
```

### Example 2: Language Tutor

```python
# src/agents/LanguageTutor.py
SYSTEM_PROMPT = """
You are a Spanish language tutor helping English speakers learn Spanish.

Teaching approach:
- Start with the user's skill level
- Provide translations and explanations
- Correct grammar gently and constructively
- Use examples in context
- Encourage practice with conversation prompts

Format corrections as:
Original: [user's text]
Corrected: [corrected version]
Explanation: [why the correction was needed]
"""

language_tutor = AgentChat(
    system_prompt=SYSTEM_PROMPT,
    model=ChatOllama(model="llama3.2:3b", temperature=0.4)
)
```

### Example 3: Code Review Assistant

```python
# src/agents/CodeReviewer.py
SYSTEM_PROMPT = """
You are a code review assistant specializing in Python.

Review criteria:
- Code quality and readability
- PEP 8 compliance
- Performance considerations
- Security vulnerabilities
- Best practices

Provide feedback as:
✅ Strengths: [what's good]
⚠️  Issues: [what needs improvement]
💡 Suggestions: [specific improvements]
"""

code_reviewer = AgentChat(
    system_prompt=SYSTEM_PROMPT,
    model=ChatOllama(model="llama3.2:3b", temperature=0.3)
)
```

## 🔍 Debugging Agents

### Enable Verbose Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# In your agent
response = agent.chat(query)
logger.debug(f"Query: {query}")
logger.debug(f"Response: {response.content}")
```

### Trace Chain Execution

```python
from langchain.callbacks import StdOutCallbackHandler

agent._chain.invoke(
    {"query": "test"},
    config={"callbacks": [StdOutCallbackHandler()]}
)
```

## 📖 Additional Resources

- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [Ollama Model Library](https://ollama.ai/library)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/)

## 🤝 Contributing

To contribute new agent types or improvements:

1. Create your agent in `src/agents/`
2. Add comprehensive docstrings
3. Include usage examples
4. Add unit tests
5. Update this documentation
6. Submit a pull request

## 📧 Questions?

For questions about agent development, open an issue or discussion on GitHub.
