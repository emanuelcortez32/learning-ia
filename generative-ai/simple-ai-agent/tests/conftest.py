import pytest
from unittest.mock import Mock, MagicMock
from langchain_ollama import ChatOllama
from langchain_core.messages.ai import AIMessage


@pytest.fixture
def mock_chat_llm():
    """Mock Chat LLM model for testing"""
    llm = MagicMock(spec=ChatOllama)
    llm.invoke = Mock(return_value=AIMessage(content="Test response"))
    llm.stream = Mock(return_value=[AIMessage(content="Test"), AIMessage(content=" response")])
    llm.with_structured_output = Mock(return_value=llm)
    return llm


@pytest.fixture
def mock_llm():
    """Mock base LLM model for testing"""
    llm = MagicMock()
    llm.invoke = Mock(return_value="Test response")
    llm.with_structured_output = Mock(return_value=llm)
    return llm


@pytest.fixture
def mock_ai_message():
    """Mock AIMessage for testing"""
    return AIMessage(content="Test AI response")
