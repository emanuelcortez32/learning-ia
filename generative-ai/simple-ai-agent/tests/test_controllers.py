import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
from langchain_core.messages.ai import AIMessage
from controllers.chat_controller import router as chat_router
from controllers.health_controller import router as health_router


@pytest.fixture
def test_app():
    app = FastAPI()
    app.include_router(health_router, prefix="/health")
    app.include_router(chat_router, prefix="/chat")
    return app


@pytest.fixture
def client(test_app):
    return TestClient(test_app)


class TestHealthController:
    def test_health_endpoint(self, client):
        response = client.get("/health/")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestChatController:
    @patch('controllers.chat_controller.sales_assistant')
    def test_chat_endpoint_success(self, mock_agent, client):
        mock_agent.chat.return_value = AIMessage(content="Hello! How can I help you?")
        
        response = client.post(
            "/chat/",
            json={"query": "Hello"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "How can I help you" in data["content"] or data["content"] == "Hello! How can I help you?"
        assert data["status"] is True
        assert data["message"] == "Success"
        mock_agent.chat.assert_called_once_with("Hello")
    
    @patch('controllers.chat_controller.sales_assistant')
    def test_chat_endpoint_error(self, mock_agent, client):
        mock_agent.chat.side_effect = Exception("LLM error")
        
        response = client.post(
            "/chat/",
            json={"query": "Hello"}
        )
        
        assert response.status_code == 500
        assert "LLM error" in response.json()["detail"]
    
    def test_chat_endpoint_invalid_request(self, client):
        response = client.post("/chat/", json={})
        assert response.status_code == 422
    
    @patch('controllers.chat_controller.sales_assistant')
    def test_chat_stream_endpoint(self, mock_agent, client):
        mock_chunks = [
            Mock(content="Hello"),
            Mock(content=" there"),
            Mock(content="!")
        ]
        mock_agent.stream.return_value = mock_chunks
        
        response = client.post(
            "/chat/stream",
            json={"query": "Hello"}
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        assert "no-cache" in response.headers["cache-control"]
        
        content = response.text
        assert "data: " in content
        assert "[DONE]" in content
    
    @patch('controllers.chat_controller.sales_assistant')
    def test_chat_stream_content_format(self, mock_agent, client):
        mock_chunks = [Mock(content="Test")]
        mock_agent.stream.return_value = mock_chunks
        
        response = client.post(
            "/chat/stream",
            json={"query": "Test"}
        )
        
        lines = response.text.strip().split('\n')
        assert any('data: {"content": "Test"}' in line for line in lines)
        assert any('data: [DONE]' in line for line in lines)
