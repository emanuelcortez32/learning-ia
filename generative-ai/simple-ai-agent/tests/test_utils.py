import pytest
from langchain_core.messages.ai import AIMessage
from utils.response_wrapper import api_response
from models.ChatResponseModel import ChatResponseModel


class TestResponseWrapper:
    def test_api_response_default_values(self, mock_ai_message):
        result = api_response(mock_ai_message)
        
        assert isinstance(result, ChatResponseModel)
        assert result.content == "Test AI response"
        assert result.message == "Success"
        assert result.status is True
        assert result.error is None
    
    def test_api_response_custom_message(self, mock_ai_message):
        result = api_response(mock_ai_message, message="Custom message")
        
        assert result.content == "Test AI response"
        assert result.message == "Custom message"
        assert result.status is True
        assert result.error is None
    
    def test_api_response_error_state(self, mock_ai_message):
        result = api_response(
            mock_ai_message,
            message="Error occurred",
            status=False
        )
        
        assert result.content == "Test AI response"
        assert result.message == "Error occurred"
        assert result.status is False
        assert result.error is None
