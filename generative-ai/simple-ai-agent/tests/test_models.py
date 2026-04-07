import pytest
from pydantic import ValidationError
from models.ChatRequestModel import ChatRequestModel
from models.ChatResponseModel import ChatResponseModel


class TestChatRequestModel:
    def test_valid_request(self):
        request = ChatRequestModel(query="Hello, AI!")
        assert request.query == "Hello, AI!"
    
    def test_missing_query(self):
        with pytest.raises(ValidationError):
            ChatRequestModel()
    
    def test_empty_query(self):
        request = ChatRequestModel(query="")
        assert request.query == ""
    
    def test_query_type(self):
        with pytest.raises(ValidationError):
            ChatRequestModel(query=123)


class TestChatResponseModel:
    def test_default_values(self):
        response = ChatResponseModel()
        assert response.content is None
        assert response.status is True
        assert response.message == "Success"
        assert response.error is None
    
    def test_custom_values(self):
        response = ChatResponseModel(
            content="Test content",
            status=False,
            message="Error occurred"
        )
        assert response.content == "Test content"
        assert response.status is False
        assert response.message == "Error occurred"
        assert response.error is None
    
    def test_partial_values(self):
        response = ChatResponseModel(content="Only content")
        assert response.content == "Only content"
        assert response.status is True
        assert response.message == "Success"
        assert response.error is None
