import pytest
from unittest.mock import patch
from langchain_core.messages.ai import AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


class TestAgentChatStructure:
    def test_create_prompt_structure(self):
        """Test ChatPromptTemplate creation"""
        from langchain_core.messages import SystemMessage
        
        test_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="Test system message"),
            ("human", "{query}")
        ])
        
        assert len(test_prompt.messages) == 2
        assert test_prompt.messages[0].content == "Test system message"
        assert isinstance(test_prompt, ChatPromptTemplate)


class TestAgentLlmStructure:
    def test_create_prompt_structure(self):
        """Test PromptTemplate creation"""
        test_prompt = PromptTemplate.from_template(
            "Test system message\n\nHuman: {query}\nAssistant:"
        )
        
        assert isinstance(test_prompt, PromptTemplate)
        assert "Test system message" in test_prompt.template
        assert "{query}" in test_prompt.template
