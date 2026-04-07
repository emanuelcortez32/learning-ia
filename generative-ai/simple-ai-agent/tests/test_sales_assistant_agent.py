import pytest
from unittest.mock import patch, Mock
from langchain_core.messages.ai import AIMessage
from ai.agents.SalesAssistantAgent import sales_assistant, SYSTEM_PROMPT, LLM_MODEL


class TestSalesAssistantAgent:
    def test_agent_configuration(self):
        assert sales_assistant.system_prompt == SYSTEM_PROMPT
        assert sales_assistant.model is not None
        assert "friendly and knowledgeable clothing sales assistant" in SYSTEM_PROMPT
        assert "unrelated with clothes" in SYSTEM_PROMPT
    
    def test_llm_model_name(self):
        assert LLM_MODEL == "llama3.2:3b"
    
    def test_agent_has_chat_method(self):
        assert hasattr(sales_assistant, 'chat')
        assert callable(sales_assistant.chat)
    
    def test_agent_has_stream_method(self):
        assert hasattr(sales_assistant, 'stream')
        assert callable(sales_assistant.stream)
    
    @patch('ai.agents.SalesAssistantAgent.sales_assistant._chain')
    def test_chat_invocation(self, mock_chain):
        mock_chain.invoke.return_value = AIMessage(content="I can help you with clothing!")
        
        result = sales_assistant.chat("I need a dress")
        
        mock_chain.invoke.assert_called_once()
        assert isinstance(result, AIMessage)
    
    @patch('ai.agents.SalesAssistantAgent.sales_assistant._chain')
    def test_stream_invocation(self, mock_chain):
        mock_chunks = [
            AIMessage(content="I can"),
            AIMessage(content=" help you")
        ]
        mock_chain.stream.return_value = mock_chunks
        
        result = list(sales_assistant.stream("Show me shirts"))
        
        mock_chain.stream.assert_called_once()
        assert len(result) == 2
