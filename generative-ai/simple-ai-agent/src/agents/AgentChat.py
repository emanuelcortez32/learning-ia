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
        return self._chain.invoke({"query": query})
    
    def stream(self, query: str):
        return self._chain.stream({"query": query})