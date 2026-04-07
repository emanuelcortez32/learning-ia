from .BaseAgent import BaseAgent
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM

class AgentLlm(BaseAgent):
    model: LLM

    def _create_prompt(self) -> PromptTemplate:
        return PromptTemplate.from_template(
            f"{self.system_prompt}\n\nHuman: {{query}}\nAssistant:"
        )
    
    def generate(self, query: str):
        return self._chain.invoke({"query": query})