from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.runnables import RunnableSequence
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts.base import BasePromptTemplate
from abc import ABC, abstractmethod

class BaseAgent(BaseModel, ABC):
    system_prompt: str = Field(description="The system message", default="")
    model: BaseLanguageModel = Field(description="LLM Model", default=None)
    structured_output: dict | type | None = Field(description="Model wrapper that returns outputs formatted to match the given schema.", default=None)

    _llm: BaseLanguageModel = PrivateAttr()
    _prompt: BasePromptTemplate = PrivateAttr()
    _chain: RunnableSequence = PrivateAttr()


    def __init__(self, **data):
        super().__init__(**data)
        self._setup_agent()

    def _setup_agent(self):
        self._llm = self.model
        if self.structured_output is not None:
            self._llm = self._llm.with_structured_output(self.structured_output)
        
        self._prompt = self._create_prompt()
        self._chain = self._prompt | self._llm

    @abstractmethod
    def _create_prompt(self) -> BasePromptTemplate:
        """
        Cada subclase debe definir su propio prompt (chat o completion)
        """
        pass


