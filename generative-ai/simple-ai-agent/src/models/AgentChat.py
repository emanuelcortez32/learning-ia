import json
import logging

from pydantic import BaseModel, Field, PrivateAttr
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import BaseTool
from typing import Sequence
from langchain.messages import HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel


logger = logging.getLogger("uvicorn")

class AgentChat(BaseModel):
    system_prompt: str = Field(description="The system message", default="")
    model: BaseChatModel = Field(description="LLM Model", default=None)
    structured_output: dict | type | None = Field(description="Model wrapper that returns outputs formatted to match the given schema.", default=None)
    tools: Sequence[BaseTool] = Field(description="Tools", default=[])

    _agent = PrivateAttr()


    def __init__(self, **data):
        super().__init__(**data)
        self._setup_agent()

    def _setup_agent(self):

        response_format = None
        if self.structured_output is not None:
            response_format = ToolStrategy(self.structured_output)
        
        self._agent = create_agent(
            model=self.model, 
            tools=self.tools, 
            system_prompt=self.system_prompt,
            response_format=response_format)

    def chat(self, query: str):
        result = self._agent.invoke(
            {"messages": [HumanMessage(query)]}
        )

        last_ai_message = next(
            (msg for msg in reversed(result["messages"]) if isinstance(msg, AIMessage)),
            None
        )

        if last_ai_message:
            return last_ai_message
    
    def stream(self, query: str):
        
        async def generate():

            for chunk in self._agent.stream(
                { "messages": [HumanMessage(query)]}, 
                stream_mode="values"
            ):
    
                latest_message = chunk["messages"][-1]

                if latest_message.content:
                    if isinstance(latest_message, HumanMessage):
                        logger.info(f"User: {latest_message.content}")

                    elif isinstance(latest_message, AIMessage):
                        logger.info(f"Agent: {latest_message.content}")
                        yield f"data: {json.dumps({
                            'type': 'content',
                            'content': latest_message.content
                        })}\n\n"

                elif latest_message.tool_calls:
                    tool_names = [tc["name"] for tc in latest_message.tool_calls]

                    logger.info(f"Calling tools: {tool_names}")
                    yield f"data: {json.dumps({
                        'type': 'tool_call',
                        'tools': tool_names
                    })}\n\n"

            yield "data: [DONE]\n\n"

        return generate()