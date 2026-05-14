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

    def chat(self, query: str) -> AIMessage:
        result = self._agent.invoke(
            {"messages": [HumanMessage(query)]}
        )

        last_ai_message = next(
            (msg for msg in reversed(result["messages"]) if isinstance(msg, AIMessage)),
            None
        )

        if not last_ai_message:
            raise ValueError("No AI response generated")

        return last_ai_message
    
    def stream(self, query: str):
        
        async def generate():
            full_response = ""
            async for chunk in self._agent.astream(
                { "messages": [HumanMessage(query)]}, 
                stream_mode="messages"
            ):
                msg, _ = chunk

                if msg.content:
                    if isinstance(msg, HumanMessage):
                        logger.info(f"User: {msg.content}")

                    elif isinstance(msg, AIMessage):
                        full_response += msg.content
                        yield f"data: {json.dumps({
                            'type': 'content',
                            'content': msg.content
                        })}\n\n"

                elif msg.tool_calls:
                    tool_names = [tc["name"] for tc in msg.tool_calls]

                    logger.info(f"Calling tools: {tool_names}")
                    yield f"data: {json.dumps({
                        'type': 'tool_call',
                        'tools': tool_names
                    })}\n\n"

            logger.info(f"Agent: {full_response}")

            yield "data: [DONE]\n\n"

        return generate()