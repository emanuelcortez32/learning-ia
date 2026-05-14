from pydantic import BaseModel, Field
from typing import List
from langchain_core.messages.content import InvalidToolCall
from langchain_core.messages.tool import ToolCall


class Metadata(BaseModel):
    id: str | None = Field(default=None)
    tools_call: List[ToolCall] | None = Field(default=None)
    invalid_tool_calls: List[InvalidToolCall] | None = Field(default=None)
class ChatResponseModel(BaseModel):
    content: str = Field(default=None)
    status: bool = Field(default=True)
    message: str = Field(default="Success")
    error: None = Field(default=None)
    meta: Metadata | None = Field(default=None)
    