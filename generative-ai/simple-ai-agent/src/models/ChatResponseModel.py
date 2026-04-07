from pydantic import BaseModel, Field

class ChatResponseModel(BaseModel):
    content: str = Field(default=None)
    status: bool = Field(default=True)
    message: str = Field(default="Success")
    error: None = Field(default=None)