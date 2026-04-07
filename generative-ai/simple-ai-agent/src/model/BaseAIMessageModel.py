from pydantic import BaseModel, Field

class BaseAIMessageModel(BaseModel):
    message: str = Field(description="The message from AI")