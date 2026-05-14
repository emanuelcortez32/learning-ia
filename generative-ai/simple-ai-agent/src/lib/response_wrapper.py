from models import ChatResponseModel
from langchain_core.messages.ai import AIMessage

def api_response(data: AIMessage | None=None, message="Success", status=True, error=None):
    return ChatResponseModel(content=data.content, message=message, status=status, error=error)