from models import ChatResponseModel, Metadata
from langchain_core.messages.ai import AIMessage

def transform_ai_message_to_api_response(data: AIMessage | None = None, message="Success", status=True, error=None):
    
    response = ChatResponseModel(
        content=data.content, 
        message=message, 
        status=status, 
        error=error,
        meta=Metadata(id=data.id, tools_call=data.tool_calls, invalid_tool_calls=data.invalid_tool_calls))
    
    return response.model_dump_json()