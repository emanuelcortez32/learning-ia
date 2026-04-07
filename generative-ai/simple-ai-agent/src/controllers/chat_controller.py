import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from models.ChatRequestModel import ChatRequestModel
from src.ai.agents.SalesAssistantAgent import sales_assistant
from utils.response_wrapper import api_response


router = APIRouter()

@router.post("/")
async def chat(request: ChatRequestModel):
    """
    Standard HTTP endpoint that returns the complete response.
    """
    try:
        response = sales_assistant.chat(request.query)
        return api_response(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/stream")
async def chat_stream(request: ChatRequestModel):
    """
    Streaming endpoint that returns the response in chunks as Server-Sent Events.
    """
    try:
        async def generate():
            for chunk in sales_assistant.stream(request.query):
                if hasattr(chunk, 'content'):
                    yield f"data: {json.dumps({'content': chunk.content})}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))