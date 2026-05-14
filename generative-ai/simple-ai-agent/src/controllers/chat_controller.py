from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from models.ChatRequestModel import ChatRequestModel
from lib import api_response
from agents import wikipedia_agent

router = APIRouter()

@router.post("/")
async def chat(request: ChatRequestModel):
    """
    Standard HTTP endpoint that returns the complete response.
    """
    try:
        response = wikipedia_agent.chat(request.query)
        return api_response(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/stream")
async def chat_stream(request: ChatRequestModel):
    """
    Streaming endpoint that returns the response in chunks as Server-Sent Events.
    """
    try:
        return StreamingResponse(
            wikipedia_agent.stream(request.query),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))