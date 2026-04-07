from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from model.ChatRequestModel import ChatRequestModel
from model.ChatResponseModel import ChatResponseModel
from agents.SalesAssistantAgent import sales_assistant
import json

load_dotenv()

app = FastAPI(title="AI Agent API", version="1.0.0")


@app.get("/")
async def root():
    return {"message": "AI Agent API is running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponseModel)
async def chat(request: ChatRequestModel):
    """
    Standard HTTP endpoint that returns the complete response.
    """
    try:
        response = sales_assistant.chat(request.query)
        return ChatResponseModel(response=response.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
