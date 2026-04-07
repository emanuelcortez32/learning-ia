from fastapi import FastAPI
from dotenv import load_dotenv
from controllers.health_controller import router as health_router
from controllers.chat_controller import router as chat_router

load_dotenv()

app = FastAPI(
    title="AI Agent API", 
    version="1.0.0"
)

app.include_router(health_router, prefix="/health", tags=["Health"])
app.include_router(chat_router, prefix="/chat", tags=["Chat", "AI", "Agent"])

@app.get("/")
async def root():
    return {"message": "AI Agent API is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
