from fastapi import FastAPI
from controllers.health_controller import router as health_router
from controllers.chat_controller import router as chat_router
from config.config import config


app = FastAPI(
    title=config.app_name, 
    version=config.app_version
)

app.include_router(health_router, prefix="/health", tags=["Health"])
app.include_router(chat_router, prefix="/chat", tags=["Chat", "AI", "Agent"])

@app.get("/")
async def root():
    return {"message": "AI Agent API is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port)
