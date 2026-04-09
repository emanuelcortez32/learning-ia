from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse

def register_health_controller(mcp: FastMCP):
    
    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> PlainTextResponse:
        """Endpoint de health check para verificar que el servidor esta funcionando"""
        return PlainTextResponse("OK")