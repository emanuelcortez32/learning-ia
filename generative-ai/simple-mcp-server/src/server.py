import asyncio

from fastmcp import FastMCP
from controllers.health_controller import register_health_controller
from servers.users_server import users_server
from lib.logger import logger

mcp = FastMCP(name="simple-mcp-server")

register_health_controller(mcp)

users_server(mcp)

if __name__ == "__main__":
    logger.info("Iniciando MCP Server... ")

    asyncio.run(
        mcp.run_http_async(
            transport="streamable-http", 
            host="0.0.0.0",
            port=8088, 
            stateless_http=True))


