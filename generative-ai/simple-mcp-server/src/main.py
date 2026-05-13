import asyncio

from fastmcp import FastMCP
from lib import logger
from config import config
from tools.hello import register_hello_tools
from prompts.hello import register_generate_cool_greetings_prompt
from resources.hello import register_get_greeting_resource
from controllers.health_controller import register_health_controller

mcp = FastMCP(
    name="simple-mcp-server"
)


# Custom Routes / Controllers
register_health_controller(mcp)

# Prompts
register_generate_cool_greetings_prompt(mcp)

# Resources
register_get_greeting_resource(mcp)

# Tools
register_hello_tools(mcp)

if __name__ == "__main__":
    logger.info("Iniciando MCP Server... ")

    asyncio.run(
        mcp.run_http_async(
            transport="streamable-http", 
            host=config.host,
            port=config.port, 
            stateless_http=True))


