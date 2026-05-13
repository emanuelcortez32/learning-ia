from fastmcp import FastMCP
from lib import logger

def load_tools(tools: list, mcp: FastMCP):
    for tool in tools:
        tool_name = tool.__name__.removesuffix("_tool")
        logger.debug(f"Registrando {tool_name}")
        mcp.tool(name=tool_name)(tool)