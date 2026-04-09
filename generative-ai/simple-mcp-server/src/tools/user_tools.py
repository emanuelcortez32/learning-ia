from fastmcp import FastMCP
from lib.logger import logger
from .users.get_user import get_user

def user_tools_server(mcp: FastMCP):

    logger.info("Registrando herramientas de administracion de Usuarios")

    async def get_user_tool(user_id: str) -> dict:
        return await get_user(user_id)
    
    USER_TOOLS = [get_user_tool]

    for tool in USER_TOOLS:
        tool_name = tool.__name__.removesuffix("_tool")
        logger.debug(f"Registrando {tool_name}")
        mcp.tool(name=tool_name)(tool)