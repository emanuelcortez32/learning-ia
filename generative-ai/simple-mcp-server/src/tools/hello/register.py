from fastmcp import FastMCP
from lib import logger
from utils import load_tools
from .say_hello import say_hello

def register_hello_tools(mcp: FastMCP):
    logger.info("Registrando herramientas de Hello")

    HELLO_TOOLS = [
        say_hello
    ]

    load_tools(HELLO_TOOLS, mcp)

