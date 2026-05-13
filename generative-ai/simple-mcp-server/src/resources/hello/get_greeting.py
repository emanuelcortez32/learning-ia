from fastmcp import FastMCP

def register_get_greeting_resource(mcp: FastMCP):

    @mcp.resource("resource://greeting")
    def get_greeting() -> str:
        """Provides a simple greeting message."""
        return "Hello from FastMCP Resources!"