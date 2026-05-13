from fastmcp import FastMCP

def register_generate_cool_greetings_prompt(mcp: FastMCP):

    @mcp.prompt
    def generate_cool_greetings(name: str) -> str:
        return f"Write a cool grettings for {name}"