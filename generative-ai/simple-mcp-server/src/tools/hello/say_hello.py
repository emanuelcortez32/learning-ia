from lib import safe_tool

@safe_tool
async def say_hello(name: str) -> dict:
    return f"Hello {name}!!"