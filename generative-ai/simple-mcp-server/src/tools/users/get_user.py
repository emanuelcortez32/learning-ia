from lib.tool_wrapper import safe_tool

USERS = [
    { "id": 1, "Name": "Pepito" },
    { "id": 2, "Name": "Fulano "}
]

@safe_tool
async def get_user(user_id: str) -> dict:
    return next((u for u in USERS if str(u["id"]) == user_id), None)