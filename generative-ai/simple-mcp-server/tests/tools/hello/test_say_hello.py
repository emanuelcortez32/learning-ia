import pytest

from src.tools.hello.say_hello import say_hello


@pytest.mark.asyncio
async def test_say_hello_returns_success_payload():
    result = await say_hello("Emanuel")

    assert result == {"success": True, "data": "Hello Emanuel!!"}
