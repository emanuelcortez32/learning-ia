import pytest

from src.lib.tool_wrapper import safe_tool


@pytest.mark.asyncio
async def test_safe_tool_wraps_non_dict_result():
    @safe_tool
    async def greet(name: str):
        return f"Hello {name}"

    result = await greet("Emanuel")

    assert result == {"success": True, "data": "Hello Emanuel"}


@pytest.mark.asyncio
async def test_safe_tool_preserves_success_shaped_dict():
    expected = {"success": True, "data": {"message": "ok"}}

    @safe_tool
    async def already_wrapped():
        return expected

    result = await already_wrapped()

    assert result is expected


@pytest.mark.asyncio
async def test_safe_tool_wraps_plain_dict_as_data():
    payload = {"message": "ok"}

    @safe_tool
    async def plain_dict():
        return payload

    result = await plain_dict()

    assert result == {"success": True, "data": payload}


@pytest.mark.asyncio
async def test_safe_tool_returns_error_shape_on_exception():
    @safe_tool
    async def broken_tool():
        raise ValueError("invalid input")

    result = await broken_tool()

    assert result["success"] is False
    assert result["error"]["code"] == "INTERNAL_ERROR"
    assert result["error"]["message"] == "Error inesperado en broken_tool"
    assert result["error"]["details"] == "invalid input"
