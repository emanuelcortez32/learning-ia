from src.utils.tools_loader import load_tools


class FakeMCP:
    def __init__(self):
        self.registered = []

    def tool(self, name):
        def decorator(func):
            self.registered.append((name, func))
            return func

        return decorator


async def first_tool():
    return "first"


async def second():
    return "second"


def test_load_tools_registers_each_tool_with_expected_name():
    mcp = FakeMCP()

    load_tools([first_tool, second], mcp)

    assert [name for name, _ in mcp.registered] == ["first", "second"]
    assert [tool for _, tool in mcp.registered] == [first_tool, second]
