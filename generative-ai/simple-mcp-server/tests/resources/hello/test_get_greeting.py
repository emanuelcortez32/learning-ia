from src.resources.hello.get_greeting import register_get_greeting_resource


class FakeMCP:
    def __init__(self):
        self.uri = None
        self.handler = None

    def resource(self, uri):
        self.uri = uri

        def decorator(func):
            self.handler = func
            return func

        return decorator


def test_register_get_greeting_resource_registers_resource_handler():
    mcp = FakeMCP()

    register_get_greeting_resource(mcp)

    assert mcp.uri == "resource://greeting"
    assert mcp.handler is not None
    assert mcp.handler() == "Hello from FastMCP Resources!"
