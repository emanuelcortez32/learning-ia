import pytest

from src.controllers.health_controller import register_health_controller


class FakeMCP:
    def __init__(self):
        self.route_path = None
        self.route_methods = None
        self.handler = None

    def custom_route(self, path, methods):
        self.route_path = path
        self.route_methods = methods

        def decorator(func):
            self.handler = func
            return func

        return decorator


@pytest.mark.asyncio
async def test_register_health_controller_registers_ok_endpoint():
    mcp = FakeMCP()

    register_health_controller(mcp)

    assert mcp.route_path == "/health"
    assert mcp.route_methods == ["GET"]

    response = await mcp.handler(None)
    assert response.status_code == 200
    assert response.body == b"OK"
