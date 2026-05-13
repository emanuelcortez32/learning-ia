import importlib.util
import runpy
import sys
import types
from pathlib import Path


MAIN_PATH = Path(__file__).resolve().parents[1] / "src" / "main.py"


def _build_fake_modules():
    calls = {"health": 0, "hello": 0}

    class FakeMCP:
        def __init__(self, name):
            self.name = name
            self.run_http_calls = []

        def run_http_async(self, **kwargs):
            self.run_http_calls.append(kwargs)
            return "RUN_HTTP_ASYNC_RETURN"

    fake_fastmcp = types.ModuleType("fastmcp")
    fake_fastmcp.FastMCP = FakeMCP

    fake_lib = types.ModuleType("lib")
    fake_lib.logger = types.SimpleNamespace(info=lambda *_: None)

    fake_config = types.ModuleType("config")
    fake_config.config = types.SimpleNamespace(host="127.0.0.1", port=8080)

    fake_tools_hello = types.ModuleType("tools.hello")
    fake_tools_hello.register_hello_tools = lambda mcp: calls.__setitem__("hello", calls["hello"] + 1)

    fake_health_controller = types.ModuleType("controllers.health_controller")
    fake_health_controller.register_health_controller = (
        lambda mcp: calls.__setitem__("health", calls["health"] + 1)
    )

    return calls, FakeMCP, fake_fastmcp, fake_lib, fake_config, fake_tools_hello, fake_health_controller


def test_main_initialization_registers_routes_and_tools(monkeypatch):
    calls, FakeMCP, fake_fastmcp, fake_lib, fake_config, fake_tools_hello, fake_health_controller = (
        _build_fake_modules()
    )

    monkeypatch.setitem(sys.modules, "fastmcp", fake_fastmcp)
    monkeypatch.setitem(sys.modules, "lib", fake_lib)
    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "tools.hello", fake_tools_hello)
    monkeypatch.setitem(sys.modules, "controllers.health_controller", fake_health_controller)

    spec = importlib.util.spec_from_file_location("main_under_test", MAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert isinstance(module.mcp, FakeMCP)
    assert module.mcp.name == "simple-mcp-server"
    assert calls == {"health": 1, "hello": 1}


def test_main_executes_http_server_when_run_as_script(monkeypatch):
    calls, _, fake_fastmcp, fake_lib, fake_config, fake_tools_hello, fake_health_controller = _build_fake_modules()
    asyncio_calls = {}
    fake_asyncio = types.ModuleType("asyncio")
    fake_asyncio.run = lambda arg: asyncio_calls.setdefault("arg", arg)

    monkeypatch.setitem(sys.modules, "fastmcp", fake_fastmcp)
    monkeypatch.setitem(sys.modules, "lib", fake_lib)
    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "tools.hello", fake_tools_hello)
    monkeypatch.setitem(sys.modules, "controllers.health_controller", fake_health_controller)
    monkeypatch.setitem(sys.modules, "asyncio", fake_asyncio)

    runpy.run_path(str(MAIN_PATH), run_name="__main__")

    assert calls == {"health": 1, "hello": 1}
    assert asyncio_calls["arg"] == "RUN_HTTP_ASYNC_RETURN"
