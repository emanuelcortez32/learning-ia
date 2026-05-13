import src.tools.hello.register as register_module


def test_register_hello_tools_uses_loader(monkeypatch):
    captured = {}

    def fake_load_tools(tools, mcp):
        captured["tools"] = tools
        captured["mcp"] = mcp

    monkeypatch.setattr(register_module, "load_tools", fake_load_tools)
    mcp = object()

    register_module.register_hello_tools(mcp)

    assert captured["mcp"] is mcp
    assert captured["tools"] == [register_module.say_hello]
