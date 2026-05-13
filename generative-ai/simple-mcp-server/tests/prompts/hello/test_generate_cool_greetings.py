from src.prompts.hello.generate_cool_greetings import register_generate_cool_greetings_prompt


class FakeMCP:
    def __init__(self):
        self.handler = None

    @property
    def prompt(self):
        def decorator(func):
            self.handler = func
            return func

        return decorator


def test_register_generate_cool_greetings_prompt_registers_prompt_handler():
    mcp = FakeMCP()

    register_generate_cool_greetings_prompt(mcp)

    assert mcp.handler is not None
    assert mcp.handler("Emanuel") == "Write a cool grettings for Emanuel"
