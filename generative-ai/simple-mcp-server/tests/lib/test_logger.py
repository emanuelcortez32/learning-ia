import logging

from src.lib.logger import CustomFormatter, setup_logger


def test_custom_formatter_includes_relevant_metadata():
    formatter = CustomFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="/tmp/service.py",
        lineno=12,
        msg="hello world",
        args=(),
        exc_info=None,
        func="run",
    )

    output = formatter.format(record)

    assert "[service.py]" in output
    assert "[run]" in output
    assert "hello world" in output


def test_setup_logger_returns_configured_logger():
    logger = setup_logger()

    assert logger.propagate is False
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0].formatter, CustomFormatter)
