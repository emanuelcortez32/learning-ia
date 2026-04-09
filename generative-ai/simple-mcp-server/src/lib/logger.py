import logging
import os
import sys

from logging import Formatter, StreamHandler
from datetime import datetime
from termcolor import colored

class CustomFormatter(Formatter):
    COLORS = {
        'DEBUG': 'blue',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'magenta'
    }

    def format(self, record):
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        filename = os.path.basename(record.pathname)
        function_name = record.funcName
        level = colored(record.levelname, self.COLORS.get(record.levelname, 'white'))
        log_message = f"[{timestamp}][{level}][{filename}][{function_name}] {record.getMessage()}"

        return log_message

    
def setup_logger():
    logger = logging.getLogger(__name__)
    logger.propagate = False
    log_level = "DEBUG"
    logger.setLevel(log_level)

    console_handler = StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    formatter = CustomFormatter()
    console_handler.setFormatter(formatter)

    logger.handlers = []

    logger.addHandler(console_handler)

    logging.getLogger("sqlalchemy.engine.Engine").setLevel(logging.WARNING)

    return logger


logger = setup_logger()