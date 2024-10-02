"""Module to provide a custom logging setup"""

import logging

# https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
END = "\33[0m"
BOLD = "\33[1m"
RED = "\33[31m"
GREEN = "\33[32m"
YELLOW = "\33[33m"
BLUE = "\33[94m"


class ColoredFormatter(logging.Formatter):
    """Formats log messages with color based on severity level."""

    LOG_LEVEL_COLOR = {
        logging.CRITICAL: RED,
        logging.FATAL: RED,
        logging.ERROR: RED,
        logging.WARNING: YELLOW,
        logging.INFO: GREEN,
        logging.DEBUG: BLUE,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = ColoredFormatter.LOG_LEVEL_COLOR.get(record.levelno, None)

        if color:
            record.levelname = f"{color}{record.levelname}{END}"

        return super().format(record)


def setup_logger(name: str) -> logging.Logger:
    """Configure and return logger"""
    # Create handle
    console_handler = logging.StreamHandler()

    # Set message format
    formatter = ColoredFormatter("%(levelname)s:%(name)s: %(message)s")

    # Apply formatter to handle
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)

    return logger
