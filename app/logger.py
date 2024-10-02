"""Module to provide a custom logging setup"""

import logging

# https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
END: str = "\33[0m"
BOLD: str = "\33[1m"
RED: str = "\33[31m"
GREEN: str = "\33[32m"
YELLOW: str = "\33[33m"
BLUE: str = "\33[94m"


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
        """
        Format the log record with color based on log level.

        :param record: The log record to format.

        :return: Formatted log message.
        """
        color = ColoredFormatter.LOG_LEVEL_COLOR.get(record.levelno, None)

        if color:
            record.levelname = f"{color}{record.levelname}{END}"

        return super().format(record)


def setup_logger(name: str) -> logging.Logger:
    """
    Configure and return a custom logger.

    This function creates a logger with a specified name and adds
    a console handler with a colored output format. The logger
    is configured to display messages of level DEBUG and above.

    :param name: The name of the logger to be created.

    :return: The configured logger instance.
    """
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
