"""
Logging utility for the Magi project.

This module provides a centralized logging configuration with the ability to:
- Log to both console and file
- Configure log levels globally
- Enable/disable logging globally
- Format logs consistently across the application
"""

import logging
import os
import sys
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, cast

# Type variables for decorator typing
F = TypeVar("F", bound=Callable[..., Any])

# Default log directory
LOG_DIR = Path("logs")

# Global flag to enable/disable logging
LOGGING_ENABLED = True

# Configure the root logger
root_logger = logging.getLogger()

# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Create a file handler that logs all messages
os.makedirs(LOG_DIR, exist_ok=True)
log_file = LOG_DIR / f"magi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Create a console handler with a higher log level
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add the handlers to the root logger
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Set the default log level
root_logger.setLevel(logging.INFO)

# Configure specific third-party loggers to reduce noise
# Set higher log levels for known verbose libraries
THIRD_PARTY_LOGGERS = {
    "py4j": logging.WARNING,
    "py4j.clientserver": logging.WARNING,
    "py4j.java_gateway": logging.WARNING,
    "urllib3": logging.WARNING,
    "matplotlib": logging.WARNING,
    "asyncio": logging.INFO,
    "pyspark": logging.INFO,
}

# Apply configuration to third-party loggers
for logger_name, level in THIRD_PARTY_LOGGERS.items():
    logging.getLogger(logger_name).setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Name for the logger, typically the module name

    Returns:
        A configured logger instance
    """
    logger = logging.getLogger(name)

    # Ensure our application loggers inherit from root but can be individually configured
    if name.startswith("magi"):
        # Don't override if already explicitly set
        if not hasattr(logger, "_level_set"):
            logger.setLevel(root_logger.level)
            logger._level_set = True

    return logger


def set_global_log_level(level: int) -> None:
    """
    Set the global log level for all loggers.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
    """
    root_logger.setLevel(level)

    # Update handlers
    for handler in root_logger.handlers:
        handler.setLevel(level)

    # Reset third-party loggers to their specified levels
    for logger_name, logger_level in THIRD_PARTY_LOGGERS.items():
        logging.getLogger(logger_name).setLevel(logger_level)


def set_logger_level(name: str, level: int) -> None:
    """
    Set the log level for a specific logger.

    Args:
        name: Logger name
        level: Logging level
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger._level_set = True


def enable_logging() -> None:
    """Enable logging globally."""
    global LOGGING_ENABLED
    LOGGING_ENABLED = True
    root_logger.disabled = False


def disable_logging() -> None:
    """Disable logging globally."""
    global LOGGING_ENABLED
    LOGGING_ENABLED = False
    root_logger.disabled = True


def log_function_call(logger: Optional[logging.Logger] = None) -> Callable[[F], F]:
    """
    Decorator to log function calls with parameters and return values.

    Args:
        logger: Logger to use. If None, a logger will be created based on the module name.

    Returns:
        Decorator function
    """

    def decorator(func: F) -> F:
        # Get the module name if logger is not provided
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not LOGGING_ENABLED:
                return func(*args, **kwargs)

            # Log function call
            arg_str = ", ".join(
                [str(arg) for arg in args] + [f"{k}={v}" for k, v in kwargs.items()]
            )
            logger.debug(f"Calling {func.__name__}({arg_str})")

            # Call the function
            result = func(*args, **kwargs)

            # Log the result
            logger.debug(f"{func.__name__} returned: {result}")

            return result

        return cast(F, wrapper)

    return decorator


def log_async_function_call(
    logger: Optional[logging.Logger] = None,
) -> Callable[[F], F]:
    """
    Decorator to log async function calls with parameters and return values.

    Args:
        logger: Logger to use. If None, a logger will be created based on the module name.

    Returns:
        Decorator function
    """

    def decorator(func: F) -> F:
        # Get the module name if logger is not provided
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not LOGGING_ENABLED:
                return await func(*args, **kwargs)

            # Log function call
            arg_str = ", ".join(
                [str(arg) for arg in args] + [f"{k}={v}" for k, v in kwargs.items()]
            )
            logger.debug(f"Calling {func.__name__}({arg_str})")

            # Call the function
            result = await func(*args, **kwargs)

            # Log the result
            logger.debug(f"{func.__name__} returned: {result}")

            return result

        return cast(F, wrapper)

    return decorator
