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
file_handler.setLevel(logging.DEBUG)
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


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Name for the logger, typically the module name

    Returns:
        A configured logger instance
    """
    return logging.getLogger(name)


def set_global_log_level(level: int) -> None:
    """
    Set the global log level for all loggers.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
    """
    root_logger.setLevel(level)
    console_handler.setLevel(level)


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
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not LOGGING_ENABLED:
                return func(*args, **kwargs)

            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)

            # Log function call with arguments
            arg_str = ", ".join([repr(a) for a in args])
            kwarg_str = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
            params = f"{arg_str}{', ' if arg_str and kwarg_str else ''}{kwarg_str}"
            logger.debug(f"Calling {func.__name__}({params})")

            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} returned: {repr(result)}")
                return result
            except Exception as e:
                logger.exception(f"Exception in {func.__name__}: {str(e)}")
                raise

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
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not LOGGING_ENABLED:
                return await func(*args, **kwargs)

            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)

            # Log function call with arguments
            arg_str = ", ".join([repr(a) for a in args])
            kwarg_str = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
            params = f"{arg_str}{', ' if arg_str and kwarg_str else ''}{kwarg_str}"
            logger.debug(f"Calling async {func.__name__}({params})")

            try:
                result = await func(*args, **kwargs)
                logger.debug(f"Async {func.__name__} returned: {repr(result)}")
                return result
            except Exception as e:
                logger.exception(f"Exception in async {func.__name__}: {str(e)}")
                raise

        return cast(F, wrapper)

    return decorator
