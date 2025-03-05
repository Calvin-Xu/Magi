"""Utility modules for the Magi project."""

from .logging import (
    disable_logging,
    enable_logging,
    get_logger,
    log_async_function_call,
    log_function_call,
    set_global_log_level,
)

__all__ = [
    "get_logger",
    "set_global_log_level",
    "enable_logging",
    "disable_logging",
    "log_function_call",
    "log_async_function_call",
]
