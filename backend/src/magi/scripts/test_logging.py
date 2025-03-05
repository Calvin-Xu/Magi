#!/usr/bin/env python3
"""
Test script for the logging system.

This script demonstrates how to use the logging system in Magi.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from magi.utils import (
    disable_logging,
    enable_logging,
    get_logger,
    log_async_function_call,
    log_function_call,
    set_global_log_level,
)

# Create a logger for this module
logger = get_logger(__name__)


@log_function_call()
def test_sync_function(param1: str, param2: int) -> str:
    """Test synchronous function with logging."""
    logger.debug(f"Debug message in test_sync_function with {param1=}, {param2=}")
    logger.info(f"Info message in test_sync_function with {param1=}, {param2=}")
    logger.warning(f"Warning message in test_sync_function with {param1=}, {param2=}")
    logger.error(f"Error message in test_sync_function with {param1=}, {param2=}")
    return f"Result: {param1} - {param2}"


@log_async_function_call()
async def test_async_function(param1: str, param2: int) -> str:
    """Test asynchronous function with logging."""
    logger.debug(f"Debug message in test_async_function with {param1=}, {param2=}")
    logger.info(f"Info message in test_async_function with {param1=}, {param2=}")
    logger.warning(f"Warning message in test_async_function with {param1=}, {param2=}")
    logger.error(f"Error message in test_async_function with {param1=}, {param2=}")

    # Simulate some async work
    await asyncio.sleep(1)

    return f"Async Result: {param1} - {param2}"


async def main():
    """Main function to test the logging system."""
    print("Testing Magi Logging System")
    print("===========================")

    # Test with default log level (INFO)
    print("\n1. Testing with default log level (INFO):")
    result1 = test_sync_function("test1", 42)
    print(f"Sync result: {result1}")

    result2 = await test_async_function("test2", 84)
    print(f"Async result: {result2}")

    # Test with DEBUG log level
    print("\n2. Testing with DEBUG log level:")
    set_global_log_level(logging.DEBUG)
    result3 = test_sync_function("test3", 123)
    print(f"Sync result: {result3}")

    result4 = await test_async_function("test4", 456)
    print(f"Async result: {result4}")

    # Test with logging disabled
    print("\n3. Testing with logging disabled:")
    disable_logging()
    result5 = test_sync_function("test5", 789)
    print(f"Sync result: {result5}")

    # Re-enable logging
    print("\n4. Testing with logging re-enabled:")
    enable_logging()
    set_global_log_level(logging.DEBUG)
    result6 = test_sync_function("test6", 999)
    print(f"Sync result: {result6}")

    print("\nLogging test completed. Check the logs directory for the log file.")


if __name__ == "__main__":
    asyncio.run(main())
