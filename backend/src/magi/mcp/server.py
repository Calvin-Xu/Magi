"""
MCP server implementation for Magi.

This module exposes a Model Context Protocol (MCP) server that can be mounted to the
main FastAPI application. It provides tools to interact with the knowledge graph.
"""

import os
from mcp.server.fastmcp import FastMCP

# Read a friendly name from environment or default
SERVER_NAME: str = os.environ.get("MCP_SERVER_NAME", "Magi Knowledge Graph")

mcp = FastMCP(SERVER_NAME)


@mcp.tool()
def debug_hello() -> str:
    """
    Simple MCP tool that returns a Hello World message.
    Useful for verifying that the MCP server is working.
    """
    return "Hello World"
