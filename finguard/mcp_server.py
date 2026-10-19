"""MCP server exposing the governed tools. Run: python -m finguard.mcp_server"""
from mcp.server.fastmcp import FastMCP

from . import tools

mcp = FastMCP("finguard")


@mcp.tool()
def run_select(sql: str) -> list[dict]:
    """Run a single read-only SELECT against the transactions table (max 50 rows)."""
    return tools.run_select(sql)


@mcp.tool()
def get_transaction(txn_id: int) -> dict | None:
    """Fetch one transaction by id."""
    return tools.get_transaction(txn_id)


if __name__ == "__main__":
    mcp.run()
