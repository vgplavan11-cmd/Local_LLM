# challenge5/mcp_server.py
# A local MCP server that exposes three tools over stdio.
#
# MCP (Model Context Protocol) is a standard way to package tools so
# ANY agent framework can discover and call them — not just Strands.
# The server communicates over stdin/stdout (stdio transport), which
# means no network ports, no Docker, no separate terminal needed.
#
# This file is NOT run directly by you. The Strands MCPClient in
# starter.py launches it automatically as a child process.

from datetime import date
from mcp.server.fastmcp import FastMCP

# FastMCP is the simplest way to build an MCP server in Python.
# Give the server a name — this is what shows up in tool listings.
mcp = FastMCP("LocalToolsServer")


# ── MCP Tool 1: Calculator ──────────────────────────────
# @mcp.tool() works just like Strands' @tool — it reads the
# docstring and type hints to build the tool's JSON schema.
@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate a math expression and return the result.

    Use for any arithmetic: addition, subtraction, multiplication,
    division, exponentiation, etc.

    Args:
        expression: A math expression string, e.g. '(10 + 5) * 3'
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Calculator error: {e}"


# ── MCP Tool 2: Weather ─────────────────────────────────
@mcp.tool()
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Use when the user asks about weather or temperature anywhere.

    Args:
        city: City name, e.g. 'Tokyo' or 'Mumbai'
    """
    mock_data = {
        "london":   "Cloudy, 15°C, humidity 80%",
        "new york": "Sunny, 22°C, humidity 55%",
        "tokyo":    "Rainy, 18°C, humidity 90%",
        "sydney":   "Clear, 25°C, humidity 60%",
        "paris":    "Partly cloudy, 17°C, humidity 70%",
        "mumbai":   "Hot and humid, 34°C, humidity 85%",
        "chennai":  "Sunny, 36°C, humidity 75%",
        "delhi":    "Hazy, 38°C, humidity 50%",
    }
    info = mock_data.get(city.lower().strip(),
                         f"No weather data available for '{city}'.")
    return f"Weather in {city}: {info}"


# ── MCP Tool 3: Age Calculator ──────────────────────────
@mcp.tool()
def calculate_age(birth_year: int, birth_month: int, birth_day: int) -> str:
    """Calculate a person's current age from their date of birth.

    Use when the user asks how old someone is or wants an age
    calculated from a birth date.

    Args:
        birth_year:  4-digit year, e.g. 1995
        birth_month: Month number (1-12)
        birth_day:   Day of month (1-31)
    """
    try:
        dob = date(birth_year, birth_month, birth_day)
        today = date.today()
        age = today.year - dob.year - (
            (today.month, today.day) < (dob.month, dob.day)
        )
        return f"Age: {age} years old (born {dob.strftime('%B %d, %Y')})"
    except ValueError as e:
        return f"Invalid date: {e}"


# ── Entry point ─────────────────────────────────────────
# stdio transport = communicate over stdin/stdout.
# Strands MCPClient will launch this file as a subprocess and
# pipe messages to it automatically.
if __name__ == "__main__":
    mcp.run(transport="stdio")
