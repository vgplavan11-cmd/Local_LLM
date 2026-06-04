# challenge5/starter.py
# MCP Chatbot — Strands SDK + Ollama llama3.2:3b + Local MCP Server
#
# What is MCP?
#   Model Context Protocol is an open standard (by Anthropic) that lets you
#   package tools into a server any agent can connect to — like a USB standard
#   for AI tools. Your agent doesn't need to know HOW a tool works, just that
#   it exists. This makes tools reusable across different agent frameworks.
#
# Architecture of this challenge:
#
#   ┌─────────────────────┐        stdio         ┌──────────────────────┐
#   │   starter.py        │ ◄──────────────────► │   mcp_server.py      │
#   │   Strands Agent     │   (subprocess pipe)  │   FastMCP Server     │
#   │   Ollama llama3.2   │                      │   - calculator       │
#   │   MCPClient         │                      │   - get_weather      │
#   └─────────────────────┘                      │   - calculate_age    │
#                                                └──────────────────────┘
#
# starter.py launches mcp_server.py as a child process automatically.
# You only run: python starter.py

import sys
import os
from strands import Agent
from strands.models.ollama import OllamaModel
from strands.tools.mcp import MCPClient
from mcp.client.stdio import stdio_client, StdioServerParameters


# ═══════════════════════════════════════════════════════
# SECTION 1 — MCP Server connection
# ═══════════════════════════════════════════════════════
# StdioServerParameters tells MCPClient how to launch our MCP server.
#   command → the Python executable (same one running this script)
#   args    → the server script to run as a subprocess
#
# When MCPClient starts, it:
#   1. Spawns mcp_server.py as a child process
#   2. Connects to it over stdin/stdout
#   3. Calls list_tools() to discover all available tools
#   4. Makes those tools available to the Strands agent

# Resolve the path to mcp_server.py relative to this file's location.
SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "mcp_server.py")

# StdioServerParameters holds the launch config for the subprocess.
mcp_server_params = StdioServerParameters(
    command=sys.executable,   # same Python binary as this script
    args=[SERVER_SCRIPT],     # launches mcp_server.py as a subprocess
)

# MCPClient expects a callable that returns an async context manager.
# stdio_client(params) IS that async context manager — so we pass a
# lambda that calls stdio_client(mcp_server_params) when invoked.
mcp_transport = lambda: stdio_client(mcp_server_params)


# ═══════════════════════════════════════════════════════
# SECTION 2 — Ollama model setup
# ═══════════════════════════════════════════════════════
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.7,
)


# ═══════════════════════════════════════════════════════
# SECTION 3 — MCPClient context + Agent creation
# ═══════════════════════════════════════════════════════
# MCPClient is used as a context manager (with ... as ...).
# On enter → starts the server subprocess, handshakes, fetches tool list.
# On exit  → cleanly shuts the subprocess down.
#
# mcp_client.list_tools_sync() returns a list of Strands-compatible tool
# objects built from the MCP server's tool schemas. We pass them straight
# into Agent(tools=[...]) — no extra wiring needed.

def run_chatbot():
    """Start the MCP server, wire up the agent, and run the chat loop."""

    print("Starting MCP server subprocess...")

    with MCPClient(mcp_transport) as mcp_client:

        # Discover all tools the MCP server exposes.
        mcp_tools = mcp_client.list_tools_sync()

        print(f"Connected! Tools available: "
              f"{[t.tool_name for t in mcp_tools]}\n")

        # Create the Strands agent with MCP tools.
        # The agent will call these tools automatically when appropriate.
        agent = Agent(
            model=ollama_model,
            tools=mcp_tools,
            system_prompt=(
                "You are MCPBot, a helpful assistant powered by MCP tools.\n"
                "You have access to:\n"
                "  1. calculator    — for any math expressions\n"
                "  2. get_weather   — for weather in any city\n"
                "  3. calculate_age — to compute age from a birth date\n\n"
                "Always use the right tool when the user's question calls for it. "
                "Be concise and friendly."
            ),
        )

        # ═══════════════════════════════════════════════════════
        # SECTION 4 — Interactive chat loop
        # ═══════════════════════════════════════════════════════
        print("=" * 54)
        print("  MCPBot — Strands + Ollama + Local MCP Server")
        print("=" * 54)
        print("Try asking:")
        print("  • What is 256 / 8 + 15?")
        print("  • What's the weather in Paris?")
        print("  • How old is someone born on January 20, 2000?")
        print("  • Explain what MCP is")
        print("  • Type 'exit' to quit")
        print("=" * 54 + "\n")

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            if not user_input:
                continue

            # The agent receives the message, decides which MCP tool(s)
            # to call (if any), calls them via the subprocess pipe,
            # gets the results, and composes the final reply — all automatically.
            response = agent(user_input)
            print(f"\nMCPBot: {response}\n")

    # The 'with' block ends here — MCP server subprocess is shut down cleanly.


# ═══════════════════════════════════════════════════════
# SECTION 5 — Entry point
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    run_chatbot()
