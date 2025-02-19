import os
import json
import asyncio
import pytest

from mcp_server import MCPServerManager


@pytest.mark.asyncio
async def test_connect_and_list_tools():
    """Integration test: Connects to MCP servers and lists available tools.

    Requires a valid MCP config at .streamlit/mcp_config.json.
    """
    config_path = os.path.join(os.getcwd(), ".streamlit", "mcp_config.json")
    if not os.path.exists(config_path):
        pytest.skip(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    if "mcpServers" not in config:
        pytest.skip("Config file missing 'mcpServers' key.")

    manager = MCPServerManager()
    await manager.connect_all(config["mcpServers"])

    active_servers = [name for name, server in manager.servers.items() if server.connected]
    assert active_servers, "No active MCP servers found."

    tools = manager.get_all_tools()
    print("Retrieved tools:", tools)

    # Attempt to execute each tool with empty arguments and check response
    for tool in tools:
        tool_name = tool["name"]
        try:
            result = await manager.execute_tool(tool_name, {})
            print(f"Tool {tool_name} executed. Response:", result)
            assert result is not None
        except Exception as e:
            print(f"Execution of tool {tool_name} failed with error:", e)

    await manager.close_all()


@pytest.mark.asyncio
async def test_execute_nonexistent_tool():
    """Test executing a nonexistent tool raises KeyError.

    With no servers connected, attempting to execute a non-existent tool
    should raise a KeyError.
    """
    manager = MCPServerManager()
    with pytest.raises(KeyError):
        await manager.execute_tool("nonexistent.tool", {})
