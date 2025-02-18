import asyncio
import logging
from typing import Dict, List, Optional, Any
import json
from dataclasses import dataclass
from mcp_server import MCPServerManager, Tool

logger = logging.getLogger(__name__)

@dataclass
class ToolCall:
    """Represents a tool call from the model."""
    id: str
    name: str
    arguments: Dict[str, Any]

class MCPClient:
    """Client for managing model tool calling capabilities."""
    
    def __init__(self, config_path: str = None):
        self.server_manager = MCPServerManager()
        self.active_tool_calls: List[ToolCall] = []
        self.tool_results: Dict[str, Any] = {}
        self.config_path = config_path
        self.server_configs = {}
        # Maintain these for backward compatibility
        self.tools: Dict[str, Dict[str, Tool]] = {}
        self.server_name = None
        self.connected = False
        if config_path:
            logger.debug("Loading MCP config from: %s", config_path)
            self.load_config()

    def load_config(self) -> None:
        """Load MCP server configurations from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                if 'mcpServers' in config:
                    self.server_configs = config['mcpServers']
                else:
                    raise ValueError("Invalid config file: missing 'mcpServers' key")
        except Exception as e:
            raise RuntimeError(f"Failed to load MCP config: {str(e)}")

    async def connect_to_server(self, server_name: str):
        """Connect to an MCP server using configuration.

        Args:
            server_name: Name of the server in the configuration
        """
        if not self.server_configs:
            raise RuntimeError("No server configurations loaded")
        
        if server_name not in self.server_configs:
            raise ValueError(f"Server '{server_name}' not found in configuration")

        try:
            # Connect using server manager
            await self.server_manager.connect_server(server_name)
            
            # Update legacy state for backward compatibility
            self.server_name = server_name
            self.connected = True
            
            # Update legacy tools dictionary
            server = self.server_manager.servers[server_name]
            if server_name not in self.tools:
                self.tools[server_name] = {}
            self.tools[server_name] = server.tools
            
            logger.info(
                "Connected to server '%s' with tools: %s",
                server_name,
                list(server.tools.keys())
            )
        except Exception as e:
            logger.error("Failed to connect to server '%s': %s", server_name, str(e))
            self.server_name = None
            self.connected = False
            raise

    async def close(self) -> None:
        """Clean up resources."""
        try:
            await self.server_manager.close_all()
        except Exception as e:
            logger.error(f"Error during server cleanup: {str(e)}")
        finally:
            self.connected = False
            self.server_name = None
            self.clear_tool_state()
            self.tools.clear()
            logger.debug("Cleaned up MCP client resources")

    async def initialize(self):
        """Initialize connections to all configured servers."""
        if not self.server_configs:
            raise RuntimeError("No server configurations loaded")
        await self.server_manager.connect_all(self.server_configs)

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get available tools in Azure OpenAI Realtime API format."""
        return self.server_manager.get_all_tools()

    async def handle_tool_calls(self, tool_calls: List[Dict[str, Any]], parallel: bool = True) -> Dict[str, str]:
        """Handle multiple tool calls from the model.
        
        Args:
            tool_calls: List of tool call objects from model
            parallel: Whether to execute tool calls in parallel
            
        Returns:
            Dictionary mapping tool call IDs to their results
            
        Raises:
            ValueError: If tool call format is invalid
            KeyError: If tool is not registered
        """
        if not tool_calls:
            return {}

        async def execute_tool_call(tool_call: Dict[str, Any]) -> tuple[str, str]:
            """Execute a single tool call."""
            if not isinstance(tool_call, dict):
                raise ValueError("Tool call must be a dictionary")

            required_keys = ["id", "name", "arguments"]
            if not all(key in tool_call for key in required_keys):
                raise ValueError(f"Tool call missing required keys: {required_keys}")

            try:
                arguments = json.loads(tool_call.get("arguments", "{}"))
                
                # Create and track tool call
                call = ToolCall(
                    id=tool_call["id"],
                    name=tool_call["name"],
                    arguments=arguments
                )
                self.active_tool_calls.append(call)
                
                logger.debug("Executing tool '%s' with arguments: %s", call.name, json.dumps(arguments, indent=2))
                # Execute tool using server manager
                result = await self.server_manager.execute_tool(call.name, arguments)
                logger.debug("Tool '%s' execution result: %s", call.name, result)
                self.tool_results[call.id] = result
                
                return call.id, result
                
            except json.JSONDecodeError:
                raise ValueError("Invalid tool arguments JSON")
            except Exception as e:
                raise RuntimeError(f"Tool execution failed: {str(e)}")

        # Execute tool calls
        if parallel:
            # Execute all tool calls concurrently
            tasks = [execute_tool_call(call) for call in tool_calls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle any errors
            output = {}
            for result in results:
                if isinstance(result, Exception):
                    logger.error("Tool call failed: %s", str(result))
                    continue
                call_id, call_result = result
                output[call_id] = call_result
            return output
        else:
            # Execute tool calls sequentially
            output = {}
            for call in tool_calls:
                try:
                    call_id, result = await execute_tool_call(call)
                    output[call_id] = result
                except Exception as e:
                    logger.error("Tool call failed: %s", str(e))
            return output

    def get_tool_results(self, tool_call_ids: Optional[List[str]] = None) -> Dict[str, str]:
        """Get results of tool calls.
        
        Args:
            tool_call_ids: Optional list of tool call IDs to retrieve.
                          If None, returns all results.
            
        Returns:
            Dictionary mapping tool call IDs to their results
        """
        if tool_call_ids is None:
            return self.tool_results.copy()
        return {
            call_id: self.tool_results[call_id]
            for call_id in tool_call_ids
            if call_id in self.tool_results
        }

    def clear_tool_state(self) -> None:
        """Clear tool call state."""
        self.active_tool_calls = []
        self.tool_results = {}
        
    @property
    def is_connected(self) -> bool:
        """Check if client is connected to MCP server."""
        return any(server.connected for server in self.server_manager.servers.values())
