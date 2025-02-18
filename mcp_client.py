import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
import json
from dataclasses import dataclass
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

@dataclass
class Tool:
    """Represents a tool that can be used by the model."""
    name: str
    description: str
    input_schema: Dict[str, Any]

@dataclass
class ToolCall:
    """Represents a tool call from the model."""
    id: str
    name: str
    arguments: Dict[str, Any]

class MCPClient:
    """Client for managing model tool calling capabilities."""
    
    def __init__(self, config_path: str = None):
        self.tools: Dict[str, Dict[str, Tool]] = {}  # server_name -> {tool_name -> Tool}
        self.active_tool_calls: List[ToolCall] = []
        self.tool_results: Dict[str, Any] = {}
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.stdio = None
        self.write = None
        self.config_path = config_path
        self.server_configs = {}
        self.server_name = None  # Initialize server_name
        self.connected = False  # Initialize connection state
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

        # Create a new exit stack for this connection
        self.exit_stack = AsyncExitStack()
        
        try:
            server_config = self.server_configs[server_name]
            command = server_config.get('command')
            args = server_config.get('args', [])
            env = server_config.get('env')

            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=env
            )

            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

            await self.session.initialize()

            # List available tools
            response = await self.session.list_tools()
            tools = response.tools
            logger.info("Connected to server '%s' with tools: %s", server_name, [tool.name for tool in tools])
            
            # Set server name and connection state
            self.server_name = server_name
            self.connected = True
            
            # Initialize tools dictionary for this server if needed
            if server_name not in self.tools:
                self.tools[server_name] = {}
            
            # Register tools
            for tool in tools:
                logger.debug("Registering tool from '%s':", server_name)
                logger.debug("  Name: %s", tool.name)
                logger.debug("  Description: %s", tool.description)
                logger.debug("  Input Schema: %s", json.dumps(tool.inputSchema, indent=2))
                self.register_tool(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.inputSchema
                )
        except Exception as e:
            logger.error("Failed to connect to server '%s': %s", server_name, str(e))
            # Clean up on failure
            await self.exit_stack.aclose()
            self.exit_stack = AsyncExitStack()
            self.server_name = None
            self.connected = False
            self.session = None
            self.stdio = None
            self.write = None
            raise

    async def close(self) -> None:
        """Clean up resources."""
        if self.server_name:
            logger.info("Disconnecting from server '%s'", self.server_name)
            # Clear tools for the disconnected server
            if self.server_name in self.tools:
                self.tools[self.server_name].clear()
        self.connected = False
        self.server_name = None
        self.clear_tool_state()
        
        # Create a new exit stack before closing the old one
        old_exit_stack = self.exit_stack
        self.exit_stack = AsyncExitStack()
        
        # Close the old exit stack
        await old_exit_stack.aclose()
        
        self.session = None
        self.stdio = None
        self.write = None
        logger.debug("Cleaned up MCP client resources")

    async def _access_mcp_resources(self) -> Dict[str, Any]:
        """Access MCP server resources.
        
        Returns:
            Dictionary of server resources
            
        Raises:
            ConnectionError: If resource access fails
        """
        try:
            # Use access_mcp_resource tool to get server resources
            return {
                "server_name": self.server_name,
                "tools": await self._get_server_tools()
            }
        except Exception as e:
            raise ConnectionError(f"Failed to access MCP resources: {str(e)}")
        
    async def _initialize_tools(self, resources: Dict[str, Any]) -> None:
        """Initialize tools from server resources.
        
        Args:
            resources: Server resource information
            
        Raises:
            RuntimeError: If tool initialization fails
        """
        try:
            tools = resources.get("tools", [])
            for tool in tools:
                self.register_tool(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    input_schema=tool.get("input_schema", {})
                )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize tools: {str(e)}")
            
    async def _get_server_tools(self) -> List[Dict[str, Any]]:
        """Get available tools from MCP server.
        
        Returns:
            List of tool definitions
            
        Raises:
            RuntimeError: If tool retrieval fails
            ConnectionError: If not connected to server
        """
        if not self.session:
            raise ConnectionError("Not connected to MCP server")
            
        try:
            # Get tools through session
            response = await self.session.list_tools()
            return [{
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            } for tool in response.tools]
        except Exception as e:
            raise RuntimeError(f"Failed to get server tools: {str(e)}")
            
    async def _use_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Use an MCP tool.
        
        Args:
            tool_name: Name of tool to use
            arguments: Tool arguments
            
        Returns:
            Tool response
            
        Raises:
            RuntimeError: If tool use fails
            ConnectionError: If not connected to server
        """
        if not self.session:
            raise ConnectionError("Not connected to MCP server")
            
        try:
            logger.debug("Using MCP tool on server '%s':", self.server_name)
            logger.debug("  Tool: %s", tool_name)
            logger.debug("  Arguments: %s", json.dumps(arguments, indent=2))
            
            # Execute tool through MCP server session
            result = await self.session.call_tool(tool_name, arguments)
            
            # Extract text content from TextContent object
            content = result.content.text if hasattr(result.content, 'text') else str(result.content)
            logger.debug("Tool execution response: %s", content)
            return {"result": content}
        except Exception as e:
            logger.error("Failed to execute tool '%s' on server '%s': %s", tool_name, self.server_name, str(e))
            raise RuntimeError(f"Failed to use MCP tool: {str(e)}")

    def register_tool(self, name: str, description: str, input_schema: Dict[str, Any]) -> None:
        """Register a new tool.
        
        Args:
            name: Tool name
            description: Tool description
            input_schema: JSON Schema for tool inputs
        """
        if self.server_name not in self.tools:
            self.tools[self.server_name] = {}
            
        self.tools[self.server_name][name] = Tool(
            name=name,
            description=description,
            input_schema=input_schema
        )

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get available tools in Azure OpenAI Realtime API format.
        
        Returns:
            List of tool definitions in Azure OpenAI format
        """
        # Only return tools from the currently connected server
        if self.server_name and self.server_name in self.tools:
            return [{
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema
            } for tool in self.tools[self.server_name].values()]
        return []

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

            tool_name = tool_call["name"]
            logger.debug("Attempting to execute tool '%s' on server '%s'", tool_name, self.server_name)
            if self.server_name not in self.tools or tool_name not in self.tools[self.server_name]:
                available_tools = []
                if self.server_name in self.tools:
                    available_tools = list(self.tools[self.server_name].keys())
                logger.error("Tool '%s' not found in server '%s'. Available tools: %s", 
                           tool_name, self.server_name, available_tools)
                raise KeyError(f"Tool '{tool_name}' not registered for server '{self.server_name}'")

            try:
                arguments = json.loads(tool_call.get("arguments", "{}"))
                
                # Create and track tool call
                call = ToolCall(
                    id=tool_call["id"],
                    name=tool_name,
                    arguments=arguments
                )
                self.active_tool_calls.append(call)
                
                logger.debug("Executing tool '%s' with arguments: %s", tool_name, json.dumps(arguments, indent=2))
                # Execute tool using MCP server
                result = await self._execute_tool(tool_name, arguments)
                logger.debug("Tool '%s' execution result: %s", tool_name, result)
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
                    # Log error but continue processing other results
                    print(f"Tool call failed: {str(result)}")
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
                    # Log error but continue processing other calls
                    print(f"Tool call failed: {str(e)}")
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

    async def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool using the MCP server.
        
        Args:
            tool_name: Name of tool to execute
            arguments: Tool arguments
            
        Returns:
            Tool execution result
            
        Raises:
            RuntimeError: If tool execution fails
            ConnectionError: If not connected to server
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to MCP server")
            
        try:
            # Prepare tool call arguments
            tool_args = {
                "server_name": self.server_name,
                "tool_name": tool_name,
                "arguments": arguments
            }
            
            # Execute tool through MCP server
            response = await self._use_mcp_tool(tool_name, arguments)
            
            # Extract result from response
            if "error" in response:
                raise RuntimeError(response["error"])
                
            return response.get("result", str(response))
        except Exception as e:
            raise RuntimeError(f"Tool execution failed: {str(e)}")

    def clear_tool_state(self) -> None:
        """Clear tool call state."""
        self.active_tool_calls = []
        self.tool_results = {}
        
    @property
    def is_connected(self) -> bool:
        """Check if client is connected to MCP server."""
        return getattr(self, 'connected', False)
