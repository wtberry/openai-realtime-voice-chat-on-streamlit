import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from threading import Lock

logger = logging.getLogger(__name__)

@dataclass
class Tool:
    """Represents a tool that can be used by the model."""
    name: str
    description: str
    input_schema: Dict[str, Any]

class ServerStateManager:
    """Manages the state of MCP servers."""
    def __init__(self):
        self._state_lock = Lock()
        self._active_servers: set[str] = set()
        self._server_states: Dict[str, bool] = {}
        
    def mark_server_active(self, server_name: str):
        """Mark a server as active."""
        with self._state_lock:
            self._active_servers.add(server_name)
            self._server_states[server_name] = True
            
    def mark_server_inactive(self, server_name: str):
        """Mark a server as inactive."""
        with self._state_lock:
            self._active_servers.discard(server_name)
            self._server_states[server_name] = False
            
    def is_server_active(self, server_name: str) -> bool:
        """Check if a server is active."""
        with self._state_lock:
            return self._server_states.get(server_name, False)
            
    def get_active_servers(self) -> set[str]:
        """Get set of active server names."""
        with self._state_lock:
            return self._active_servers.copy()

class MCPServerConnection:
    """Manages connection to a single MCP server."""
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.session: Optional[ClientSession] = None
        self.tools: Dict[str, Tool] = {}
        self._lock = asyncio.Lock()
        self._exit_stack = AsyncExitStack()
        self._connected = False
        self.stdio = None
        self.write = None
        
    @property
    def connected(self) -> bool:
        """Check if server is connected."""
        return self._connected
        
    async def connect(self):
        """Establish connection to server."""
        async with self._lock:
            try:
                command = self.config.get('command')
                args = self.config.get('args', [])
                env = self.config.get('env')

                server_params = StdioServerParameters(
                    command=command,
                    args=args,
                    env=env
                )

                stdio_transport = await self._exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                self.stdio, self.write = stdio_transport
                self.session = await self._exit_stack.enter_async_context(
                    ClientSession(self.stdio, self.write)
                )

                await self.session.initialize()

                # List available tools
                response = await self.session.list_tools()
                tools = response.tools
                logger.info(
                    "Connected to server '%s' with tools: %s",
                    self.name,
                    [tool.name for tool in tools]
                )
                
                # Register tools
                for tool in tools:
                    logger.debug("Registering tool from '%s':", self.name)
                    logger.debug("  Name: %s", tool.name)
                    logger.debug("  Description: %s", tool.description)
                    logger.debug("  Input Schema: %s", tool.inputSchema)
                    self.register_tool(
                        name=tool.name,
                        description=tool.description,
                        input_schema=tool.inputSchema
                    )
                    
                self._connected = True
                
            except Exception as e:
                logger.error(
                    "Failed to connect to server '%s': %s",
                    self.name,
                    str(e)
                )
                await self.close()
                raise
                
    async def close(self):
        """Close server connection and cleanup resources."""
        if not self._connected:
            return
            
        async with self._lock:
            self._connected = False
            self.tools.clear()
            
            # First clear references to session and transport
            self.session = None
            self.stdio = None
            self.write = None
            
            try:
                # Close the exit stack in the same task context
                await self._exit_stack.aclose()
            except Exception as e:
                if not isinstance(e, asyncio.CancelledError):
                    logger.error(f"Error during cleanup for server '{self.name}': {str(e)}")
            finally:
                # Create new exit stack
                self._exit_stack = AsyncExitStack()
                logger.debug("Cleaned up server '%s' resources", self.name)
            
    def register_tool(self, name: str, description: str, input_schema: Dict[str, Any]):
        """Register a new tool for this server."""
        self.tools[name] = Tool(
            name=name,
            description=description,
            input_schema=input_schema
        )
        
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool on this server."""
        if not self.connected:
            raise ConnectionError(f"Server '{self.name}' not connected")
            
        if tool_name not in self.tools:
            raise KeyError(
                f"Tool '{tool_name}' not found in server '{self.name}'"
            )
            
        async with self._lock:
            try:
                logger.debug(
                    "Executing tool '%s' on server '%s'",
                    tool_name,
                    self.name
                )
                result = await self.session.call_tool(tool_name, arguments)
                content = result.content.text if hasattr(result.content, 'text') else str(result.content)
                logger.debug("Tool execution response: %s", content)
                return content
            except Exception as e:
                logger.error(
                    "Failed to execute tool '%s' on server '%s': %s",
                    tool_name,
                    self.name,
                    str(e)
                )
                raise

class MCPServerManager:
    """Manages multiple MCP server connections."""
    def __init__(self):
        self.servers: Dict[str, MCPServerConnection] = {}
        self._lock = asyncio.Lock()
        self._state_manager = ServerStateManager()
        self.tool_queue = asyncio.Queue()
        
    async def connect_all(self, config: Dict[str, Any]):
        """Connect to all configured servers in parallel."""
        async with self._lock:
            connect_tasks = []
            for name, server_config in config.items():
                server = MCPServerConnection(name, server_config)
                self.servers[name] = server
                connect_tasks.append(self.connect_server(name))
            
            # Connect to all servers in parallel
            await asyncio.gather(*connect_tasks, return_exceptions=True)
            
    async def connect_server(self, server_name: str):
        """Connect to a specific server."""
        if server_name not in self.servers:
            raise KeyError(f"Server '{server_name}' not found in configuration")
            
        server = self.servers[server_name]
        try:
            await server.connect()
            self._state_manager.mark_server_active(server_name)
        except Exception as e:
            self._state_manager.mark_server_inactive(server_name)
            logger.error(f"Failed to connect to server '{server_name}': {str(e)}")
            raise
            
    async def close_all(self):
        """Close all server connections."""
        async with self._lock:
            # Create a list of servers to close
            servers_to_close = list(self.servers.items())
            
            # Mark all servers as inactive first
            for name, _ in servers_to_close:
                self._state_manager.mark_server_inactive(name)
            
            # Close servers sequentially to avoid task context issues
            for name, server in servers_to_close:
                try:
                    logger.debug(f"Closing server '{name}'...")
                    await server.close()
                    logger.debug(f"Server '{name}' closed successfully")
                except Exception as e:
                    if not isinstance(e, asyncio.CancelledError):
                        logger.error(f"Error closing server '{name}': {str(e)}")
            
            # Clear servers after all cleanup attempts
            self.servers.clear()
            logger.debug("All servers closed and cleared")
            
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get all available tools across all servers."""
        tools = []
        for server in self.servers.values():
            if server.connected:
                for tool in server.tools.values():
                    tools.append({
                        "type": "function",
                        "name": f"{server.name}.{tool.name}",
                        "description": tool.description,
                        "parameters": tool.input_schema
                    })
        return tools
        
    def get_server_for_tool(self, tool_name: str) -> MCPServerConnection:
        """Get the server that provides a specific tool."""
        server_name = tool_name.split('.')[0]
        if server_name not in self.servers:
            raise KeyError(f"Server '{server_name}' not found")
        return self.servers[server_name]
        
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool on the appropriate server."""
        try:
            server = self.get_server_for_tool(tool_name)
            # Remove server prefix from tool name
            actual_tool_name = '.'.join(tool_name.split('.')[1:])
            return await server.execute_tool(actual_tool_name, arguments)
        except Exception as e:
            logger.error(f"Failed to execute tool '{tool_name}': {str(e)}")
            raise
