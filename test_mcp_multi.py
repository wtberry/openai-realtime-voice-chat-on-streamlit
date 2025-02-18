import asyncio
import json
from mcp_client import MCPClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_multi_server():
    """Test multiple MCP server connections"""
    try:
        # Initialize client
        client = MCPClient(config_path=".streamlit/mcp_config.json")
        logger.info("MCP client initialized")
        
        # Initialize all servers
        await client.initialize()
        logger.info("All servers initialized")
        
        # Get all available tools
        tools = client.get_tools()
        logger.info("Available tools across all servers:")
        for tool in tools:
            logger.info(f"  {tool['name']}: {tool['description']}")
        
        # Test tool execution from different servers
        test_calls = [
            {
                'id': '1',
                'name': 'slack.slack_list_channels',
                'arguments': json.dumps({'limit': 5})
            },
            {
                'id': '2',
                'name': 'filesystem.list_directory',
                'arguments': json.dumps({'path': '/Users/wtakahashi/Desktop'})
            }
        ]
        
        logger.info("\nExecuting tool calls...")
        results = await client.handle_tool_calls(test_calls)
        
        logger.info("\nTool execution results:")
        for call_id, result in results.items():
            logger.info(f"\nCall ID: {call_id}")
            logger.info(f"Result: {result}")
            
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
    finally:
        await client.close()
        logger.info("\nTest completed, resources cleaned up")

if __name__ == "__main__":
    asyncio.run(test_multi_server())
