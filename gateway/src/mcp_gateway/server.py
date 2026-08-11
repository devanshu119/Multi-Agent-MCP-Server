"""MCP Gateway Server.

This module implements a gateway server that:
1. Reads MCP server configurations
2. Spawns and manages MCP server processes using official mcp SDK
3. Forwards requests to appropriate MCP servers
4. Aggregates responses back to clients
"""

import asyncio
import json
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from contextlib import AsyncExitStack

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    command: str
    args: List[str]
    env: Dict[str, str] = field(default_factory=dict)


@dataclass
class MCPServer:
    """Represents a running MCP server."""
    name: str
    config: MCPServerConfig
    session: ClientSession
    exit_stack: AsyncExitStack
    tools: List[Dict] = field(default_factory=list)


class Gateway:
    """MCP Gateway that manages server connections and forwards requests."""
    
    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
        
    async def start_server(self, name: str, config: MCPServerConfig) -> MCPServer:
        """Start an MCP server and initialize its client session."""
        try:
            logger.info(f"Starting MCP server: {name}")
            
            env = os.environ.copy()
            env.update(config.env)
            
            server_params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env=env
            )
            
            exit_stack = AsyncExitStack()
            
            # Start the stdio client
            stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
            read, write = stdio_transport
            
            # Start the session
            session = await exit_stack.enter_async_context(ClientSession(read, write))
            
            # Initialize connection
            await session.initialize()
            
            # Query tools
            tools_response = await session.list_tools()
            tools = [{"name": t.name, "description": t.description, "inputSchema": t.inputSchema} for t in tools_response.tools]
            
            server = MCPServer(
                name=name,
                config=config,
                session=session,
                exit_stack=exit_stack,
                tools=tools
            )
            
            self.servers[name] = server
            return server
            
        except Exception as e:
            logger.error(f"Error starting server {name}: {str(e)}")
            # Don't raise, just log so other servers can start
            return None
    
    async def start_all_servers(self, config_path: str) -> None:
        """Start all configured MCP servers."""
        try:
            logger.info(f"Loading config from: {config_path}")
            if not os.path.exists(config_path):
                logger.warning(f"Config path {config_path} does not exist")
                return

            with open(config_path) as f:
                config = json.load(f)
            
            if not config.get('mcp', {}).get('servers'):
                logger.warning("No MCP servers configured in config file")
                return
                
            tasks = []
            for name, server_config in config['mcp']['servers'].items():
                task = asyncio.create_task(
                    self.start_server(name, MCPServerConfig(**server_config))
                )
                tasks.append(task)
            
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("All servers started")
            
        except Exception as e:
            logger.error(f"Error starting servers: {str(e)}")
            raise
    
    async def list_all_tools(self) -> List[Dict[str, Any]]:
        """Get all available tools from all servers."""
        tools = []
        for server in self.servers.values():
            for tool in server.tools:
                tool_dict = dict(tool)
                tool_dict["server"] = server.name
                tools.append(tool_dict)
        return tools
    
    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Call a tool on the appropriate server."""
        for server in self.servers.values():
            if any(t["name"] == tool_name for t in server.tools):
                try:
                    logger.info(f"Calling tool {tool_name} on server {server.name}")
                    result = await server.session.call_tool(tool_name, arguments)
                    
                    # Convert CallToolResult back to dict for JSON serialization
                    content_list = []
                    for item in getattr(result, "content", []):
                        if hasattr(item, "text"):
                            content_list.append({"type": "text", "text": item.text})
                        elif hasattr(item, "data"):
                            content_list.append({"type": "image", "data": item.data, "mimeType": getattr(item, "mimeType", "")})
                            
                    return {"content": content_list, "isError": getattr(result, "isError", False)}
                except Exception as e:
                    logger.error(f"Error calling tool {tool_name}: {str(e)}")
                    raise
        raise ValueError(f"Tool {tool_name} not found")
    
    async def shutdown(self) -> None:
        """Shutdown all MCP servers."""
        for server in self.servers.values():
            try:
                await server.exit_stack.aclose()
            except Exception as e:
                logger.error(f"Error shutting down server {server.name}: {str(e)}")
        self.servers.clear()


gateway = Gateway()

@app.on_event("startup")
async def startup():
    """Initialize the gateway on startup."""
    config_path = os.environ.get("MCP_CONFIG", "config.json")
    await gateway.start_all_servers(config_path)


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    await gateway.shutdown()


@app.post("/message")
async def message_endpoint(request: Request):
    """Handle incoming messages from clients."""
    try:
        msg = await request.json()
        
        if msg.get("method") == "tools/list":
            tools = await gateway.list_all_tools()
            return JSONResponse({"tools": tools})
        
        elif msg.get("method") == "tools/call":
            params = msg.get("params", {})
            result = await gateway.call_tool(
                params.get("name"),
                params.get("arguments", {})
            )
            return JSONResponse(result)
        
        return JSONResponse({"error": "Unknown method"}, status_code=400)
    except Exception as e:
        logger.error(f"Error handling message: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("MCP_PORT", "8808"))
    uvicorn.run(app, host="0.0.0.0", port=port)
