import asyncio
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
from contextlib import AsyncExitStack

async def main():
    async with AsyncExitStack() as stack:
        params = StdioServerParameters(command="npx", args=["-y", "@modelcontextprotocol/server-memory"])
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        tools = await session.list_tools()
        print(f"Tools: {tools}")
        
if __name__ == "__main__":
    asyncio.run(main())
