import asyncio
from mcp.client.sse import sse_client
from mcp import ClientSession

async def main():
    async with sse_client("http://localhost:8000/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("사용 가능한 툴:", [t.name for t in tools.tools])

            # web_search 테스트
            print("\n--- 웹 검색 테스트 ---")
            result = await session.call_tool(
                "web_search",
                {"query": "IANN Computing QUIC P2P GPU platform"}
            )
            print(result.content[0].text)

asyncio.run(main())