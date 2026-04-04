import asyncio
from mcp.client.sse import sse_client
from mcp import ClientSession

async def main():
    async with sse_client("http://localhost:8000/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("사용 가능한 툴:", [t.name for t in tools.tools])

            result = await session.call_tool(
                "generate_image",
                {"prompt": "a beautiful mountain landscape at sunset, digital art"}
            )
            import base64
            from PIL import Image
            from io import BytesIO

            img_data = result.content[0].text
            img_data = img_data.replace("data:image/png;base64,", "")
            img_bytes = base64.b64decode(img_data)
            img = Image.open(BytesIO(img_bytes))
            img.save("mcp_output.png")
            print("mcp_output.png 저장 완료!")

asyncio.run(main())