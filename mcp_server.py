from dotenv import load_dotenv
load_dotenv()

import os
os.environ["HF_HOME"] = os.getenv("HF_HOME", "D:\\huggingface_cache")

# HF_HOME 설정 끝난 다음에 diffusers import
from mcp.server.fastmcp import FastMCP
from diffusers import StableDiffusionPipeline
from tavily import TavilyClient
import torch
import base64
from io import BytesIO

# 모델 로드
print("모델 로딩 중...")
pipe = StableDiffusionPipeline.from_pretrained(
    "prompthero/openjourney",
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False
).to("cuda")

pipe.safety_checker = None
print("모델 로딩 완료!")

mcp = FastMCP("iann-image-server")

@mcp.tool()
def generate_image(prompt: str) -> str:
    """텍스트 프롬프트로 이미지를 생성합니다"""
    print(f"이미지 생성 중: {prompt}")

    pipe.safety_checker = None
    image = pipe(prompt).images[0]
    image.save("output.png")

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/png;base64,{img_base64}"

@mcp.tool()
def web_search(query: str) -> str:
    """웹에서 최신 정보를 검색합니다"""
    print(f"웹 검색 중: {query}")

    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = client.search(query, max_results=5)

    results = response.get("results", [])

    if not results:
        return "검색 결과가 없습니다."

    output = []
    for i, r in enumerate(results, 1):
        output.append(f"{i}. {r['title']}\n   {r['url']}\n   {r.get('content', '')[:200]}")

    return "\n\n".join(output)

if __name__ == "__main__":
    mcp.run(transport="sse")