from mcp.server.fastmcp import FastMCP
from diffusers import StableDiffusionPipeline
import torch
import base64
from io import BytesIO

# 모델 로드
print("모델 로딩 중...")
# 모델 로드 후 강제로 safety_checker 제거
pipe = StableDiffusionPipeline.from_pretrained(
    "prompthero/openjourney",
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False
).to("cuda")

# 이거 추가 (강제 제거)
pipe.safety_checker = None
print("모델 로딩 완료!")

mcp = FastMCP("iann-image-server")

@mcp.tool()
def generate_image(prompt: str) -> str:
    """텍스트 프롬프트로 이미지를 생성합니다"""
    print(f"이미지 생성 중: {prompt}")

    pipe.safety_checker = None

    image = pipe(prompt).images[0]

    # 파일 저장
    image.save("output.png")

    # base64 변환
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/png;base64,{img_base64}"

if __name__ == "__main__":
    mcp.run(transport="sse")