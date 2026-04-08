from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# 로컬 LLM 연결
llm = ChatOllama(model="llama3.2")

# 프롬프트 템플릿
prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 친절한 AI 어시스턴트야. 한국어로 답해줘."),
    ("user", "{question}")
])

# Chain 연결
chain = prompt | llm

# 실행
response = chain.invoke({"question": "랭체인이 뭐야? 한 줄로 설명해줘"})
print(response.content)