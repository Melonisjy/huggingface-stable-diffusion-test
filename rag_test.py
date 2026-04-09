from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. LLM 연결
llm = ChatOllama(model="llama3.2")

# 2. 파일 읽기
loader = TextLoader("mcp_server.py", encoding="utf-8")
docs = loader.load()

# 3. 청크로 쪼개기
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"청크 개수: {len(chunks)}")

# 4. vector DB에 저장
embeddings = OllamaEmbeddings(model="llama3.2")
vectorstore = Chroma.from_documents(chunks, embeddings)
print("vector DB 저장 완료!")

# 5. RAG 체인 구성
retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_template("""
아래 코드를 참고해서 질문에 한국어로 답해줘.

코드:
{context}

질문: {question}
""")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. 질문
result = chain.invoke("web_search 툴은 어떻게 작동해?")
print(result)

result = chain.invoke("이 코드에서 사용하는 모델 이름이 뭐야?")
print(result)