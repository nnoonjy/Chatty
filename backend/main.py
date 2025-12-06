import os
import shutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv  # .env 파일 로드용

# LangChain 관련
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. 환경 설정 (.env 파일에서 키 가져오기)
load_dotenv()

# API 키 확인 (디버깅용)
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️ 오류: .env 파일을 찾을 수 없거나 OPENAI_API_KEY가 없습니다.")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 설정 ---
TARGET_URLS = [
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000002",
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000003",
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000421",
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000004",
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000005",
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000006",
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000007",
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000008",
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000009"
]
DATA_PATH = "./data"
CHROMA_PATH = "./chroma_db"


def load_and_process_data():
    documents = []
    print("🌐 웹 데이터 수집 중...")
    try:
        web_loader = WebBaseLoader(TARGET_URLS)
        documents.extend(web_loader.load())
    except Exception as e:
        print(f"⚠️ 웹 크롤링 스킵: {e}")

    if os.path.exists(DATA_PATH):
        print("📂 로컬 파일(PDF/TXT) 수집 중...")
        # PDF
        pdf_loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents.extend(pdf_loader.load())
        # TXT
        txt_loader = DirectoryLoader(DATA_PATH, glob="**/*.txt")  # TextLoader 자동 적용됨
        documents.extend(txt_loader.load())

    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # DB 초기화 및 재생성
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(),
        persist_directory=CHROMA_PATH
    )
    print("✅ 임베딩 완료 및 DB 저장됨!")
    return vectorstore


# 서버 시작 로직
if os.path.exists(CHROMA_PATH):
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())
    print("✅ 기존 DB 로드됨")
else:
    vectorstore = load_and_process_data()

retriever = vectorstore.as_retriever() if vectorstore else None
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

template = """
당신은 부산대학교 공개 정보 봇입니다. 
[문맥]을 보고 답변하세요. 모르면 모른다고 하세요.

[문맥]:
{context}

[질문]:
{question}

[답변]:
"""
prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = None
if retriever:
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )


class QueryRequest(BaseModel):
    query: str


@app.post("/chat")
async def chat(request: QueryRequest):
    if not rag_chain:
        return {"answer": "학습된 데이터가 없습니다. ./data 폴더에 파일을 넣고 재시작해보세요."}
    response = rag_chain.invoke(request.query)
    return {"answer": response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)