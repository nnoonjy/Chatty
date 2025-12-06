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
    #학적
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000002", #학적변동-일반휴학
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000003", #학적변동-병역휴학
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000421", #학적변동-기타휴학
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000004", #학적변동-복학
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000005", #학적변동-교내전과
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000006", #학적변동-재입학
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000007", #학적변동-제적
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000008", #학적변동-자퇴
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000009", #학적변동-학적부기재사항정정
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000405", #전공지원신청안내-복수전공
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000406", #전공지원신청안내-부전공
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000407", #전공지원신청안내-연계전공
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000429", #전공지원신청안내-소단위전공
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000433", #전공지원신청안내-학생자율전공
    #교육과정
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000030", #교과목개요-교양과목
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000031", #교과목개요-전공과목
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000032", #교과목개요-일반선택과목
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000153", #교육과정-교육과정
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000249", #학점취득및성적평가-학점취득
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000250", #학점취득및성적평가-학업성적평가
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000251", #학점취득및성적평가-계절수업
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000026", #타대학이수학점안내
    #수업
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000043", #수강신청안내-수강신청 유의사항
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000044", #수강신청안내-수강신청 및 수강정정 안내
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000045", #수강신청안내-교과목 이수에 관한 사항
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000046", #수강신청안내-직업능력 개발과정
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000047", #수강신청안내-현장실습학기제
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000048", #수강신청안내-교직과목 이수안내
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000049", #수강신청안내-수강취소
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000050", #수강신청안내-출석시수 및 시험
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000051", #수강신청안내-폐강 기준 및 수강신청 불인정
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000052", #수강신청안내-교양선택 동일영역 대비표
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000252", #교육과정별수강신청안내-교욱과정 적용대상자
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000257", #교직과정안내-교직과정설치 및 운영목적
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000261", #교직과정안내-일반대학 교직설치학과
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000262", #교직과정안내-교직이수예정자선발 및 교직이수
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000263", #교직과정안내-교직복수(연계)전공 선발
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000264", #교직과정안내-교원자격 무시험검
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000265", #교직과정안내-교직교육과정 적용 기준년도
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000327", #교수계획표-교수계획표 조회
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000335", #수강편람-수강편람
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000360", #수강편람-타대생 수강편람
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000388", #수강신청및확인-수강신청
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000339", #수강신청및확인-희망과목담기 신청인원 확인
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000368", #시간표조회-강의실별 시간표 조회
    #성적
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000059", #성적관리기준-성적관리기준안내
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000060", #성적관리기준-성적관리기준안내
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000061", #성적관리기준-성적관리기준안내
    #장학
    "", #장학공지사항
    "", #장학금개요안내-우리대학장학금개요
    "", #종류별장학안내-학부장학금
    "", #종류별장학안내-대학원장학금
    "", #종류별장학안내-외국인장학금
    "", #종류별장학안내-공고.공모형장학금
    "", #종류별장학안내-학자금대출
    "", #절차별장학안내-장학금 신청
    "", #절차별장학안내-장학생 선말 및 제외
    "", #절차별장학안내-장학금 지급 및 반납
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000370", #장학금신청-우선선발장학금신청
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000371", #장학금신청-장학생(근로)신청
    "https://onestop.pusan.ac.kr/page?menuCD=000000000000372" #장학금신청-유니웰장학금신청
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