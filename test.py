from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple

import streamlit as st

# ====================== LangChain imports ======================
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_upstage import ChatUpstage, UpstageEmbeddings

# ====================== CONFIG ======================
SYSTEM_PROMPT = """당신은 Dot AI입니다. 업로드된 문서에서 제공된 맥락만을 사용하여 질문에 답변하세요. 규칙:
- 답변은 제공된 맥락에 엄격히 기반하세요.
- 질문이 리스트를 묻는 경우 (예: "세 가지 ...는 무엇인가?"), 맥락에 존재하면 완전한 리스트를 반환하세요.
- 맥락에 답변의 일부만 있는 경우, "문서에 부분 정보만 포함되어 있습니다"라고 말하고 찾은 내용을 보여주세요.
- 맥락에 답변이 없는 경우, "업로드된 문서에 해당 내용이 없습니다"라고 말하세요.
- 가능하면 맥락에서 짧은 구절을 인용하세요."""

CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "solar-embedding-1-large"
LLM_MODEL = "solar-1-mini-chat"
BATCH_SIZE = 50
MAX_SNIPPET_CHARS = 900


# ====================== HELPERS ======================
def files_to_documents(uploaded_files) -> List[Document]:
    """Streamlit UploadedFile → LangChain Document (TemporaryDirectory로 자동 정리)."""
    documents: List[Document] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for uploaded_file in uploaded_files:
            file_path = Path(tmp_dir) / uploaded_file.name
            file_path.write_bytes(uploaded_file.getvalue())

            loader = (
                PyPDFLoader(str(file_path))
                if uploaded_file.name.lower().endswith(".pdf")
                else TextLoader(str(file_path))
            )
            documents.extend(loader.load())

    return documents


def chunk_documents(docs: List[Document]) -> List[Document]:
    """문서를 1000자 청크로 분할."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "?", "!", " "],
        chunk_size=1000,
        chunk_overlap=100,
    )
    return splitter.split_documents(docs)


def format_sources(docs: List[Document]) -> List[Tuple[str, str]]:
    """출처 중복 제거 + 스니펫 정리."""
    sources: List[Tuple[str, str]] = []
    seen = set()

    for doc in docs:
        source = doc.metadata.get("source", "알 수 없음")
        page = doc.metadata.get("page")
        title = f"{source} (p. {page + 1})" if page is not None else source

        if title in seen:
            continue
        seen.add(title)

        snippet = (doc.page_content or "").strip().replace("\n", " ")
        if len(snippet) > MAX_SNIPPET_CHARS:
            snippet = snippet[:MAX_SNIPPET_CHARS] + "..."

        sources.append((title, snippet))

    return sources


def setup_api_key(api_key: str) -> None:
    """API 키 검증 + 환경변수 설정."""
    if not api_key:
        st.error("API 키가 필요합니다.")
        st.stop()
    os.environ["UPSTAGE_API_KEY"] = api_key

    if st.session_state.get("validated_api_key") == api_key:
        return

    try:
        UpstageEmbeddings(model=EMBEDDING_MODEL).embed_query("test")
        st.session_state["validated_api_key"] = api_key
    except Exception as e:
        st.session_state.pop("validated_api_key", None)
        if "401" in str(e) or "invalid_api_key" in str(e):
            st.error("API 키가 유효하지 않습니다. https://console.upstage.ai 에서 확인해주세요.")
            st.stop()
        st.error(f"API 연결 실패: {e}")
        raise


def initialize_vectorstore() -> Chroma | None:
    """기존 Chroma DB 자동 로드 (없으면 None)."""
    if not Path(CHROMA_DB_PATH).exists():
        return None

    api_key = st.session_state.get("api_key")
    if not api_key:
        st.warning("API 키를 입력해주세요 (기존 DB 로드).")
        return None

    setup_api_key(api_key)

    with st.spinner("기존 DB 로드 중..."):
        embeddings = UpstageEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings,
        )
        st.info("기존 DB 로드 완료")
        return vectorstore


# ====================== UI ======================
st.set_page_config(page_title="한국어 문서 Q&A", page_icon="📚")

st.title("한국어 문서 Q&A")
st.caption("문서를 업로드한 뒤 자연어로 질문하세요.")

# Sidebar
with st.sidebar:
    st.header("설정")

    api_key = st.text_input(
        "Upstage API 키",
        value=os.getenv("UPSTAGE_API_KEY", ""),
        type="password",
        help="https://console.upstage.ai 에서 발급",
    )
    st.session_state["api_key"] = api_key

    temperature = st.slider("창의성", 0.0, 1.0, 0.2, 0.1)
    top_k = st.slider("검색 수", 2, 12, 8, 1)

    st.divider()

    uploaded_files = st.file_uploader(
        "문서 업로드",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns(2)
    build_index = col1.button("문서 등록", use_container_width=True, disabled=not uploaded_files)
    clear_chat = col2.button("대화 초기화", use_container_width=True)

    if st.button("DB 전체 초기화", use_container_width=True):
        vs = st.session_state.pop("vectorstore", None)
        if vs is not None:
            try:
                vs._client.clear_system_cache()  # SQLite 연결 해제
            except Exception:
                pass
            del vs

        if Path(CHROMA_DB_PATH).exists():
            shutil.rmtree(CHROMA_DB_PATH)

        st.success("DB가 초기화되었습니다.")
        st.rerun()

if clear_chat:
    st.session_state.pop("messages", None)
    st.rerun()

# Auto-load existing DB
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = initialize_vectorstore()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("출처", expanded=False):
                for title, snippet in message["sources"]:
                    st.markdown(f"**{title}**")
                    st.write(snippet)

# Index building
if build_index and uploaded_files:
    setup_api_key(api_key)

    with st.spinner("문서 분석 중..."):
        documents = files_to_documents(uploaded_files)

    with st.spinner("텍스트 청킹 중..."):
        chunks = chunk_documents(documents)

    with st.spinner("벡터 DB 준비 중..."):
        embeddings = UpstageEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings,
        )

    st.info(f"{len(chunks):,}개 청크 저장 중...")
    progress_bar = st.progress(0)

    for i in range(0, len(chunks), BATCH_SIZE):
        vectorstore.add_documents(chunks[i : i + BATCH_SIZE])
        progress_bar.progress(min((i + BATCH_SIZE) / len(chunks), 1.0))

    st.session_state["vectorstore"] = vectorstore
    st.success(f"완료! 문서 {len(documents)}개 → 청크 {len(chunks)}개")

# Chat
if user_query := st.chat_input("질문을 입력하세요..."):
    if "vectorstore" not in st.session_state or st.session_state["vectorstore"] is None:
        st.warning("먼저 문서를 등록해주세요.")
        st.stop()

    setup_api_key(api_key)

    # 사용자 메시지
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # RAG + LLM
    vectorstore: Chroma = st.session_state["vectorstore"]
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            docs = retriever.invoke(user_query)
            context = "\n\n---\n\n".join(doc.page_content for doc in docs)

            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
            ])

            llm = ChatUpstage(
                model=LLM_MODEL,
                temperature=temperature,
                streaming=True,
            )

            messages = prompt.format_messages(context=context, question=user_query)

            answer_placeholder = st.empty()
            full_answer = ""

            for chunk in llm.stream(messages):
                full_answer += chunk.content
                answer_placeholder.markdown(full_answer + "▌")

            answer_placeholder.markdown(full_answer)

            # 출처
            sources = format_sources(docs)
            if sources:
                with st.expander("출처", expanded=False):
                    for title, snippet in sources:
                        st.markdown(f"**{title}**")
                        st.write(snippet)

    # 대화 기록 저장
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "sources": sources,
    })