from __future__ import annotations

import os
import tempfile
import shutil
from typing import List, Tuple

import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_upstage import ChatUpstage, UpstageEmbeddings

SYSTEM_PROMPT = """당신은 Dot AI입니다. 업로드된 문서에서 제공된 맥락만을 사용하여 질문에 답변하세요. 규칙:
- 답변은 제공된 맥락에 엄격히 기반하세요.
- 질문이 리스트를 묻는 경우 (예: "세 가지 ...는 무엇인가?"), 맥락에 존재하면 완전한 리스트를 반환하세요.
- 맥락에 답변의 일부만 있는 경우, "문서에 부분 정보만 포함되어 있습니다"라고 말하고 찾은 내용을 보여주세요.
- 맥락에 답변이 없는 경우, "업로드된 문서에 해당 내용이 없습니다"라고 말하세요.
- 가능하면 맥락에서 짧은 구절을 인용하세요."""

# ---------------- Helpers ----------------
def uploaded_files_to_documents(uploaded_files) -> List[Document]:
    docs: List[Document] = []
    for f in uploaded_files:
        data = f.read()
        name = f.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(name)[1]) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        if name.lower().endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
            docs.extend(loader.load())
        else:
            loader = TextLoader(tmp_path)
            docs.extend(loader.load())
        os.unlink(tmp_path)
    return docs

def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "?", "!", " "],
        chunk_size=1000,
        chunk_overlap=100,
    )
    return splitter.split_documents(docs)

def format_sources(docs: List[Document], max_chars: int = 900) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    seen = set()
    for d in docs:
        source = d.metadata.get("source", "알 수 없음")
        page = d.metadata.get("page", None)
        if page is not None:
            title = f"{source} (p. {page + 1})"
        else:
            title = source
        if title in seen:
            continue
        seen.add(title)
        snippet = (d.page_content or "").strip().replace("\n", " ")
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars] + "..."
        out.append((title, snippet))
    return out

# ---------------- Sidebar ----------------
st.sidebar.title("설정")

api_key = st.sidebar.text_input(
    "API 키",
    value=os.getenv("UPSTAGE_API_KEY", ""),
    type="password",
)

temperature = st.sidebar.slider("창의성", 0.0, 1.0, 0.2, 0.1)

top_k = st.sidebar.slider("검색 수", 2, 12, 8, 1)

st.sidebar.divider()

uploaded = st.sidebar.file_uploader(
    "문서 업로드",
    type=["txt", "md", "pdf"],
    accept_multiple_files=True,
)

col1, col2 = st.sidebar.columns(2)
build_index = col1.button("등록", use_container_width=True, disabled=not uploaded)
clear_chat = col2.button("대화 초기화", use_container_width=True)

reset_kb = st.sidebar.button("DB 초기화", use_container_width=True)

if clear_chat:
    st.session_state.pop("messages", None)

if reset_kb:
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    st.session_state.pop("vectorstore", None)
    st.success("DB 초기화 완료.")
    st.stop()

# ---------------- Auto-load Chroma DB ----------------
if "vectorstore" not in st.session_state and os.path.exists("./chroma_db"):
    if not api_key:
        st.warning("API 키 입력 필요 (DB 로드).")
    else:
        os.environ["UPSTAGE_API_KEY"] = api_key
        with st.spinner("기존 DB 로드 중…"):
            embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
            vs = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
            st.session_state["vectorstore"] = vs
            st.info("기존 DB 로드 완료. 질문 가능합니다.")

# ---------------- Main UI ----------------
st.title("한국어 문서 Q&A")
st.caption("문서 업로드 후 질문하세요.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("출처"):
                for title, snippet in m["sources"]:
                    st.markdown(f"**{title}**")
                    st.write(snippet)

# Build/Upsert
if build_index:
    if not api_key:
        st.error("API 키 입력 필요.")
        st.stop()

    os.environ["UPSTAGE_API_KEY"] = api_key

    with st.spinner("분석 중…"):
        docs = uploaded_files_to_documents(uploaded)

    with st.spinner("분할 중…"):
        chunks = chunk_documents(docs)

    with st.spinner("DB 준비 중…"):
        embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
        vs = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    st.info(f"{len(chunks)}개 저장 중…")
    progress = st.progress(0)

    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        vs.add_documents(chunks[i : i + batch_size])
        progress.progress(min((i + batch_size) / len(chunks), 1.0))

    st.session_state["vectorstore"] = vs
    st.success(f"완료: 파일 {len(docs)}개, 청크 {len(chunks)}개.")

# Chat
user_q = st.chat_input("질문…")

if user_q:
    if "vectorstore" not in st.session_state:
        st.warning("문서 등록 필요.")
        st.stop()

    if not api_key:
        st.error("API 키 입력 필요.")
        st.stop()

    os.environ["UPSTAGE_API_KEY"] = api_key

    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    llm = ChatUpstage(
        model="solar-1-mini-chat",
        temperature=temperature,
        streaming=True,
    )

    vs = st.session_state["vectorstore"]

    # Retriever
    retriever = vs.as_retriever(search_kwargs={"k": top_k})

    # Retrieve docs
    docs = retriever.invoke(user_q)

    context = "\n\n---\n\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
        ]
    )

    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer = ""

        messages = prompt.format_messages(context=context, question=user_q)

        for chunk in llm.stream(messages):
            token = getattr(chunk, "content", "") or ""
            answer += token
            placeholder.markdown(answer)

        sources = format_sources(docs)
        if sources:
            with st.expander("출처"):
                for title, snippet in sources:
                    st.markdown(f"**{title}**")
                    st.write(snippet)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})