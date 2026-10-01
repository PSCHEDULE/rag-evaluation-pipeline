from __future__ import annotations

import hashlib
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, AuthenticationError, RateLimitError

from rag import (
    CHROMA_DB_PATH,
    LLM_MODEL,
    create_embeddings,
    create_llm,
    format_sources,
    has_documents,
    index_documents,
    open_vectorstore,
    reset_vectorstore,
    stream_answer,
)


def sync_api_key(api_key: str) -> None:
    """Drop session-local clients when credentials change, without changing process env."""
    fingerprint = hashlib.sha256(api_key.encode()).hexdigest() if api_key else None
    if st.session_state.get("client_key") != fingerprint:
        for name in ("vectorstore", "llm", "llm_temperature"):
            st.session_state.pop(name, None)
        st.session_state["client_key"] = fingerprint


def get_vectorstore(api_key: str):
    if not api_key:
        raise ValueError("OpenAI API 키를 입력해주세요.")
    if st.session_state.get("vectorstore") is None:
        st.session_state["vectorstore"] = open_vectorstore(create_embeddings(api_key))
    return st.session_state["vectorstore"]


def get_llm(api_key: str, temperature: float):
    if st.session_state.get("llm") is None or st.session_state.get("llm_temperature") != temperature:
        st.session_state["llm"] = create_llm(api_key, temperature)
        st.session_state["llm_temperature"] = temperature
    return st.session_state["llm"]


def show_error(error: Exception) -> None:
    # Provider error bodies can contain credentials or document text; do not echo them.
    if isinstance(error, AuthenticationError):
        st.error("OpenAI API 키가 유효하지 않습니다. 키를 확인한 뒤 다시 시도해주세요.")
    elif isinstance(error, RateLimitError):
        st.error("OpenAI 요청 한도 또는 API 사용 잔액을 확인해주세요. 잠시 후 다시 시도할 수 있습니다.")
    elif isinstance(error, APIConnectionError):
        st.error("OpenAI에 연결하지 못했습니다. 네트워크를 확인한 뒤 다시 시도해주세요.")
    elif isinstance(error, APIStatusError):
        st.error("OpenAI 요청에 실패했습니다. 모델 접근 권한과 API 상태를 확인해주세요.")
    elif isinstance(error, (UnicodeError, ValueError)):
        st.error("문서를 처리하지 못했습니다. PDF 상태와 TXT/MD 파일의 UTF-8 인코딩을 확인해주세요.")
    else:
        st.error("처리 중 오류가 발생했습니다. 파일과 로컬 DB 상태를 확인한 뒤 다시 시도해주세요.")


def render_sources(sources) -> None:
    if sources:
        with st.expander("출처", expanded=False):
            for title, snippet in sources:
                st.write(title)
                st.write(snippet)


def main() -> None:
    load_dotenv(Path(__file__).with_name(".env"))
    st.set_page_config(page_title="Dot AI · 한국어 문서 Q&A", page_icon="📚")
    st.title("한국어 문서 Q&A")
    st.caption(f"Dot AI · {LLM_MODEL} · 문서를 업로드한 뒤 자연어로 질문하세요.")
    st.session_state.setdefault("messages", [])

    with st.sidebar:
        st.header("설정")
        api_key = st.text_input(
            "OpenAI API 키",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            key="api_key",
            help="https://platform.openai.com/api-keys 에서 발급하거나 .env에 OPENAI_API_KEY를 설정하세요.",
        ).strip()
        sync_api_key(api_key)
        temperature = st.slider("창의성", 0.0, 1.0, 0.2, 0.1)
        top_k = st.slider("검색 수", 2, 12, 8, 1)
        st.divider()
        uploaded_files = st.file_uploader(
            "문서 업로드", type=["txt", "md", "pdf"], accept_multiple_files=True
        )
        col1, col2 = st.columns(2)
        build_index = col1.button("문서 등록", use_container_width=True, disabled=not uploaded_files)
        clear_chat = col2.button("대화 초기화", use_container_width=True)
        clear_db = st.button("OpenAI DB 초기화", use_container_width=True)
        st.caption("등록한 문서는 이 앱의 로컬 DB에 보관됩니다. 질문은 매번 독립적으로 검색합니다.")

    if clear_chat:
        st.session_state["messages"] = []

    if clear_db:
        try:
            vectorstore = st.session_state.get("vectorstore")
            if vectorstore is None and Path(CHROMA_DB_PATH).exists():
                # Reset is local and does not require an OpenAI key.
                vectorstore = open_vectorstore(None)
            if vectorstore is not None:
                reset_vectorstore(vectorstore)
            st.session_state.pop("vectorstore", None)
            st.session_state["messages"] = []
            st.success("OpenAI DB와 대화가 초기화되었습니다.")
        except Exception as error:
            show_error(error)

    # Retry loading after a key is entered or changed, including after an initial empty key.
    if api_key and Path(CHROMA_DB_PATH).exists() and st.session_state.get("vectorstore") is None:
        try:
            get_vectorstore(api_key)
        except Exception as error:
            show_error(error)

    if Path("chroma_db").exists():
        st.info("기존 Solar DB는 OpenAI 임베딩과 호환되지 않습니다. 원본 문서를 다시 등록해주세요.")

    if build_index:
        if not api_key:
            st.warning("OpenAI API 키를 입력해주세요.")
        else:
            progress = st.progress(0.0)
            try:
                with st.spinner("문서 분석 및 등록 중..."):
                    stats = index_documents(get_vectorstore(api_key), uploaded_files, progress.progress)
                st.success(f"문서 {stats.indexed_files}개 등록 · 새 청크 {stats.chunks:,}개 저장")
                if stats.skipped_files:
                    st.info(f"이미 등록된 문서 {stats.skipped_files}개는 건너뛰었습니다.")
                if stats.empty_files:
                    st.warning(f"텍스트가 없는 파일 {stats.empty_files}개는 건너뛰었습니다. 스캔 PDF에는 OCR이 필요합니다.")
            except Exception as error:
                show_error(error)
            finally:
                progress.empty()

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(message.get("sources"))

    if user_query := st.chat_input("질문을 입력하세요..."):
        if not api_key:
            st.warning("OpenAI API 키를 입력해주세요.")
            return
        user_query = user_query.strip()
        if not user_query:
            return
        try:
            vectorstore = get_vectorstore(api_key)
            if not has_documents(vectorstore):
                st.warning("먼저 문서를 등록해주세요.")
                return
            with st.chat_message("user"):
                st.markdown(user_query)
            with st.chat_message("assistant"):
                with st.spinner("문서 검색 중..."):
                    docs = vectorstore.similarity_search(user_query, k=top_k)
                if not docs:
                    st.warning("검색된 문서가 없습니다. 문서를 다시 등록해주세요.")
                    return
                answer = st.write_stream(stream_answer(get_llm(api_key, temperature), user_query, docs))
                sources = format_sources(docs)
                render_sources(sources)
            # Save only complete turns, so failed/partial requests can be retried cleanly.
            st.session_state["messages"].extend([
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": answer, "sources": sources},
            ])
        except Exception as error:
            show_error(error)


if __name__ == "__main__":
    main()
