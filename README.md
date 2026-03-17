# Capstone Design and Practice  
**Dot AI: RAG 기반 한국어 문서 Q&A 시스템**

**개발 기간**: 2026년 1학기  

---

## 1. 프로젝트 개요
Upstage Solar LLM과 LangChain을 활용한 **문서 업로드형 질문 답변(RAG) 웹 애플리케이션**입니다.  
PDF 및 텍스트 파일을 업로드하면 자동으로 벡터 DB를 구축하고, 사용자의 질문에 대해 **문서 내용에 엄격히 기반한 정확한 답변**과 **출처**를 제공합니다.

**주요 목표** - 환각(Hallucination) 최소화  
- 영속적 로컬 벡터 DB 구현  
- 실시간 스트리밍 답변 제공  

---

## 2. 주요 기능
- 다중 PDF/TXT 파일 업로드 및 자동 분석
- Chroma DB 영속화 (앱 재시작 후에도 유지)
- 출처(Source) 표시 (파일명 + 페이지 번호)
- 답변 품질 조절 (Temperature, Top-K)
- 대화 기록 유지 및 DB 초기화 기능

---

## 3. 사용 기술 스택
| 분야          | 기술 스택                                    |
|---------------|----------------------------------------------|
| Frontend      | Streamlit                                    |
| LLM           | Upstage solar-1-mini-chat                    |
| Embeddings    | Upstage solar-embedding-1-large              |
| Vector DB     | Chroma (persist_directory)                   |
| Framework     | LangChain (RAG, 문서 로더, splitter)           |
| 기타          | PyPDFLoader, RecursiveCharacterTextSplitter  |

---

## 4. 설치 및 실행 방법

### 1) 의존성 설치
미리 준비된 요구사항 파일을 통해 필요한 라이브러리를 한 번에 설치합니다.
```bash
pip install -r requirements.txt
```

### 2) 실행
```bash
streamlit run test.py
```

### 3) 사용 절차
1. 사이드바에 Upstage API 키 입력  
2. 문서 업로드 후 **등록** 버튼 클릭  
3. 하단 채팅창에서 질문 입력  

**주의**: 문서를 등록하면 로컬에 `./chroma_db` 폴더가 생성되며, 사이드바의 **DB 초기화** 버튼으로 전체 삭제가 가능합니다.

---

**Capstone Design and Practice** 본 프로젝트는 응답의 정확도를 넘어 '신뢰할 수 있는 AI'를 위한 평가 체계 구축에 집중합니다.
