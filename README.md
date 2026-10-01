# Dot AI — 문서 기반 Q&A

PDF, TXT, Markdown 문서를 등록하고, 관련 내용을 찾아 한국어로 질문할 수 있는 **개인 RAG 프로젝트**입니다. 문서 검색과 답변 생성의 흐름을 직접 구현하고 개선하기 위해 만들었습니다.

Streamlit으로 화면을 구성하고, OpenAI의 **`gpt-4o-mini`**로 답변을 생성합니다. 문서와 질문의 벡터화에는 별도 임베딩 모델인 **`text-embedding-3-small`**을 사용하며, 검색용 데이터는 로컬 Chroma DB에 보관합니다.

## 주요 기능

- 여러 PDF / TXT / MD 파일을 등록하고 관련 문서 검색
- 검색한 문서 내용을 바탕으로 답변 스트리밍
- 근거 문서의 파일명과 PDF 페이지 번호 표시
- 앱을 재시작해도 유지되는 로컬 벡터 DB
- 같은 내용의 파일을 다시 등록할 때 중복 임베딩 방지
- 검색 문서 수(`top-k`)와 답변 생성 온도(`temperature`) 조절
- 세션 내 대화 기록, 대화 초기화 및 OpenAI DB 초기화

## 동작 방식

1. 업로드한 문서에서 텍스트를 추출하고 작은 조각으로 분할합니다.
2. `text-embedding-3-small`로 각 조각을 벡터화하여 Chroma에 저장합니다.
3. 질문과 관련된 문서 조각을 검색합니다.
4. 검색 결과를 `gpt-4o-mini`에 전달해 답변을 생성하고, 검색된 출처를 함께 보여줍니다.

문서 처리와 검색 로직은 `rag.py`, 화면과 세션 관리는 `test.py`에 분리했습니다. API 클라이언트와 벡터 저장소를 세션에서 재사용하고, 이미 등록된 파일은 다시 임베딩하지 않아 불필요한 초기화와 API 호출을 줄입니다.

## 기술 구성

| 용도 | 기술 |
| --- | --- |
| 웹 UI | Streamlit |
| 답변 생성 | OpenAI `gpt-4o-mini` |
| 임베딩 | OpenAI `text-embedding-3-small` |
| 문서 처리 및 RAG 구성 | LangChain, PyPDFLoader, RecursiveCharacterTextSplitter |
| 벡터 저장소 | 로컬 Chroma |
| 설정 | 환경 변수, python-dotenv |

## 설치 및 실행

Python 3.11 이상과 OpenAI API 키가 필요합니다. 프로젝트 폴더에서 실행하세요.

```bash
git clone https://github.com/PSCHEDULE/rag-evaluation-pipeline.git
cd rag-evaluation-pipeline
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

`.env`의 값을 본인 키로 채우세요.

```dotenv
OPENAI_API_KEY=your-openai-api-key
```

`.env`를 사용하지 않고 실행 후 사이드바에 키를 입력해도 됩니다. 사이드바 입력값이 있으면 해당 키를 사용합니다. 키를 소스 코드에 직접 작성하거나 Git에 커밋하지 마세요. `.env`는 `.gitignore`에 포함되어 있습니다.

Windows에서는 다음 명령으로 실행합니다.

```powershell
.\.venv\Scripts\python -m streamlit run test.py
```

macOS / Linux에서는 가상 환경을 활성화한 상태에서 실행합니다.

```bash
streamlit run test.py
```

## 사용 방법

1. 사이드바에서 OpenAI API 키를 입력하거나 `.env` 설정을 사용합니다.
2. PDF, TXT 또는 MD 파일을 선택하고 **문서 등록**을 누릅니다.
3. 채팅창에 질문을 입력하고 답변과 출처를 확인합니다.
4. 필요에 따라 **검색 수**(`top-k`)와 **창의성**(`temperature`)을 조절합니다.

파일 내용의 SHA-256 해시를 비교해 같은 파일의 중복 등록을 건너뜁니다. 파일명을 바꿔 올려도 내용이 같으면 기존 출처를 유지합니다. 내용이 바뀐 파일은 새 버전으로 추가되므로, 이전 내용까지 제거하려면 **OpenAI DB 초기화** 후 유지할 문서만 다시 등록하세요. **대화 초기화**는 현재 대화 기록만 지웁니다.

대화 기록은 현재 브라우저 세션에서 확인할 수 있습니다. 각 질문은 독립적으로 검색·답변하므로, 이전 대화에만 있는 내용을 지칭하기보다 질문에 필요한 조건을 함께 적어주세요.

## 저장소와 기존 Upstage 데이터 전환

OpenAI 임베딩 데이터는 프로젝트 폴더의 `chroma_db_openai/`에 저장합니다. 이전 Upstage 버전의 `chroma_db/`와 분리되어 기존 데이터가 자동으로 변경되지는 않습니다.

Solar 임베딩은 OpenAI 임베딩과 호환되지 않으므로 **기존 원본 문서를 다시 업로드해야 합니다.** 폴더 이름만 바꾸거나 기존 벡터를 복사해서 재사용할 수 없습니다. 앱의 **OpenAI DB 초기화**는 새 저장소의 등록 문서를 지우며 기존 `chroma_db/`는 건드리지 않습니다.

이 앱은 **한 사람이 로컬에서 사용하는 방식**을 전제로 합니다. 같은 서버를 사용하는 세션은 하나의 문서 DB를 공유하며, API 키를 바꿔도 문서 저장소가 분리되지는 않습니다.

## 사용 시 참고

- TXT와 MD는 UTF-8 인코딩을 지원합니다. PDF는 텍스트 추출이 가능해야 하며, 스캔 이미지에 대한 OCR은 제공하지 않습니다.
- 문서 텍스트와 질문은 임베딩을 위해 OpenAI API로 전송되며, 답변 생성에는 검색된 문서 내용과 현재 질문이 전달됩니다. 로컬 DB에도 추출한 문서 내용과 벡터가 저장됩니다.
- 문서 등록 시 임베딩 비용이, 질문 시 질문 임베딩과 답변 생성 비용이 발생합니다. 모델별 사용량은 [OpenAI 공식 모델 문서](https://developers.openai.com/api/docs/models/gpt-4o-mini)와 [임베딩 안내](https://developers.openai.com/api/docs/guides/embeddings)를 참고하세요.
- 출처는 검색된 근거를 보여주며 답변의 정확성을 보증하지 않습니다. 중요한 내용은 원문과 함께 확인하세요.

## 프로젝트 구조

```text
.
├── test.py             # Streamlit 앱 진입점
├── rag.py              # 문서 처리, 중복 확인, 벡터 저장소 및 검색
├── tests/              # API 호출 없이 실행하는 회귀 테스트
├── requirements.txt    # 직접 사용하는 런타임 의존성
├── .env.example        # API 키 설정 예시
└── .gitignore          # 키, 로컬 DB, 가상 환경 제외
```

## 테스트

의존성을 설치한 가상 환경에서 실행합니다. 테스트는 실제 OpenAI API 키나 유료 API 호출 없이 동작합니다.

```bash
python -m unittest discover -s tests -v
```

Windows에서 가상 환경을 활성화하지 않았다면 `python` 대신 `.\.venv\Scripts\python`을 사용하세요.
