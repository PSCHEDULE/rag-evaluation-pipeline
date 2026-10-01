"""Offline regression tests: no OpenAI requests or real credentials are needed."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

import rag


@dataclass
class Upload:
    name: str
    data: bytes

    def getvalue(self) -> bytes:
        return self.data


class MemoryStore:
    """Small store double that records the documents actually sent for embedding."""

    def __init__(self):
        self.documents: dict[str, Document] = {}
        self.embedded_ids: list[str] = []
        self.add_calls = 0
        self.fail_on_call: int | None = None

    def get(self, where=None, include=None, limit=None):
        matches = list(self.documents.items())
        if where:
            conditions = where.get("$and", [where])
            matches = [
                (key, document)
                for key, document in matches
                if all(
                    document.metadata.get(field) == value
                    for condition in conditions
                    for field, value in condition.items()
                )
            ]
        if limit:
            matches = matches[:limit]
        return {
            "ids": [key for key, _ in matches],
            "metadatas": [document.metadata for _, document in matches],
        }

    def add_documents(self, documents, ids):
        self.add_calls += 1
        if self.add_calls == self.fail_on_call:
            raise RuntimeError("interrupted embedding request")
        for key, document in zip(ids, documents):
            self.embedded_ids.append(key)
            self.documents[key] = document


class ClientTests(unittest.TestCase):
    def test_real_clients_send_openai_models_and_consume_mocked_responses(self):
        requests = []

        def respond(request):
            body = json.loads(request.content)
            requests.append((request.url.path, body))
            if request.url.path.endswith("/embeddings"):
                return httpx.Response(
                    200,
                    json={
                        "object": "list",
                        "model": rag.EMBEDDING_MODEL,
                        "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
                        "usage": {"prompt_tokens": 1, "total_tokens": 1},
                    },
                )
            chunk = {
                "id": "chatcmpl-offline",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": rag.LLM_MODEL,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "답변"}, "finish_reason": None}],
            }
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode(),
            )

        with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
            with (
                patch.object(rag, "OpenAIEmbeddings", partial(rag.OpenAIEmbeddings, http_client=http_client)),
                patch.object(rag, "ChatOpenAI", partial(rag.ChatOpenAI, http_client=http_client)),
            ):
                embeddings = rag.create_embeddings("sk-offline-test")
                # Avoid downloading tokenizer assets in this transport-level offline check.
                embeddings.check_embedding_ctx_length = False
                self.assertEqual(embeddings.embed_query("text"), [0.1, 0.2, 0.3])
                llm = rag.create_llm("sk-offline-test")
                self.assertEqual("".join(rag.stream_answer(llm, "질문", [])), "답변")
        self.assertEqual(requests[0][1]["model"], "text-embedding-3-small")
        self.assertEqual(requests[1][1]["model"], "gpt-4o-mini")
        self.assertTrue(requests[1][1]["stream"])

    def test_credentials_stay_session_local_without_validation_calls(self):
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "environment-key"}),
            patch.object(rag, "OpenAIEmbeddings") as embeddings_class,
            patch.object(rag, "ChatOpenAI") as llm_class,
        ):
            rag.create_embeddings("  session-key  ")
            rag.create_llm("session-key", temperature=0.4)
            self.assertEqual(os.environ["OPENAI_API_KEY"], "environment-key")
            self.assertEqual(embeddings_class.call_args.kwargs["api_key"], "session-key")
            self.assertEqual(embeddings_class.call_args.kwargs["model"], "text-embedding-3-small")
            self.assertEqual(llm_class.call_args.kwargs["api_key"], "session-key")
            self.assertEqual(llm_class.call_args.kwargs["model"], "gpt-4o-mini")
            self.assertGreater(llm_class.call_args.kwargs["max_tokens"], 0)
            self.assertLessEqual(llm_class.call_args.kwargs["max_retries"], 2)
            embeddings_class.return_value.embed_query.assert_not_called()
            llm_class.return_value.invoke.assert_not_called()

    def test_blank_key_rejected_before_client_creation(self):
        with patch.object(rag, "ChatOpenAI") as llm_class:
            with self.assertRaisesRegex(ValueError, "API"):
                rag.create_llm(" \t")
            llm_class.assert_not_called()


class DocumentTests(unittest.TestCase):
    def test_utf8_bom_and_untrusted_filename_preserve_safe_source(self):
        upload = Upload("../../private\\보고서.md", "\ufeff한국어 문서".encode("utf-8"))
        document = rag.files_to_documents([upload])[0]
        self.assertEqual(document.page_content, "한국어 문서")
        self.assertEqual(document.metadata["source"], "보고서.md")
        self.assertEqual(document.metadata["file_hash"], hashlib.sha256(upload.data).hexdigest())

    def test_pdf_tempfile_cannot_escape_directory_and_is_cleaned(self):
        captured_paths = []

        def load_pdf(path):
            captured_paths.append(Path(path))
            self.assertEqual(Path(path).name, "document.pdf")
            self.assertEqual(Path(path).read_bytes(), b"fake-pdf")
            return Mock(load=lambda: [Document(page_content="text", metadata={"source": path, "page": 0})])

        with patch.object(rag, "PyPDFLoader", side_effect=load_pdf):
            document = rag.files_to_documents([Upload("C:\\escape\\report.PDF", b"fake-pdf")])[0]
        self.assertFalse(captured_paths[0].exists())
        self.assertEqual(document.metadata["source"], "report.PDF")
        self.assertEqual(document.metadata["page"], 0)

    def test_invalid_encoding_and_unsupported_file_type_report_errors(self):
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            rag.files_to_documents([Upload("notes.txt", b"\xff")])
        with self.assertRaisesRegex(ValueError, "지원하지"):
            rag.files_to_documents([Upload("notes.exe", b"text")])

    def test_long_unbroken_text_is_bounded(self):
        chunks = rag.chunk_documents([Document(page_content="한" * 3400, metadata={"source": "a.txt"})])
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(chunk.page_content) <= rag.CHUNK_SIZE for chunk in chunks))
        self.assertTrue(all(chunk.metadata["source"] == "a.txt" for chunk in chunks))


class IndexTests(unittest.TestCase):
    def test_duplicate_content_skips_loading_and_embedding_even_after_rename(self):
        store = MemoryStore()
        first = rag.index_documents(store, [Upload("first.txt", b"same content")])
        with patch.object(rag, "_load_file") as load_file:
            second = rag.index_documents(store, [Upload("renamed.txt", b"same content")])
            load_file.assert_not_called()
        self.assertEqual((first.indexed_files, first.chunks), (1, 1))
        self.assertEqual((second.skipped_files, second.chunks), (1, 0))
        self.assertEqual(len(store.embedded_ids), 1)

    def test_changed_content_with_same_filename_is_indexed(self):
        store = MemoryStore()
        stats = rag.index_documents(store, [Upload("notes.txt", b"before"), Upload("notes.txt", b"after")])
        self.assertEqual(stats.indexed_files, 2)
        self.assertEqual(len(store.documents), 2)

    def test_interrupted_batch_resumes_without_reembedding_completed_chunks(self):
        store = MemoryStore()
        store.fail_on_call = 2
        upload = Upload("long.txt", b"a" * 4500)
        expected_count = len(rag.chunk_documents(rag.files_to_documents([upload])))
        with patch.object(rag, "BATCH_SIZE", 2):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                rag.index_documents(store, [upload])
            self.assertEqual(len(store.documents), 2)
            store.fail_on_call = None
            resumed = rag.index_documents(store, [upload])
        self.assertEqual(resumed.chunks, expected_count - 2)
        self.assertEqual(len(store.embedded_ids), expected_count)
        self.assertEqual(len(set(store.embedded_ids)), expected_count)
        self.assertEqual(rag.index_documents(store, [upload]).skipped_files, 1)

    def test_empty_inputs_do_not_embed_and_progress_completes(self):
        store = MemoryStore()
        progress = []
        stats = rag.index_documents(store, [Upload("blank.txt", b" \n\t")], progress.append)
        self.assertEqual(stats.empty_files, 1)
        self.assertEqual(stats.chunks, 0)
        self.assertEqual(store.add_calls, 0)
        self.assertEqual(progress[-1], 1.0)
        self.assertEqual(rag.index_documents(store, [], progress.append).chunks, 0)
        self.assertEqual(progress[-1], 1.0)


class AnswerTests(unittest.TestCase):
    def test_prompt_includes_citable_sources_and_stream_yields_text_only(self):
        docs = [Document(page_content="근거 내용", metadata={"source": "report.pdf", "page": 2})]
        llm = Mock()
        llm.stream.return_value = [AIMessageChunk(content=""), AIMessageChunk(content="답변")]
        self.assertEqual(list(rag.stream_answer(llm, "질문", docs)), ["답변"])
        sent = llm.stream.call_args.args[0]
        self.assertIn("report.pdf (p. 3)", sent[1].content)
        self.assertIn("근거 내용", sent[1].content)
        self.assertIn("질문", sent[1].content)

    def test_sources_deduplicate_page_and_limit_snippet(self):
        docs = [
            Document(page_content="a" * 1200, metadata={"source": "report.pdf", "page": 0}),
            Document(page_content="duplicate", metadata={"source": "report.pdf", "page": 0}),
            Document(page_content="other", metadata={"source": "report.pdf", "page": 1}),
        ]
        sources = rag.format_sources(docs)
        self.assertEqual([title for title, _ in sources], ["report.pdf (p. 1)", "report.pdf (p. 2)"])
        self.assertEqual(len(sources[0][1]), rag.MAX_SNIPPET_CHARS + 3)


class ChromaIntegrationTests(unittest.TestCase):
    def test_persistent_store_dedup_retrieval_and_public_reset(self):
        # A subprocess releases Chroma's SQLite handles before Windows removes the directory.
        script = textwrap.dedent(
            """
            import sys
            from langchain_core.embeddings import Embeddings
            import rag

            class OfflineEmbeddings(Embeddings):
                calls = 0
                def embed_documents(self, texts):
                    self.calls += len(texts)
                    return [[float(len(text) % 7 + 1), 1.0, 0.5] for text in texts]
                def embed_query(self, text):
                    return [1.0, 1.0, 0.5]

            class Upload:
                name = 'example.txt'
                def getvalue(self):
                    return b'Offline persistent document retrieval.'

            embeddings = OfflineEmbeddings()
            store = rag.open_vectorstore(embeddings, sys.argv[1])
            assert not rag.has_documents(store)
            assert rag.index_documents(store, [Upload()]).chunks == 1
            reopened = rag.open_vectorstore(embeddings, sys.argv[1])
            assert rag.has_documents(reopened)
            assert rag.index_documents(reopened, [Upload()]).skipped_files == 1
            assert embeddings.calls == 1
            found = reopened.similarity_search('document', k=1)
            assert found[0].metadata['source'] == 'example.txt'
            keyless = rag.open_vectorstore(None, sys.argv[1])
            rag.reset_vectorstore(keyless)
            assert not rag.has_documents(keyless)
            assert not rag.has_documents(store)
            assert not rag.has_documents(reopened)
            assert rag.index_documents(store, [Upload()]).chunks == 1
            assert rag.has_documents(reopened)
            print('offline Chroma integration passed')
            """
        )
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-c", script, directory],
                cwd=Path(rag.__file__).parent,
                capture_output=True,
                text=True,
                timeout=90,
                env={**os.environ, "ANONYMIZED_TELEMETRY": "False"},
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
