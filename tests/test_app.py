"""Offline regression tests for Streamlit session and chat lifecycle behavior."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

import rag


APP_PATH = Path(__file__).resolve().parents[1] / "test.py"
SPEC = importlib.util.spec_from_file_location("rag_streamlit_app", APP_PATH)
app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(app)


class SessionClientTests(unittest.TestCase):
    def setUp(self):
        self.state = {}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(app.st, "session_state", self.state))

    def test_key_rotation_discards_clients_and_retains_conversation(self):
        app.sync_api_key("first-key")
        messages = [{"role": "user", "content": "earlier question"}]
        self.state.update(vectorstore=object(), llm=object(), llm_temperature=0.2, messages=messages)
        app.sync_api_key("second-key")
        self.assertEqual(self.state["client_key"], hashlib.sha256(b"second-key").hexdigest())
        self.assertIs(self.state["messages"], messages)
        self.assertNotIn("vectorstore", self.state)
        self.assertNotIn("llm", self.state)
        self.assertNotIn("llm_temperature", self.state)
        app.sync_api_key("")
        self.assertIsNone(self.state["client_key"])

    def test_clients_reused_until_key_or_temperature_changes(self):
        with patch.object(app, "create_embeddings") as embeddings, patch.object(
            app, "open_vectorstore"
        ) as open_store, patch.object(app, "create_llm") as create_llm:
            app.sync_api_key("first-key")
            first_store = app.get_vectorstore("first-key")
            first_llm = app.get_llm("first-key", 0.2)
            app.sync_api_key("first-key")
            self.assertIs(app.get_vectorstore("first-key"), first_store)
            self.assertIs(app.get_llm("first-key", 0.2), first_llm)
            embeddings.assert_called_once_with("first-key")
            open_store.assert_called_once_with(embeddings.return_value)
            create_llm.assert_called_once_with("first-key", 0.2)

            app.get_llm("first-key", 0.5)
            self.assertEqual(create_llm.call_count, 2)
            app.sync_api_key("second-key")
            app.get_vectorstore("second-key")
            app.get_llm("second-key", 0.5)
            embeddings.assert_called_with("second-key")
            create_llm.assert_called_with("second-key", 0.5)
            self.assertEqual(open_store.call_count, 2)

    def test_failed_vectorstore_initialization_can_retry(self):
        store = object()
        with patch.object(app, "create_embeddings"), patch.object(
            app, "open_vectorstore", side_effect=[RuntimeError("unavailable"), store]
        ) as open_store:
            with self.assertRaises(RuntimeError):
                app.get_vectorstore("test-key")
            self.assertNotIn("vectorstore", self.state)
            self.assertIs(app.get_vectorstore("test-key"), store)
            self.assertEqual(open_store.call_count, 2)

    def test_failed_llm_initialization_can_retry(self):
        llm = object()
        with patch.object(app, "create_llm", side_effect=[RuntimeError("unavailable"), llm]):
            with self.assertRaises(RuntimeError):
                app.get_llm("test-key", 0.2)
            self.assertNotIn("llm", self.state)
            self.assertNotIn("llm_temperature", self.state)
            self.assertIs(app.get_llm("test-key", 0.2), llm)
            self.assertEqual(self.state["llm_temperature"], 0.2)

    def test_session_credentials_do_not_mutate_process_environment(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "environment-key"}), patch.object(
            app, "create_embeddings"
        ), patch.object(app, "open_vectorstore"), patch.object(app, "create_llm"):
            original_environment = dict(os.environ)
            app.sync_api_key("session-key")
            app.get_vectorstore("session-key")
            app.get_llm("session-key", 0.2)
            app.sync_api_key("")
            self.assertEqual(dict(os.environ), original_environment)

    def test_empty_key_rejected_before_constructing_vectorstore(self):
        with patch.object(app, "create_embeddings") as embeddings:
            with self.assertRaises(ValueError):
                app.get_vectorstore("")
            embeddings.assert_not_called()


class StreamlitLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        temp_dir = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.db_path = Path(temp_dir) / "existing_index"
        self.db_path.mkdir()
        self.stack.enter_context(patch.dict(os.environ, {"OPENAI_API_KEY": ""}))
        self.stack.enter_context(patch("dotenv.load_dotenv"))
        self.stack.enter_context(patch.object(rag, "CHROMA_DB_PATH", str(self.db_path)))
        self.embeddings = self.stack.enter_context(patch.object(rag, "create_embeddings"))
        self.create_llm = self.stack.enter_context(patch.object(rag, "create_llm"))
        self.store = Mock()
        self.store.similarity_search.return_value = [object()]
        self.open_store = self.stack.enter_context(
            patch.object(rag, "open_vectorstore", return_value=self.store)
        )
        self.stack.enter_context(patch.object(rag, "has_documents", return_value=True))
        self.sources = [("example.txt", "Supporting document excerpt")]
        self.stack.enter_context(patch.object(rag, "format_sources", return_value=self.sources))
        self.stream_answer = self.stack.enter_context(
            patch.object(rag, "stream_answer", side_effect=lambda *args: iter(["Complete", " answer"]))
        )
        self.reset_store = self.stack.enter_context(patch.object(rag, "reset_vectorstore"))

    def run_app(self):
        result = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        self.assertEqual(len(result.exception), 0)
        return result

    def test_startup_without_key_does_not_create_clients(self):
        result = self.run_app()
        self.assertEqual(result.session_state["messages"], [])
        self.assertEqual(result.text_input[0].value, "")
        self.embeddings.assert_not_called()
        self.open_store.assert_not_called()
        self.create_llm.assert_not_called()

    def test_late_key_reloads_existing_db_and_changed_key_recreates_client(self):
        result = self.run_app()
        result.text_input[0].set_value("first-key").run()
        self.assertEqual(len(result.exception), 0)
        self.embeddings.assert_called_once_with("first-key")
        self.open_store.assert_called_once()
        result.run()
        self.assertEqual(self.open_store.call_count, 1)
        result.text_input[0].set_value("second-key").run()
        self.assertEqual(len(result.exception), 0)
        self.embeddings.assert_called_with("second-key")
        self.assertEqual(self.open_store.call_count, 2)
        self.assertEqual(os.environ["OPENAI_API_KEY"], "")

    def test_existing_db_load_failure_retries_on_next_run(self):
        self.open_store.side_effect = [RuntimeError("private-provider-error"), self.store]
        result = self.run_app()
        result.text_input[0].set_value("test-key").run()
        self.assertEqual(len(result.exception), 0)
        self.assertEqual(len(result.error), 1)
        self.assertNotIn("private-provider-error", result.error[0].value)
        self.assertNotIn("vectorstore", result.session_state.filtered_state)
        result.run()
        self.assertEqual(len(result.exception), 0)
        self.assertEqual(len(result.error), 0)
        self.assertEqual(self.open_store.call_count, 2)

    def test_complete_answer_persists_question_answer_and_sources(self):
        result = self.run_app()
        result.text_input[0].set_value("test-key").run()
        result.chat_input[0].set_value("  What does the document say?  ").run()
        self.assertEqual(len(result.exception), 0)
        self.assertEqual(result.session_state["messages"], [
            {"role": "user", "content": "What does the document say?"},
            {"role": "assistant", "content": "Complete answer", "sources": self.sources},
        ])
        self.store.similarity_search.assert_called_once_with("What does the document say?", k=8)
        self.create_llm.assert_called_once_with("test-key", 0.2)
        result.run()
        self.assertEqual(len(result.chat_message), 2)
        self.assertEqual(self.stream_answer.call_count, 1)

    def test_partial_answer_failure_does_not_save_turn_and_allows_retry(self):
        def failed_stream(*args):
            yield "Partial answer"
            raise RuntimeError("private-provider-error")

        self.stream_answer.side_effect = failed_stream
        result = self.run_app()
        result.text_input[0].set_value("test-key").run()
        result.chat_input[0].set_value("First attempt").run()
        self.assertEqual(len(result.exception), 0)
        self.assertEqual(result.session_state["messages"], [])
        self.assertEqual(len(result.error), 1)
        self.assertNotIn("private-provider-error", result.error[0].value)
        self.stream_answer.side_effect = lambda *args: iter(["Retry succeeded"])
        result.chat_input[0].set_value("Second attempt").run()
        self.assertEqual(len(result.exception), 0)
        self.assertEqual(result.session_state["messages"][0]["content"], "Second attempt")
        self.assertEqual(result.session_state["messages"][1]["content"], "Retry succeeded")
        self.assertEqual(len(result.session_state["messages"]), 2)

    def test_existing_db_can_be_reset_without_api_key(self):
        result = self.run_app()
        reset_button = next(button for button in result.button if button.label == "OpenAI DB 초기화")
        reset_button.click().run()
        self.assertEqual(len(result.exception), 0)
        self.open_store.assert_called_once_with(None)
        self.reset_store.assert_called_once_with(self.store)
        self.embeddings.assert_not_called()
        self.assertEqual(result.session_state["messages"], [])


if __name__ == "__main__":
    unittest.main()
