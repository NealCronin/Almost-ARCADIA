"""
Tests for LLM inference client adapters — mock HTTP responses for each format.
"""

from unittest.mock import MagicMock

from django.test import TestCase

from core_orchestrator.services.llm_client import (
    LLMInferenceClient,
    LLMResult,
    LLMInferenceError,
)
from core_orchestrator.services.settings_store import LLMServiceSettings
from .test_helpers import setup_test_config_dir, teardown_test_config_dir, reset_all_singletons


class LLMClientBaseTests(TestCase):
    def setUp(self):
        self.tmpdir = setup_test_config_dir()
        reset_all_singletons()
        self.client = LLMInferenceClient()
        self.mock_session = MagicMock()
        self.client._session = self.mock_session

    def tearDown(self):
        teardown_test_config_dir(self.tmpdir)
        reset_all_singletons()


class LlamaCompletionTests(LLMClientBaseTests):
    def test_llama_completion_correct_endpoint_and_payload(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"content": "Hello world"}
        self.mock_session.post.return_value = mock_resp

        cfg = LLMServiceSettings(
            service_mode="managed",
            base_url="http://127.0.0.1:8081",
            api_format="llama-completion",
            model_id="local",
            request_timeout_seconds=30,
        )
        result = self.client.evaluate(cfg, "test prompt")

        self.assertEqual(result.content, "Hello world")
        self.mock_session.post.assert_called_once()
        args, kwargs = self.mock_session.post.call_args
        self.assertIn("/completion", args[0])
        self.assertEqual(kwargs["json"]["prompt"], "test prompt")
        self.assertEqual(kwargs["json"]["n_predict"], 512)

    def test_llama_completion_extracts_generation_field(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"generation": "alt field"}
        self.mock_session.post.return_value = mock_resp

        cfg = LLMServiceSettings(base_url="http://localhost:8081", api_format="llama-completion")
        result = self.client.evaluate(cfg, "prompt")
        self.assertEqual(result.content, "alt field")


class OpenAIChatTests(LLMClientBaseTests):
    def test_openai_chat_correct_endpoint_and_payload(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Chat response"}}],
            "usage": {"tokens": 42},
        }
        self.mock_session.post.return_value = mock_resp

        cfg = LLMServiceSettings(
            base_url="http://127.0.0.1:40000/v1",
            api_format="openai-chat",
            model_id="gpt-4-local",
            request_timeout_seconds=30,
        )
        result = self.client.evaluate(cfg, "Hello", context="Some context")

        self.assertEqual(result.content, "Chat response")
        self.assertEqual(result.model_id, "gpt-4-local")
        self.mock_session.post.assert_called_once()
        args, kwargs = self.mock_session.post.call_args
        self.assertIn("/chat/completions", args[0])
        self.assertEqual(kwargs["json"]["model"], "gpt-4-local")
        self.assertEqual(kwargs["json"]["messages"][0]["role"], "system")
        self.assertEqual(kwargs["json"]["messages"][1]["role"], "user")

    def test_openai_chat_requires_model_id(self):
        cfg = LLMServiceSettings(
            base_url="http://localhost:8081",
            api_format="openai-chat",
            model_id="",
        )
        with self.assertRaises(LLMInferenceError):
            self.client.evaluate(cfg, "prompt")

    def test_openai_chat_5xx_raises_error(self):
        import requests
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        self.mock_session.post.return_value = mock_resp
        cfg = LLMServiceSettings(base_url="http://localhost:8081", api_format="openai-chat", model_id="test")
        with self.assertRaises(LLMInferenceError):
            self.client.evaluate(cfg, "prompt")


class OpenAIResponsesTests(LLMClientBaseTests):
    def test_openai_responses_correct_endpoint(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "Response text"}]}]
        }
        self.mock_session.post.return_value = mock_resp

        cfg = LLMServiceSettings(
            base_url="http://127.0.0.1:40000/v1",
            api_format="openai-responses",
            model_id="model-id",
            request_timeout_seconds=30,
        )
        result = self.client.evaluate(cfg, "test")
        self.assertEqual(result.content, "Response text")

    def test_openai_responses_timeout_raises(self):
        import requests
        self.mock_session.post.side_effect = requests.Timeout("Timeout")

        cfg = LLMServiceSettings(base_url="http://localhost:8081", api_format="openai-responses", model_id="m")
        with self.assertRaises(LLMInferenceError):
            self.client.evaluate(cfg, "prompt")


class LLMClientHealthCheckTests(LLMClientBaseTests):
    def test_health_200_means_healthy(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        self.mock_session.get.return_value = mock_resp

        cfg = LLMServiceSettings(base_url="http://localhost:8081", api_format="llama-completion")
        self.assertTrue(self.client.health_check(cfg))

    def test_health_400_not_healthy(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        self.mock_session.get.return_value = mock_resp

        cfg = LLMServiceSettings(base_url="http://localhost:8081", api_format="llama-completion")
        self.assertFalse(self.client.health_check(cfg))

    def test_health_500_not_healthy(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        self.mock_session.get.return_value = mock_resp

        cfg = LLMServiceSettings(base_url="http://localhost:8081", api_format="llama-completion")
        self.assertFalse(self.client.health_check(cfg))


class LLMClientResultSchemaTests(LLMClientBaseTests):
    def test_result_has_consistent_schema(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"content": "test"}
        self.mock_session.post.return_value = mock_resp

        cfg = LLMServiceSettings(base_url="http://localhost:8081", api_format="llama-completion")
        result = self.client.evaluate(cfg, "prompt")

        self.assertTrue(hasattr(result, "content"))
        self.assertTrue(hasattr(result, "model_id"))
        self.assertTrue(hasattr(result, "usage"))
        self.assertTrue(hasattr(result, "metadata"))
        self.assertEqual(result.content, "test")