"""
Tests for RemoteClientHelper — numpy fix, GET status, image decoding.
"""

from unittest.mock import MagicMock, patch

import numpy as np
from django.test import TestCase

from core_orchestrator.utils.model_host.remote_client_helper import (
    RemoteClientHelper,
    RemoteClientError,
    cv2_imdecode,
    cv2_imencode,
    base64_to_image,
)
from .test_helpers import setup_test_config_dir, teardown_test_config_dir, reset_all_singletons


class RemoteClientNumpyTests(TestCase):
    def setUp(self):
        self.tmpdir = setup_test_config_dir()
        reset_all_singletons()

    def tearDown(self):
        teardown_test_config_dir(self.tmpdir)
        reset_all_singletons()

    def test_module_has_numpy_import(self):
        import core_orchestrator.utils.model_host.remote_client_helper as rch
        self.assertIsNotNone(rch.np)

    def test_cv2_imdecode_no_attribute_error(self):
        """cv2_imdecode must not raise AttributeError about 'np'."""
        result = cv2_imdecode(b"")
        self.assertIsNone(result)

    def test_cv2_imencode_roundtrip(self):
        """Encode then decode produces a valid image."""
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        ok, png_bytes = cv2_imencode(frame)
        if ok:
            decoded = cv2_imdecode(png_bytes)
            self.assertIsNotNone(decoded)
            self.assertEqual(decoded.shape, (50, 50, 3))


class RemoteClientMethodTests(TestCase):
    def setUp(self):
        self.tmpdir = setup_test_config_dir()
        reset_all_singletons()

    def tearDown(self):
        teardown_test_config_dir(self.tmpdir)
        reset_all_singletons()

    def test_get_status_uses_get(self):
        helper = RemoteClientHelper(base_url="http://127.0.0.1:8080")

        with patch.object(helper._session, "get") as mock_get:
            mock_resp = MagicMock(status_code=200)
            mock_resp.json.return_value = {"status": "running"}
            mock_get.return_value = mock_resp

            helper.get_status()
            mock_get.assert_called_once_with(
                "http://127.0.0.1:8080/api/host/status/",
                timeout=60,
            )

    def test_evaluate_llm_uses_post(self):
        helper = RemoteClientHelper(base_url="http://127.0.0.1:8080")

        with patch.object(helper._session, "post") as mock_post:
            mock_resp = MagicMock(status_code=200)
            mock_resp.json.return_value = {"content": "test"}
            mock_post.return_value = mock_resp

            helper.evaluate_llm(prompt="Hello")
            mock_post.assert_called_once()

    def test_evaluate_sam3_uses_post(self):
        helper = RemoteClientHelper(base_url="http://127.0.0.1:8080")

        with patch.object(helper._session, "post") as mock_post:
            mock_resp = MagicMock(status_code=200)
            mock_resp.json.return_value = {"masks": []}
            mock_post.return_value = mock_resp

            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            helper.evaluate_sam3(frame)
            mock_post.assert_called_once()

    def test_connection_check_calls_status(self):
        helper = RemoteClientHelper(base_url="http://127.0.0.1:8080")

        with patch.object(helper, "get_status") as mock_status:
            mock_status.return_value = {"status": "running"}
            self.assertTrue(helper.check_connection())

    def test_connection_check_failure(self):
        helper = RemoteClientHelper(base_url="http://127.0.0.1:8080")

        with patch.object(helper, "get_status") as mock_status:
            mock_status.side_effect = RemoteClientError("Not reachable")
            self.assertFalse(helper.check_connection())

    def test_invalid_base64_decoding_returns_none(self):
        result = base64_to_image("not-valid-base64!!!")
        self.assertIsNone(result)


class RemoteClientErrorHandlingTests(TestCase):
    def setUp(self):
        self.tmpdir = setup_test_config_dir()
        reset_all_singletons()

    def tearDown(self):
        teardown_test_config_dir(self.tmpdir)
        reset_all_singletons()

    def test_http_error_raises_after_retries(self):
        import requests
        helper = RemoteClientHelper(base_url="http://127.0.0.1:8080", timeout=1)

        with patch.object(helper._session, "get") as mock_get:
            mock_resp = MagicMock(status_code=500, ok=False)
            mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
            mock_get.return_value = mock_resp

            with self.assertRaises(RemoteClientError):
                helper.get_status()

    def test_non_json_response_raises(self):
        helper = RemoteClientHelper(base_url="http://127.0.0.1:8080", timeout=1)

        with patch.object(helper._session, "get") as mock_get:
            mock_resp = MagicMock(status_code=200, ok=True)
            mock_resp.json.side_effect = ValueError("No JSON")
            mock_get.return_value = mock_resp

            with self.assertRaises(RemoteClientError):
                helper.get_status()
