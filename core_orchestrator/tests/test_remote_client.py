"""
Tests for RemoteClientHelper — fixes for missing NumPy import, GET status, etc.
"""

from unittest.mock import MagicMock, patch

import numpy as np
from django.test import TestCase

from core_orchestrator.utils.model_host.remote_client_helper import (
    RemoteClientHelper,
    RemoteClientError,
    cv2_imdecode,
    base64_to_image,
)


class RemoteClientHelperNumpyTests(TestCase):
    """Prove the missing-numpy import bug is fixed."""

    def test_module_has_numpy_import(self):
        """Module-level numpy import is present."""
        import core_orchestrator.utils.model_host.remote_client_helper as rch
        self.assertIsNotNone(rch.np)

    def test_cv2_imdecode_has_numpy(self):
        """cv2_imdecode uses np.frombuffer without crashing."""
        result = cv2_imdecode(b"")
        # Without cv2 installed, returns None — but should not raise AttributeError
        # about missing 'np'
        self.assertIsNone(result)


class RemoteClientHelperMethodTests(TestCase):
    """Test HTTP method selection."""

    def test_get_status_uses_get(self):
        """get_status() should use GET, not POST."""
        helper = RemoteClientHelper(base_url="http://127.0.0.1:8080")

        with patch.object(helper._session, "get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            mock_get.return_value.json.return_value = {"status": "running"}

            helper.get_status()

            mock_get.assert_called_once_with(
                "http://127.0.0.1:8080/api/host/status/",
                timeout=60,
            )

    def test_evaluate_llm_uses_post(self):
        """evaluate_llm() should use POST."""
        helper = RemoteClientHelper(base_url="http://127.0.0.1:8080")

        with patch.object(helper._session, "post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            mock_post.return_value.json.return_value = {"content": "test"}

            helper.evaluate_llm(prompt="Hello")

            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            self.assertIn("/api/host/evaluate-llm/", args[0])
            self.assertEqual(kwargs["json"]["prompt"], "Hello")

    def test_evaluate_sam3_uses_post(self):
        """evaluate_sam3() should use POST."""
        helper = RemoteClientHelper(base_url="http://127.0.0.1:8080")

        with patch.object(helper._session, "post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            mock_post.return_value.json.return_value = {"masks": []}

            # Create a simple test image
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            result = helper.evaluate_sam3(frame)

            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            self.assertIn("/api/host/evaluate-sam3/", args[0])
            self.assertIn("frame_b64", kwargs["json"])

    def test_connection_check_calls_status(self):
        """check_connection() relies on get_status()."""
        helper = RemoteClientHelper(base_url="http://127.0.0.1:8080")

        with patch.object(helper, "get_status") as mock_status:
            mock_status.return_value = {"status": "running"}
            self.assertTrue(helper.check_connection())
            mock_status.assert_called_once()

    def test_connection_check_failure(self):
        """check_connection() returns False on error."""
        helper = RemoteClientHelper(base_url="http://127.0.0.1:8080")

        with patch.object(helper, "get_status") as mock_status:
            mock_status.side_effect = RemoteClientError("Not reachable")
            self.assertFalse(helper.check_connection())


class RemoteClientHelperResponseTests(TestCase):
    """Test response parsing."""

    def test_non_json_response(self):
        """Non-JSON response should raise RemoteClientError."""
        helper = RemoteClientHelper(base_url="http://127.0.0.1:8080")

        with patch.object(helper._session, "get") as mock_get:
            mock_response = MagicMock(status_code=200)
            mock_response.ok = True
            mock_response.json.side_effect = ValueError("No JSON")
            mock_get.return_value = mock_response

            with self.assertRaises(RemoteClientError):
                helper.get_status()

    def test_http_error_raises(self):
        """HTTP error status should raise RemoteClientError after retries."""
        import requests
        helper = RemoteClientHelper(base_url="http://127.0.0.1:8080")

        with patch.object(helper._session, "get") as mock_get:
            mock_response = MagicMock(status_code=500, ok=False)
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
            mock_get.return_value = mock_response

            with self.assertRaises(RemoteClientError):
                helper.get_status()


class RemoteClientHelperImageTests(TestCase):
    """Test image encoding/decoding."""

    def test_image_encoding_valid(self):
        """Valid image should encode without error."""
        helper = RemoteClientHelper(base_url="http://127.0.0.1:8080")
        frame = np.zeros((50, 50, 3), dtype=np.uint8)

        with patch.object(helper._session, "post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            mock_post.return_value.json.return_value = {"masks": []}

            result = helper.evaluate_sam3(frame)
            self.assertIsNotNone(result)

    def test_image_too_small(self):
        """Very small image should still work."""
        helper = RemoteClientHelper(base_url="http://127.0.0.1:8080")
        frame = np.ones((5, 5, 3), dtype=np.uint8)

        with patch.object(helper._session, "post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            mock_post.return_value.json.return_value = {"masks": []}

            result = helper.evaluate_sam3(frame)
            self.assertIsNotNone(result)

    def test_invalid_base64_decoding(self):
        """Invalid base64 input should return None."""
        result = base64_to_image("not-valid-base64!!!")
        self.assertIsNone(result)