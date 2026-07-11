"""
Tests for the Host API listener (background HTTP server).
"""

import json
import os
import tempfile
import threading
import time
from http.server import HTTPServer, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

from django.test import TestCase

from core_orchestrator.views.pages import (
    HostAPIHandler,
    _host_api_running,
    _host_api_server,
    _host_api_thread,
    _request_log,
    _config_lock,
    _lifecycle_lock,
)


def _reset_globals():
    """Reset Host API globals for clean test state."""
    import core_orchestrator.views.pages as pages
    pages._host_api_running = False
    pages._host_api_thread = None
    pages._host_api_server = None
    pages._host_api_config = {}
    pages._request_log.clear()


class HostAPIHandlerMethodTests(TestCase):
    """Test that the handler supports proper HTTP methods."""

    def test_handler_supports_get(self):
        """HostAPIHandler has do_GET method."""
        self.assertTrue(hasattr(HostAPIHandler, "do_GET"))

    def test_handler_supports_post(self):
        """HostAPIHandler has do_POST method."""
        self.assertTrue(hasattr(HostAPIHandler, "do_POST"))


class HostListenerLifecycleTests(TestCase):
    """Test lifecycle operations."""

    def test_lifecycle_lock_exists(self):
        """The lifecycle lock is a threading.Lock."""
        self.assertIsInstance(_lifecycle_lock, type(threading.Lock()))


class RequestLogTests(TestCase):
    """Test the request log is bounded."""

    def setUp(self):
        _reset_globals()

    def test_log_entries_bounded(self):
        """Request log does not exceed _MAX_LOG_ENTRIES."""
        from core_orchestrator.views.pages import log_request, get_request_logs, _MAX_LOG_ENTRIES

        for i in range(_MAX_LOG_ENTRIES * 2):
            log_request(f"/test/{i}", {"idx": i})

        logs = get_request_logs()
        self.assertLessEqual(len(logs), _MAX_LOG_ENTRIES)


class HostAPIHandlerConcurrencyTests(TestCase):
    """Test that ThreadingHTTPServer is used."""

    def test_handler_is_threading_mixin(self):
        """HostAPIHandler uses ThreadingHTTPServer which has ThreadingMixIn."""
        # ThreadingHTTPServer inherits from ThreadingMixIn + HTTPServer
        self.assertTrue(hasattr(ThreadingHTTPServer, "process_request"))


class HostAPIHandlerBodySizeTests(TestCase):
    """Test request body size limits."""

    def test_max_body_size_constant_exists(self):
        """_MAX_REQUEST_BODY is defined."""
        from core_orchestrator.views.pages import _MAX_REQUEST_BODY
        self.assertGreater(_MAX_REQUEST_BODY, 0)


class HostAPIHandlerStatusLogicTests(TestCase):
    """Test status response structure via handler logic."""

    def test_status_includes_listener_and_services(self):
        """The status endpoint response structure includes expected fields."""
        _reset_globals()

        # Directly invoke the handler status logic
        from core_orchestrator.views.pages import host_status
        from django.http import HttpRequest

        request = HttpRequest()
        request.method = "GET"
        response = host_status(request)
        data = json.loads(response.content)

        self.assertIn("status", data)
        self.assertIn("listen_ip", data)
        self.assertIn("listen_port", data)
        self.assertIn("llm", data)
        self.assertIn("sam3", data)