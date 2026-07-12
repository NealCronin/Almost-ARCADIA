"""
Tests for the Host API listener — sends real HTTP requests to a running server.
"""

import json
import threading
import time
from http.server import ThreadingHTTPServer
from unittest.mock import patch, MagicMock

from django.test import TestCase

from core_orchestrator.views.pages import HostAPIHandler
from .test_helpers import setup_test_config_dir, teardown_test_config_dir, reset_all_singletons


class HostListenerRealServerTests(TestCase):
    """Tests that bind a real ThreadingHTTPServer and send real HTTP requests."""

    def setUp(self):
        self.tmpdir = setup_test_config_dir()
        reset_all_singletons()
        self.server = None
        self.thread = None

    def tearDown(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.thread:
            self.thread.join(timeout=5)
        teardown_test_config_dir(self.tmpdir)
        reset_all_singletons()

    def _start_server(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), HostAPIHandler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        time.sleep(0.1)

    def test_get_status_returns_200(self):
        self._start_server()
        import requests
        resp = requests.get(f"http://127.0.0.1:{self.port}/api/host/status/", timeout=5)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("status", data)
        self.assertIn("llm", data)
        self.assertIn("sam3", data)

    def test_unknown_endpoint_returns_404(self):
        self._start_server()
        import requests
        resp = requests.get(f"http://127.0.0.1:{self.port}/api/unknown/", timeout=5)
        self.assertEqual(resp.status_code, 404)
        data = resp.json()
        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], "endpoint_not_found")

    def test_post_invalid_json_returns_400(self):
        self._start_server()
        import requests
        resp = requests.post(
            f"http://127.0.0.1:{self.port}/api/host/evaluate-llm/",
            data="not json",
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertEqual(data["error"]["code"], "invalid_json")

    def test_post_non_object_json_returns_400(self):
        self._start_server()
        import requests
        resp = requests.post(
            f"http://127.0.0.1:{self.port}/api/host/evaluate-llm/",
            json=[1, 2, 3],
            timeout=5,
        )
        self.assertEqual(resp.status_code, 400)

    def test_post_missing_prompt_returns_400(self):
        self._start_server()
        import requests
        resp = requests.post(
            f"http://127.0.0.1:{self.port}/api/host/evaluate-llm/",
            json={},
            timeout=5,
        )
        self.assertEqual(resp.status_code, 400)

    def test_post_eval_llm_service_not_running(self):
        """When no LLM service is running, returns 503."""
        self._start_server()
        import requests
        resp = requests.post(
            f"http://127.0.0.1:{self.port}/api/host/evaluate-llm/",
            json={"prompt": "test"},
            timeout=5,
        )
        self.assertEqual(resp.status_code, 503)
        data = resp.json()
        self.assertEqual(data["error"]["code"], "service_not_running")

    def test_oversized_content_length_returns_413(self):
        """Server rejects requests with Content-Length > MAX."""
        self._start_server()
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request(
                "POST",
                "/api/host/evaluate-llm/",
                body=b'{"prompt":"x"}',
                headers={"Content-Length": "999999999", "Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            body = resp.read()
            self.assertIn(resp.status, (400, 413),
                          f"Expected 400 or 413, got {resp.status}: {body.decode(errors='replace')[:200]}")
        finally:
            conn.close()

    def test_status_works_during_concurrent_request(self):
        """Status should work even while another request is in progress."""
        self._start_server()
        import requests
        results = []

        def get_status():
            try:
                resp = requests.get(f"http://127.0.0.1:{self.port}/api/host/status/", timeout=5)
                results.append(resp.status_code)
            except Exception as e:
                results.append(str(e))

        threads = [threading.Thread(target=get_status) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertTrue(all(r == 200 for r in results), f"Results: {results}")


class HostListenerDuplicateStartTests(TestCase):
    def setUp(self):
        self.tmpdir = setup_test_config_dir()
        reset_all_singletons()

    def tearDown(self):
        teardown_test_config_dir(self.tmpdir)
        reset_all_singletons()

    def test_duplicate_start_returns_already_running(self):
        """Starting a listener when one is running returns 409."""
        import core_orchestrator.views.pages as pages
        from core_orchestrator.views.pages import _start_listener
        from core_orchestrator.services.settings_store import AppSettings

        pages._host_api_running = True
        pages._host_api_server = object()
        try:
            result = _start_listener({"listen_ip": "127.0.0.1", "listen_port": 9999}, AppSettings())
            content = json.loads(result.content)
            self.assertEqual(result.status_code, 409)
            self.assertEqual(content["error"]["code"], "already_running")
        finally:
            pages._host_api_running = False
            pages._host_api_server = None
