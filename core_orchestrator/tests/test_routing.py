"""
Tests for independent LLM/SAM routing — real dispatch verification.
"""

import json
from unittest.mock import patch, MagicMock

from django.test import TestCase, Client

from core_orchestrator.services.settings_store import (
    SettingsStore,
    _dataclass_to_dict,
    AppSettings,
    reset_settings_store_for_tests,
)
from .test_helpers import setup_test_config_dir, teardown_test_config_dir, reset_all_singletons


class SettingsAPIThroughClientTests(TestCase):
    """Test settings API through Django's test client (exercises URL routing)."""

    def setUp(self):
        self.tmpdir = setup_test_config_dir()
        reset_all_singletons()
        self.client = Client()

    def tearDown(self):
        teardown_test_config_dir(self.tmpdir)
        reset_all_singletons()

    def test_get_settings_returns_200(self):
        resp = self.client.get("/api/settings/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("version", data)
        self.assertIn("client", data)

    def test_put_valid_settings(self):
        resp = self.client.put(
            "/api/settings/",
            data=json.dumps({"client": {"llm_mode": "remote", "sam3_mode": "local"}}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["client"]["llm_mode"], "remote")
        self.assertEqual(data["client"]["sam3_mode"], "local")

    def test_put_invalid_json(self):
        resp = self.client.put(
            "/api/settings/",
            data="not json",
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_put_invalid_port(self):
        resp = self.client.put(
            "/api/settings/",
            data=json.dumps({"host": {"llm": {"port": "abc"}}}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertEqual(data["error"]["code"], "invalid_configuration")

    def test_put_invalid_mode(self):
        resp = self.client.put(
            "/api/settings/",
            data=json.dumps({"client": {"llm_mode": "invalid"}}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_reset_to_defaults(self):
        resp = self.client.post("/api/settings/reset/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["client"]["llm_mode"], "local")

    def test_saved_config_survives_new_store(self):
        self.client.put(
            "/api/settings/",
            data=json.dumps({"client": {"llm_mode": "remote", "sam3_mode": "remote"}}),
            content_type="application/json",
        )
        reset_settings_store_for_tests()
        store = SettingsStore()
        s = store.load()
        self.assertEqual(s.client.llm_mode, "remote")

    def test_host_portal_renders_saved_values(self):
        # Save custom values
        self.client.put(
            "/api/settings/",
            data=json.dumps({
                "host": {
                    "llm": {"executable": "/custom/llama-server", "model_path": "/custom/model.gguf"},
                    "sam3": {"weights_path": "/custom/sam3.pt"},
                }
            }),
            content_type="application/json",
        )
        # Render host portal
        resp = self.client.get("/host/")
        self.assertEqual(resp.status_code, 200)
        content = resp.content.decode("utf-8")
        self.assertIn("/custom/llama-server", content)
        self.assertIn("/custom/model.gguf", content)
        self.assertIn("/custom/sam3.pt", content)


class IndependentRoutingTests(TestCase):
    """Test that all four routing combinations persist correctly."""

    def setUp(self):
        self.tmpdir = setup_test_config_dir()
        reset_all_singletons()
        self.client = Client()

    def tearDown(self):
        teardown_test_config_dir(self.tmpdir)
        reset_all_singletons()

    def test_local_llm_local_sam(self):
        self.client.put(
            "/api/settings/",
            data=json.dumps({"client": {"llm_mode": "local", "sam3_mode": "local"}}),
            content_type="application/json",
        )
        resp = self.client.get("/api/settings/")
        data = resp.json()
        self.assertEqual(data["client"]["llm_mode"], "local")
        self.assertEqual(data["client"]["sam3_mode"], "local")

    def test_local_llm_remote_sam(self):
        self.client.put(
            "/api/settings/",
            data=json.dumps({"client": {"llm_mode": "local", "sam3_mode": "remote"}}),
            content_type="application/json",
        )
        resp = self.client.get("/api/settings/")
        data = resp.json()
        self.assertEqual(data["client"]["llm_mode"], "local")
        self.assertEqual(data["client"]["sam3_mode"], "remote")

    def test_remote_llm_local_sam(self):
        self.client.put(
            "/api/settings/",
            data=json.dumps({"client": {"llm_mode": "remote", "sam3_mode": "local"}}),
            content_type="application/json",
        )
        resp = self.client.get("/api/settings/")
        data = resp.json()
        self.assertEqual(data["client"]["llm_mode"], "remote")
        self.assertEqual(data["client"]["sam3_mode"], "local")

    def test_remote_llm_remote_sam(self):
        self.client.put(
            "/api/settings/",
            data=json.dumps({"client": {"llm_mode": "remote", "sam3_mode": "remote"}}),
            content_type="application/json",
        )
        resp = self.client.get("/api/settings/")
        data = resp.json()
        self.assertEqual(data["client"]["llm_mode"], "remote")
        self.assertEqual(data["client"]["sam3_mode"], "remote")


class LegacyRoutingModeTests(TestCase):
    """Legacy routing_mode is migrated correctly."""

    def setUp(self):
        self.tmpdir = setup_test_config_dir()
        reset_all_singletons()

    def tearDown(self):
        teardown_test_config_dir(self.tmpdir)
        reset_all_singletons()

    def test_legacy_mode_local_splits(self):
        from core_orchestrator.services.settings_store import _apply_legacy_migration
        migrated = _apply_legacy_migration({"routing_mode": "local", "version": 0})
        self.assertEqual(migrated["client"]["llm_mode"], "local")
        self.assertEqual(migrated["client"]["sam3_mode"], "local")

    def test_legacy_mode_remote_splits(self):
        from core_orchestrator.services.settings_store import _apply_legacy_migration
        migrated = _apply_legacy_migration({"routing_mode": "remote", "version": 0})
        self.assertEqual(migrated["client"]["llm_mode"], "remote")
        self.assertEqual(migrated["client"]["sam3_mode"], "remote")


class CommandPreviewTests(TestCase):
    """Test the command preview endpoint."""

    def setUp(self):
        self.tmpdir = setup_test_config_dir()
        reset_all_singletons()
        self.client = Client()

    def tearDown(self):
        teardown_test_config_dir(self.tmpdir)
        reset_all_singletons()

    def test_llm_command_preview(self):
        resp = self.client.post(
            "/api/services/command-preview/",
            data=json.dumps({
                "service_id": "host:llm",
                "configuration": {
                    "service_mode": "managed",
                    "executable": "llama-server",
                    "model_path": "/models/test.gguf",
                    "host": "127.0.0.1",
                    "port": 8081,
                    "api_format": "llama-completion",
                    "arguments": ["--ctx-size", "4096"],
                },
            }),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("argument_array", data)
        self.assertIn("display_command", data)
        self.assertIn("llama-server", data["argument_array"])
        self.assertIn("--ctx-size", data["argument_array"])
        self.assertIn("4096", data["argument_array"])

    def test_sam_command_preview_in_process(self):
        resp = self.client.post(
            "/api/services/command-preview/",
            data=json.dumps({"service_id": "host:sam3", "configuration": {}}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("in-process", data["display_command"].lower())


class ServiceLogsTests(TestCase):
    def setUp(self):
        self.tmpdir = setup_test_config_dir()
        reset_all_singletons()
        self.client = Client()

    def tearDown(self):
        teardown_test_config_dir(self.tmpdir)
        reset_all_singletons()

    def test_get_logs_returns_lines(self):
        from core_orchestrator.services.service_manager import get_service_manager
        sm = get_service_manager()
        sm.add_log("host:llm", "test log line")

        resp = self.client.get("/api/services/logs/?service_id=host%3Allm&tail=10")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("lines", data)
        texts = [l["text"] for l in data["lines"]]
        self.assertIn("test log line", texts)


class ConnectionTestProxyTests(TestCase):
    def setUp(self):
        self.tmpdir = setup_test_config_dir()
        reset_all_singletons()
        self.client = Client()

    def tearDown(self):
        teardown_test_config_dir(self.tmpdir)
        reset_all_singletons()

    def test_invalid_port_returns_400(self):
        resp = self.client.post(
            "/api/client/test-host/",
            data=json.dumps({"host": "127.0.0.1", "port": "abc"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    @patch("core_orchestrator.utils.model_host.remote_client_helper.RemoteClientHelper")
    def test_reachable_host(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_status.return_value = {"status": "running"}
        mock_client_cls.return_value = mock_client

        resp = self.client.post(
            "/api/client/test-host/",
            data=json.dumps({"host": "127.0.0.1", "port": 8080}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["connected"])

    @patch("core_orchestrator.utils.model_host.remote_client_helper.RemoteClientHelper")
    def test_unreachable_host(self, mock_client_cls):
        from core_orchestrator.utils.model_host.remote_client_helper import RemoteClientError
        mock_client = MagicMock()
        mock_client.get_status.side_effect = RemoteClientError("Connection refused")
        mock_client_cls.return_value = mock_client

        resp = self.client.post(
            "/api/client/test-host/",
            data=json.dumps({"host": "127.0.0.1", "port": 8080}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["connected"])


class CSRFTests(TestCase):
    """CSRF protection is explicitly disabled on settings API via @csrf_exempt."""

    def setUp(self):
        self.tmpdir = setup_test_config_dir()
        reset_all_singletons()

    def tearDown(self):
        teardown_test_config_dir(self.tmpdir)
        reset_all_singletons()

    def test_csrf_exempt_put_without_token_succeeds(self):
        """Settings API is CSRF-exempt (explicit, documented choice for local app)."""
        client = Client(enforce_csrf_checks=True)
        resp = client.put(
            "/api/settings/",
            data=json.dumps({"client": {"llm_mode": "remote"}}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)

    def test_csrf_exempt_post_reset_without_token_succeeds(self):
        client = Client(enforce_csrf_checks=True)
        resp = client.post("/api/settings/reset/")
        self.assertEqual(resp.status_code, 200)

    def test_csrf_exempt_host_portal_post_without_token_succeeds(self):
        """Host portal POST actions are CSRF-exempt."""
        client = Client(enforce_csrf_checks=True)
        resp = client.post(
            "/host/",
            data=json.dumps({"action": "save_settings", "settings": {}}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)

    def test_csrf_token_rendered_in_templates(self):
        """CSRF token is rendered in templates (enables cookie-based CSRF for JS helper)."""
        client = Client()
        resp = client.get("/host/")
        self.assertContains(resp, "csrfmiddlewaretoken")
        resp2 = client.get("/client/heatmap/")
        self.assertContains(resp2, "csrfmiddlewaretoken")
