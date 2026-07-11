"""
Tests for independent LLM/SAM routing and backward-compatible legacy routing_mode.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.test import TestCase, RequestFactory
from django.http import JsonResponse

from core_orchestrator.services.settings_store import (
    SettingsStore,
    _apply_legacy_migration,
    _dataclass_to_dict,
    AppSettings,
)


class RoutingModeBackwardCompatTests(TestCase):
    """Test that legacy routing_mode is migrated correctly."""

    def test_legacy_mode_migration_local(self):
        """Legacy routing_mode='local' becomes llm_mode='local', sam3_mode='local'."""
        old = {"routing_mode": "local", "version": 0}
        migrated = _apply_legacy_migration(old)
        self.assertEqual(migrated["client"]["llm_mode"], "local")
        self.assertEqual(migrated["client"]["sam3_mode"], "local")

    def test_legacy_mode_migration_remote(self):
        """Legacy routing_mode='remote' becomes llm_mode='remote', sam3_mode='remote'."""
        old = {"routing_mode": "remote", "version": 0}
        migrated = _apply_legacy_migration(old)
        self.assertEqual(migrated["client"]["llm_mode"], "remote")
        self.assertEqual(migrated["client"]["sam3_mode"], "remote")

    def test_independent_routing_survives_roundtrip(self):
        """llm_mode=local and sam3_mode=remote survive save/load."""
        tmpdir = Path(tempfile.mkdtemp())
        try:
            store = SettingsStore(directory=tmpdir)
            settings = store.load()
            settings.client.llm_mode = "local"
            settings.client.sam3_mode = "remote"
            store.save(settings)

            store2 = SettingsStore(directory=tmpdir)
            loaded = store2.load()
            self.assertEqual(loaded.client.llm_mode, "local")
            self.assertEqual(loaded.client.sam3_mode, "remote")
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


class SettingsAPIViewTests(TestCase):
    """Test the settings GET/PUT API endpoints."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.env_patch = patch.dict(os.environ, {"ALMOST_ARCADIA_CONFIG_DIR": str(self.tmpdir)})
        self.env_patch.start()
        self.factory = RequestFactory()

    def tearDown(self):
        self.env_patch.stop()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_settings_get_returns_json(self):
        """GET /api/settings/ returns settings JSON."""
        from core_orchestrator.views.settings_api import settings_view

        request = self.factory.get("/api/settings/")
        response = settings_view(request)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIn("version", data)
        self.assertIn("client", data)
        self.assertIn("host", data)

    def test_settings_put_updates(self):
        """PUT /api/settings/ updates and returns settings."""
        from core_orchestrator.views.settings_api import settings_view

        request = self.factory.put(
            "/api/settings/",
            data=json.dumps({"client": {"llm_mode": "remote"}}),
            content_type="application/json",
        )
        response = settings_view(request)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data["client"]["llm_mode"], "remote")

    def test_settings_put_invalid_json(self):
        """PUT with invalid JSON returns error."""
        from core_orchestrator.views.settings_api import settings_view

        request = self.factory.put(
            "/api/settings/",
            data="not json",
            content_type="application/json",
        )
        response = settings_view(request)
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.content)
        self.assertIn("error", data)

    def test_settings_put_non_object(self):
        """PUT with non-object JSON returns error."""
        from core_orchestrator.views.settings_api import settings_view

        request = self.factory.put(
            "/api/settings/",
            data=json.dumps([1, 2, 3]),
            content_type="application/json",
        )
        response = settings_view(request)
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.content)
        self.assertIn("error", data)

    def test_settings_reset_returns_defaults(self):
        """POST /api/settings/reset/ returns defaults."""
        from core_orchestrator.views.settings_api import settings_reset

        request = self.factory.post("/api/settings/reset/")
        response = settings_reset(request)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data["client"]["llm_mode"], "local")
        self.assertEqual(data["client"]["sam3_mode"], "local")


class HostStatusViewTests(TestCase):
    """Test the Host status view returns structured data."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.env_patch = patch.dict(os.environ, {"ALMOST_ARCADIA_CONFIG_DIR": str(self.tmpdir)})
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_host_status_returns_json_with_service_state(self):
        """host_status returns structured JSON with services."""
        from core_orchestrator.views.pages import host_status
        from django.http import HttpRequest

        request = HttpRequest()
        request.method = "GET"
        response = host_status(request)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIn("status", data)
        self.assertIn("llm", data)
        self.assertIn("sam3", data)


class HeatmapDashboardViewTests(TestCase):
    """Test heatmap dashboard renders with config."""

    def setUp(self):
        self.factory = RequestFactory()

    def test_dashboard_renders(self):
        """heatmap_dashboard returns 200."""
        from core_orchestrator.views.pages import heatmap_dashboard

        request = self.factory.get('/client/heatmap/')
        response = heatmap_dashboard(request)
        self.assertEqual(response.status_code, 200)

class IndependentRoutingTests(TestCase):
    """Test all four routing combinations work."""

    def test_local_llm_local_sam(self):
        """llm_mode=local, sam3_mode=local."""
        settings = AppSettings()
        settings.client.llm_mode = "local"
        settings.client.sam3_mode = "local"
        d = _dataclass_to_dict(settings)
        self.assertEqual(d["client"]["llm_mode"], "local")
        self.assertEqual(d["client"]["sam3_mode"], "local")

    def test_local_llm_remote_sam(self):
        """llm_mode=local, sam3_mode=remote."""
        settings = AppSettings()
        settings.client.llm_mode = "local"
        settings.client.sam3_mode = "remote"
        d = _dataclass_to_dict(settings)
        self.assertEqual(d["client"]["llm_mode"], "local")
        self.assertEqual(d["client"]["sam3_mode"], "remote")

    def test_remote_llm_local_sam(self):
        """llm_mode=remote, sam3_mode=local."""
        settings = AppSettings()
        settings.client.llm_mode = "remote"
        settings.client.sam3_mode = "local"
        d = _dataclass_to_dict(settings)
        self.assertEqual(d["client"]["llm_mode"], "remote")
        self.assertEqual(d["client"]["sam3_mode"], "local")

    def test_remote_llm_remote_sam(self):
        """llm_mode=remote, sam3_mode=remote."""
        settings = AppSettings()
        settings.client.llm_mode = "remote"
        settings.client.sam3_mode = "remote"
        d = _dataclass_to_dict(settings)
        self.assertEqual(d["client"]["llm_mode"], "remote")
        self.assertEqual(d["client"]["sam3_mode"], "remote")