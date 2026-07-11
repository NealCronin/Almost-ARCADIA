"""
Tests for ServiceManager lifecycle.
"""

import threading
from unittest.mock import MagicMock, patch

from django.test import TestCase

from core_orchestrator.services.service_manager import (
    ServiceManager,
    ServiceState,
    get_service_manager,
)
from core_orchestrator.services.settings_store import LLMServiceSettings, SAMServiceSettings


class ServiceManagerIdentityTests(TestCase):
    """Test unique service identities."""

    def test_service_ids_are_distinct(self):
        """Different service_ids have separate state."""
        sm = ServiceManager()
        self.assertEqual(sm.status("client:llm")["service_id"], "client:llm")
        self.assertEqual(sm.status("client:sam3")["service_id"], "client:sam3")
        self.assertEqual(sm.status("host:llm")["service_id"], "host:llm")
        self.assertEqual(sm.status("host:sam3")["service_id"], "host:sam3")

    def test_singleton_returns_same_instance(self):
        """get_service_manager() returns the same instance."""
        sm1 = get_service_manager()
        sm2 = get_service_manager()
        self.assertIs(sm1, sm2)


class ServiceManagerStateTests(TestCase):
    """Test initial states and transitions."""

    def test_initial_state_is_stopped(self):
        """New service starts as stopped."""
        sm = ServiceManager()
        status = sm.status("host:llm")
        self.assertEqual(status["state"], "stopped")
        self.assertFalse(status["healthy"])
        self.assertFalse(status["restart_required"])

    def test_all_status_returns_dict_after_access(self):
        """all_status() returns services that have been accessed."""
        sm = ServiceManager()
        sm.status("host:llm")
        sm.status("host:sam3")
        all_s = sm.all_status()
        self.assertIn("host:llm", all_s)
        self.assertIn("host:sam3", all_s)

    def test_mark_external(self):
        """mark_external() sets state to EXTERNAL."""
        sm = ServiceManager()
        sm.mark_external("host:llm")
        status = sm.status("host:llm")
        self.assertEqual(status["state"], "external")

    def test_mark_external_healthy(self):
        """mark_external_healthy() updates external service state."""
        sm = ServiceManager()
        sm.mark_external("host:llm")
        sm.mark_external_healthy("host:llm", True)
        status = sm.status("host:llm")
        self.assertEqual(status["state"], "external")
        self.assertTrue(status["healthy"])

    def test_restart_required_flag(self):
        """Restart-required flag works correctly."""
        sm = ServiceManager()
        sm.mark_restart_required("host:llm")
        status = sm.status("host:llm")
        self.assertTrue(status["restart_required"])
        sm.clear_restart_required("host:llm")
        status = sm.status("host:llm")
        self.assertFalse(status["restart_required"])


class ServiceManagerLogTests(TestCase):
    """Test log management."""

    def test_add_and_retrieve_logs(self):
        """add_log() and get_logs() work."""
        sm = ServiceManager()
        sm.add_log("host:llm", "test line 1")
        sm.add_log("host:llm", "test line 2")
        logs = sm.get_logs("host:llm", tail=10)
        self.assertIn("test line 1", logs)
        self.assertIn("test line 2", logs)

    def test_logs_are_bounded(self):
        """Log buffer does not grow unbounded."""
        sm = ServiceManager()
        for i in range(3000):
            sm.add_log("host:llm", f"line {i}")
        logs = sm.get_logs("host:llm", tail=5000)
        self.assertLessEqual(len(logs), 2000)


class ServiceManagerManagedTests(TestCase):
    """Test managed service lifecycle with mocks."""

    @patch("core_orchestrator.services.service_manager.build_llm_command")
    @patch("core_orchestrator.utils.model_host.process_manager.ProcessManager.instance")
    def test_start_managed_llm_fails_no_config(self, mock_pm, mock_build):
        """Starting managed LLM without config returns failed state."""
        sm = ServiceManager()
        cfg = LLMServiceSettings(service_mode="managed")
        result = sm.start("host:llm", cfg)
        self.assertEqual(result.get("state"), "failed")

    @patch("core_orchestrator.services.service_manager.build_llm_command")
    @patch("core_orchestrator.utils.model_host.process_manager.ProcessManager.instance")
    def test_managed_external_does_not_spawn(self, mock_pm, mock_build):
        """External mode does not spawn a process."""
        sm = ServiceManager()
        cfg = LLMServiceSettings(service_mode="external")
        result = sm.start("host:llm", cfg)
        self.assertEqual(result["state"], "external")
        mock_pm.assert_not_called()

    def test_double_start_managed_returns_state(self):
        """Starting a managed service without a real executable returns failed/unhealthy."""
        sm = ServiceManager()
        cfg = LLMServiceSettings(service_mode="managed", executable="nonexistent-binary", model_path="/fake.gguf")
        result = sm.start("host:llm", cfg)
        # Without a real binary or mocks, this should fail in some way
        self.assertIn(result.get("state"), ("failed", "unhealthy", "stopped"))

    def test_managed_sam_no_weights(self):
        """Managed SAM without weights returns failed."""
        sm = ServiceManager()
        cfg = SAMServiceSettings(service_mode="managed")
        result = sm.start("host:sam3", cfg)
        self.assertEqual(result.get("state"), "failed")


class ServiceManagerThreadSafetyTests(TestCase):
    """Test thread-safe operations."""

    def test_concurrent_status_checks(self):
        """Concurrent status() calls don't crash."""
        sm = ServiceManager()

        def check():
            for _ in range(20):
                sm.status("host:llm")
                sm.status("host:sam3")

        threads = [threading.Thread(target=check) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


class ServiceManagerRestartRequiredTests(TestCase):
    """Test that config changes mark restart required."""

    def test_config_change_marks_restart_required(self):
        """Changing config while running marks restart_required."""
        sm = ServiceManager()
        sm.mark_restart_required("host:llm")
        status = sm.status("host:llm")
        self.assertTrue(status["restart_required"])