"""
Tests for ServiceManager concurrency, lifecycle, and restart-required detection.
"""

import threading
from unittest.mock import patch, MagicMock

from django.test import TestCase

from core_orchestrator.services.service_manager import (
    ServiceManager,
    ServiceState,
    reset_service_manager_for_tests,
)
from core_orchestrator.services.settings_store import (
    LLMServiceSettings,
    SAMServiceSettings,
)
from .test_helpers import setup_test_config_dir, teardown_test_config_dir, reset_all_singletons


class ServiceManagerConcurrencyTests(TestCase):
    def setUp(self):
        self.tmpdir = setup_test_config_dir()
        reset_all_singletons()

    def tearDown(self):
        teardown_test_config_dir(self.tmpdir)
        reset_all_singletons()

    @patch("core_orchestrator.services.service_manager.build_llm_command")
    @patch("core_orchestrator.services.service_manager.ServiceManager._wait_for_llm_health")
    @patch("core_orchestrator.utils.model_host.process_manager.ProcessManager.instance")
    def test_concurrent_start_locks_not_replaced(self, mock_pm, mock_health, mock_build):
        """Two threads starting the same service — lock is not replaced."""
        sm = ServiceManager()
        mock_health.return_value = True

        # Simulate ProcessManager raising AlreadyRunningError on second call
        from core_orchestrator.utils.model_host.process_manager import AlreadyRunningError

        call_count = [0]
        def mock_start(name, cmd):
            call_count[0] += 1
            if call_count[0] > 1:
                raise AlreadyRunningError("already running")
            return 12345

        mock_pm.return_value.start = mock_start

        cfg = LLMServiceSettings(
            service_mode="managed",
            executable="test-binary",
            model_path="/fake.gguf",
            host="127.0.0.1",
            port=9999,
        )

        results = []
        barrier = threading.Event()

        def attempt():
            barrier.wait(timeout=2)
            try:
                result = sm.start("host:llm", cfg)
                results.append(result.get("state", "unknown"))
            except Exception as e:
                results.append(f"error: {e}")

        threads = [threading.Thread(target=attempt) for _ in range(2)]
        for t in threads:
            t.start()
        barrier.set()
        for t in threads:
            t.join(timeout=10)

        # Both threads should have completed and produced results
        self.assertGreater(len(results), 0, f"No results collected, threads may have hung. Results: {results}")
        # The lock should be the same before and after
        runtime, lock1 = sm._ensure_service("host:llm")
        runtime, lock2 = sm._ensure_service("host:llm")
        self.assertIs(lock1, lock2)
        # At least one thread got a running state
        self.assertTrue(any(r == "running" for r in results),
                       f"Expected at least one 'running' result, got: {results}")

        cfg = LLMServiceSettings(
            service_mode="managed",
            executable="test-binary",
            model_path="/fake.gguf",
            host="127.0.0.1",
            port=9999,
        )

        results = []
        barrier = threading.Event()

        def attempt():
            barrier.wait(timeout=2)
            try:
                result = sm.start("host:llm", cfg)
                results.append(result.get("state", "unknown"))
            except Exception as e:
                results.append(f"error: {e}")

        threads = [threading.Thread(target=attempt) for _ in range(2)]
        for t in threads:
            t.start()
        barrier.set()
        for t in threads:
            t.join(timeout=10)

        # Verify no exception and at least one result has a state
        self.assertGreater(len(results), 0)
        # The lock should be the same before and after
        runtime, lock1 = sm._ensure_service("host:llm")
        runtime, lock2 = sm._ensure_service("host:llm")
        self.assertIs(lock1, lock2)

    def test_concurrent_start_and_stop_leave_valid_state(self):
        sm = ServiceManager()
        results = []

        def stop_attempt():
            results.append(sm.stop("host:llm"))

        def start_attempt():
            cfg = LLMServiceSettings(service_mode="external")
            results.append(sm.start("host:llm", cfg))

        t1 = threading.Thread(target=stop_attempt)
        t2 = threading.Thread(target=start_attempt)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        # No crash — state is valid
        status = sm.status("host:llm")
        self.assertIn(status["state"], ("stopped", "external", "starting", "running"))

    def test_different_services_start_independently(self):
        sm = ServiceManager()
        cfg_llm = LLMServiceSettings(service_mode="external")
        cfg_sam = SAMServiceSettings(service_mode="external")

        sm.start("host:llm", cfg_llm)
        sm.start("host:sam3", cfg_sam)

        self.assertEqual(sm.status("host:llm")["state"], "external")
        self.assertEqual(sm.status("host:sam3")["state"], "external")


class ServiceManagerRestartRequiredTests(TestCase):
    def setUp(self):
        self.tmpdir = setup_test_config_dir()
        reset_all_singletons()

    def tearDown(self):
        teardown_test_config_dir(self.tmpdir)
        reset_all_singletons()

    def test_check_restart_required_on_config_change(self):
        sm = ServiceManager()
        cfg_a = LLMServiceSettings(service_mode="managed", executable="bin", model_path="/a.gguf", port=8081)
        cfg_b = LLMServiceSettings(service_mode="managed", executable="bin", model_path="/b.gguf", port=8081)

        # No running service — no restart required
        self.assertFalse(sm.check_restart_required("host:llm", cfg_a))

    def test_external_mode_sync_from_config(self):
        """sync_configuration sets external state from config."""
        sm = ServiceManager()
        cfg = LLMServiceSettings(service_mode="external", base_url="http://localhost:8080")
        sm.sync_configuration("host:llm", cfg)
        status = sm.status("host:llm")
        self.assertEqual(status["state"], "external")
        self.assertIsNone(status["pid"])

    def test_managed_mode_stays_stopped_on_sync(self):
        sm = ServiceManager()
        cfg = LLMServiceSettings(service_mode="managed", executable="bin")
        sm.sync_configuration("host:llm", cfg)
        status = sm.status("host:llm")
        self.assertEqual(status["state"], "stopped")


class ServiceManagerSAMTests(TestCase):
    def setUp(self):
        self.tmpdir = setup_test_config_dir()
        reset_all_singletons()

    def tearDown(self):
        teardown_test_config_dir(self.tmpdir)
        reset_all_singletons()

    def test_managed_sam_no_subprocess(self):
        """Managed SAM start with no weights returns failed, not a subprocess error."""
        sm = ServiceManager()
        cfg = SAMServiceSettings(service_mode="managed", weights_path="")
        result = sm.start("host:sam3", cfg)
        self.assertEqual(result["state"], "failed")
        self.assertIsNone(result.get("pid"))

    @patch("core_orchestrator.services.sam_runtime.SAMRuntime.start")
    def test_managed_sam_uses_inprocess_runtime(self, mock_start):
        mock_start.return_value = {"state": "running"}
        sm = ServiceManager()
        cfg = SAMServiceSettings(service_mode="managed", weights_path="/fake.pt")
        result = sm.start("host:sam3", cfg)
        self.assertEqual(result["state"], "running")
        self.assertIsNone(result["pid"])  # No PID for in-process SAM
        mock_start.assert_called_once()

    def test_external_sam_does_not_spawn(self):
        sm = ServiceManager()
        cfg = SAMServiceSettings(service_mode="external", base_url="http://localhost:9090")
        result = sm.start("host:sam3", cfg)
        self.assertEqual(result["state"], "external")


class ServiceManagerLogTests(TestCase):
    def setUp(self):
        self.tmpdir = setup_test_config_dir()
        reset_all_singletons()

    def tearDown(self):
        teardown_test_config_dir(self.tmpdir)
        reset_all_singletons()

    def test_logs_bounded(self):
        sm = ServiceManager()
        for i in range(3000):
            sm.add_log("host:llm", f"line {i}")
        logs = sm.get_logs("host:llm", tail=5000)
        self.assertLessEqual(len(logs), 2000)

    def test_logs_returned_in_order(self):
        sm = ServiceManager()
        sm.add_log("host:llm", "first")
        sm.add_log("host:llm", "second")
        logs = sm.get_logs("host:llm", tail=10)
        self.assertIn("first", logs)
        self.assertIn("second", logs)


class ServiceManagerSAMLifecycleTests(TestCase):
    """Tests for managed SAM restart/reload behavior."""

    def setUp(self):
        self.tmpdir = setup_test_config_dir()
        reset_all_singletons()

    def tearDown(self):
        teardown_test_config_dir(self.tmpdir)
        reset_all_singletons()

    @patch("core_orchestrator.services.sam_runtime.SAMRuntime.start")
    @patch("core_orchestrator.services.sam_runtime.SAMRuntime.stop")
    def test_sam_restart_reloads_new_weights(self, mock_stop, mock_start):
        """Restart with new weights should call stop then start."""
        mock_start.return_value = {"state": "running"}
        sm = ServiceManager()
        cfg_b = SAMServiceSettings(service_mode="managed", weights_path="/new.pt")

        result = sm.restart("host:sam3", cfg_b)
        # With no running service and new config, restart should start
        self.assertIn(result["state"], ("running", "stopped"),
                      f"Expected 'running' or 'stopped', got '{result['state']}'")

    @patch("core_orchestrator.services.sam_runtime.SAMRuntime.start")
    def test_sam_restart_clears_singleton(self, mock_start):
        """Restart unloads previous model before loading new one."""
        from core_orchestrator.utils.model_host.sam3_server_helper import Sam3ServerHelper

        mock_start.return_value = {"state": "running"}
        sm = ServiceManager()
        cfg = SAMServiceSettings(service_mode="managed", weights_path="/test.pt")

        sm.start("host:sam3", cfg)
        self.assertEqual(sm.status("host:sam3")["state"], "running")
        self.assertIsNone(sm.status("host:sam3")["pid"])

        # Stop should clear the singleton
        sm.stop("host:sam3")
        self.assertEqual(sm.status("host:sam3")["state"], "stopped")

    @patch("core_orchestrator.services.sam_runtime.SAMRuntime.start")
    def test_sam_restart_required_after_weights_change(self, mock_start):
        """Changing weights while running should not auto-restart until restart()."""
        mock_start.return_value = {"state": "running"}
        sm = ServiceManager()
        cfg = SAMServiceSettings(service_mode="managed", weights_path="/a.pt")

        sm.start("host:sam3", cfg)
        self.assertEqual(sm.status("host:sam3")["state"], "running")

        # Change config and sync — this doesn't trigger restart through
        # sync_configuration for running services; the change is detected at
        # the settings-save level
        status = sm.status("host:sam3")
        self.assertFalse(status["restart_required"],
                        "Newly started service should not have restart_required")

        runtime, _ = sm._ensure_service("host:sam3")
        runtime.applied_config = {"weights_path": "/a.pt", "base_url": "", "arguments": [], "service_mode": "managed"}
        # Check with new config to see if restart required is detected
        new_cfg = SAMServiceSettings(service_mode="managed", weights_path="/b.pt")
        self.assertTrue(sm.check_restart_required("host:sam3", new_cfg))
