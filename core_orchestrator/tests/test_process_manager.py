"""
Tests for ProcessManager concurrency — spawn race, AlreadyRunningError.
"""

import threading
import time
from unittest.mock import patch, MagicMock

from django.test import TestCase

from core_orchestrator.utils.model_host.process_manager import (
    ProcessManager,
    AlreadyRunningError,
    reset_process_manager_for_tests,
)
from .test_helpers import setup_test_config_dir, teardown_test_config_dir, reset_all_singletons


class ProcessManagerConcurrencyTests(TestCase):
    def setUp(self):
        self.tmpdir = setup_test_config_dir()
        reset_all_singletons()

    def tearDown(self):
        teardown_test_config_dir(self.tmpdir)
        reset_all_singletons()

    @patch("core_orchestrator.utils.model_host.process_manager.subprocess.Popen")
    def test_concurrent_start_calls_popen_once(self, mock_popen):
        """Two threads calling start() for the same name — Popen called once."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None  # alive
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_popen.return_value = mock_proc

        pm = ProcessManager.instance()
        results = []
        barrier = threading.Event()

        def start_attempt():
            barrier.wait(timeout=2)
            try:
                pid = pm.start("test:same", ["echo", "hello"])
                results.append(("ok", pid))
            except AlreadyRunningError as e:
                results.append(("already_running", str(e)))
            except Exception as e:
                results.append(("error", str(e)))

        t1 = threading.Thread(target=start_attempt)
        t2 = threading.Thread(target=start_attempt)
        t1.start()
        t2.start()
        barrier.set()
        t1.join(timeout=5)
        t2.join(timeout=5)

        # Popen should be called at most once
        self.assertLessEqual(mock_popen.call_count, 1)

        # At least one thread should succeed or get already_running
        states = [r[0] for r in results]
        self.assertTrue(any(s in ("ok", "already_running") for s in states),
                        f"Unexpected results: {results}")

    @patch("core_orchestrator.utils.model_host.process_manager.subprocess.Popen")
    def test_different_names_start_independently(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_popen.return_value = mock_proc

        pm = ProcessManager.instance()
        pid1 = pm.start("test:llm", ["echo", "1"])
        pid2 = pm.start("test:sam", ["echo", "2"])
        self.assertEqual(pid1, 12345)
        self.assertEqual(pid2, 12345)

    @patch("core_orchestrator.utils.model_host.process_manager.subprocess.Popen")
    def test_failed_spawn_clears_reservation(self, mock_popen):
        mock_popen.side_effect = FileNotFoundError("Binary not found")
        pm = ProcessManager.instance()
        with self.assertRaises(FileNotFoundError):
            pm.start("test:failed", ["nonexistent"])

        # Should be able to start again (no stuck reservation)
        mock_popen.side_effect = None
        mock_proc = MagicMock()
        mock_proc.pid = 999
        mock_proc.poll.return_value = None
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_popen.return_value = mock_proc
        pid = pm.start("test:failed", ["echo"])
        self.assertEqual(pid, 999)


class ProcessManagerBoundedBufferTests(TestCase):
    def setUp(self):
        self.tmpdir = setup_test_config_dir()
        reset_all_singletons()

    def tearDown(self):
        teardown_test_config_dir(self.tmpdir)
        reset_all_singletons()

    @patch("core_orchestrator.utils.model_host.process_manager.subprocess.Popen")
    def test_output_buffer_is_bounded(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_popen.return_value = mock_proc

        pm = ProcessManager.instance()
        pm.start("test:bounded", ["echo"])

        # Write more than MAX_LOG_LINES
        record = pm._processes["test:bounded"]
        for i in range(3000):
            record.stdout_buf.append(f"line {i}")

        stdout, _ = pm.get_output("test:bounded")
        self.assertLessEqual(len(stdout), 2000)
