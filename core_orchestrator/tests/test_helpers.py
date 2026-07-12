"""
Test helpers for singleton isolation.
Import these in setUp/tearDown to ensure clean state.
"""

import os
import tempfile
from pathlib import Path


def setup_test_config_dir() -> str:
    """Create a temporary config directory and set the env var."""
    tmpdir = tempfile.mkdtemp(prefix="arcadia_test_")
    os.environ["ALMOST_ARCADIA_CONFIG_DIR"] = tmpdir
    return tmpdir


def teardown_test_config_dir(tmpdir: str) -> None:
    """Remove the temporary config directory."""
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    os.environ.pop("ALMOST_ARCADIA_CONFIG_DIR", None)


def reset_all_singletons():
    """Reset all module-level singletons for test isolation."""
    from core_orchestrator.services.settings_store import reset_settings_store_for_tests
    from core_orchestrator.services.service_manager import reset_service_manager_for_tests
    from core_orchestrator.utils.model_host.process_manager import (
        ProcessManager,
        reset_process_manager_for_tests,
    )
    from core_orchestrator.utils.model_host.sam3_server_helper import Sam3ServerHelper

    reset_settings_store_for_tests()
    reset_service_manager_for_tests()
    reset_process_manager_for_tests()

    # Reset SAM singleton
    Sam3ServerHelper.reset_singleton()

    # Reset Host listener globals
    import core_orchestrator.views.pages as pages
    pages._host_api_running = False
    pages._host_api_thread = None
    pages._host_api_server = None
    pages._host_api_config = {}
    pages._request_log.clear()
