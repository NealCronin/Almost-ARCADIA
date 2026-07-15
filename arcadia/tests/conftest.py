import pytest
from arcadia.process import ProcessLauncher


@pytest.fixture
def launcher():
    """Provide a launcher that cleans up after itself."""
    launcher = ProcessLauncher()
    yield launcher
    launcher.stop_all()
