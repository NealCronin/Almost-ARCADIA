from unittest.mock import Mock, patch

from core.services.controller import ServiceController
from core.services.specs import ServiceSpec


@patch("core.services.controller.subprocess.Popen")
def test_start_tracks_service(mock_popen: Mock, tmp_path) -> None:
    process = Mock()
    process.poll.return_value = None
    mock_popen.return_value = process

    controller = ServiceController(log_dir=str(tmp_path))
    endpoint = controller.start(
        ServiceSpec(
            service_type="llm",
            port=8081,
            settings={
                "command": ["python", "-c", "import time; time.sleep(60)"]
            },
        )
    )

    assert endpoint.port == 8081
    assert controller.is_running(8081)


@patch("core.services.controller.subprocess.Popen")
def test_replacing_port_stops_previous_service(
    mock_popen: Mock, tmp_path
) -> None:
    first = Mock()
    first.poll.return_value = None
    second = Mock()
    second.poll.return_value = None
    mock_popen.side_effect = [first, second]

    controller = ServiceController(log_dir=str(tmp_path))
    spec = ServiceSpec(
        service_type="llm",
        port=8081,
        settings={"command": ["fake"]},
    )

    controller.start(spec)
    controller.start(spec)

    first.terminate.assert_called_once()
