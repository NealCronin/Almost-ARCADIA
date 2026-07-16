import threading
from unittest.mock import Mock, patch

import pytest

from core.errors import ServiceStartupError
from core.services.controller import ServiceController
from core.services.specs import ServiceEndpoint, ServiceSpec


def _process() -> Mock:
    process = Mock()
    process.poll.return_value = None
    return process


@patch("core.services.llm_runtime.LLMRuntime.wait_ready")
@patch("core.services.llm_runtime.subprocess.Popen")
def test_start_tracks_service(mock_popen: Mock, mock_ready: Mock, tmp_path) -> None:
    process = _process()
    mock_popen.return_value = process
    controller = ServiceController(log_dir=tmp_path, allow_test_commands=True)
    endpoint = controller.start(ServiceSpec("llm", 8081, {"command": ["fake"]}))
    assert endpoint.port == 8081
    assert controller.is_running(8081)
    mock_ready.assert_called_once()


@patch("core.services.llm_runtime.LLMRuntime.wait_ready")
@patch("core.services.llm_runtime.subprocess.Popen")
def test_replacing_port_stops_previous_service(mock_popen: Mock, mock_ready: Mock, tmp_path) -> None:
    first, second = _process(), _process()
    mock_popen.side_effect = [first, second]
    controller = ServiceController(log_dir=tmp_path, allow_test_commands=True)
    spec = ServiceSpec("llm", 8081, {"command": ["fake"]})
    controller.start(spec)
    controller.start(spec)
    first.terminate.assert_called_once()


@patch("core.services.llm_runtime.LLMRuntime.wait_ready", side_effect=ServiceStartupError("timeout"))
@patch("core.services.llm_runtime.subprocess.Popen")
def test_failed_startup_cleans_up_child(mock_popen: Mock, _ready: Mock, tmp_path) -> None:
    process = _process()
    mock_popen.return_value = process
    controller = ServiceController(log_dir=tmp_path, allow_test_commands=True)
    with pytest.raises(ServiceStartupError):
        controller.start(ServiceSpec("llm", 8081, {"command": ["fake"]}))
    process.terminate.assert_called_once()
    assert not controller.is_running(8081)


@patch("core.services.llm_runtime.LLMRuntime.wait_ready", side_effect=ServiceStartupError("LLM startup cancelled."))
@patch("core.services.llm_runtime.subprocess.Popen")
def test_startup_cancellation_cleans_up_owned_process(mock_popen: Mock, mock_ready: Mock, tmp_path) -> None:
    process = _process()
    mock_popen.return_value = process
    cancelled = threading.Event()
    controller = ServiceController(log_dir=tmp_path, allow_test_commands=True)

    with pytest.raises(ServiceStartupError, match="startup cancelled"):
        controller.start(ServiceSpec("llm", 8081, {"command": ["fake"]}), cancel_event=cancelled)

    mock_ready.assert_called_once_with(
        process,
        ServiceEndpoint("127.0.0.1", 8081, "llm"),
        timeout=600.0,
        cancel_event=cancelled,
    )
    process.terminate.assert_called_once()
    assert not controller.is_running(8081)


@patch("core.services.controller.LLMRuntime.wait_ready", side_effect=[None, ServiceStartupError("timeout")])
@patch("core.services.controller.LLMRuntime.launch")
def test_failed_replacement_cleans_new_process_and_leaves_port_unregistered(
    mock_launch: Mock,
    _mock_ready: Mock,
    tmp_path,
) -> None:
    first_process, second_process = _process(), _process()
    first_log, second_log = Mock(), Mock()
    spec = ServiceSpec("llm", 8081, {"model_path": "model.gguf"})
    endpoint = ServiceEndpoint("127.0.0.1", 8081, "llm")
    mock_launch.side_effect = [
        (first_process, first_log, endpoint),
        (second_process, second_log, endpoint),
    ]
    controller = ServiceController(log_dir=tmp_path)

    controller.start(spec)
    with pytest.raises(ServiceStartupError, match="previous owned service was stopped"):
        controller.start(spec)

    first_process.terminate.assert_called_once()
    second_process.terminate.assert_called_once()
    first_log.close.assert_called_once()
    second_log.close.assert_called_once()
    assert controller.list_services() == []


def test_stop_unknown_port_does_not_kill_process(tmp_path) -> None:
    controller = ServiceController(log_dir=tmp_path)
    controller.stop(6550)
    assert controller.list_services() == []
