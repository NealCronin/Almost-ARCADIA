from __future__ import annotations

import subprocess
from unittest.mock import Mock, patch

import pytest

from core.config import HostListenerConfig
from core.services.host_listener import HostListenerController, HostListenerError, HostListenerRestartError


def _process(pid: int = 123) -> Mock:
    process = Mock()
    process.pid = pid
    process.poll.return_value = None
    process.returncode = None
    return process


def _healthy_response() -> Mock:
    response = Mock()
    response.status_code = 200
    response.json.return_value = {"status": "ok", "service": "instruction"}
    return response


def _controller(tmp_path, **kwargs) -> HostListenerController:
    return HostListenerController(
        tmp_path / "instruction",
        startup_timeout=0.2,
        poll_interval=0,
        local_addresses=lambda: {"127.0.0.1", "192.168.1.20"},
        **kwargs,
    )


@patch("core.services.host_listener.requests.get")
@patch("core.services.host_listener.subprocess.Popen")
def test_start_uses_owned_instruction_server_command(mock_popen: Mock, mock_get: Mock, tmp_path) -> None:
    process = _process(321)
    mock_popen.return_value = process
    mock_get.return_value = _healthy_response()
    controller = _controller(tmp_path)

    status = controller.start(HostListenerConfig("192.168.1.20", 9010))

    command = mock_popen.call_args.args[0]
    assert command[:3] == [__import__("sys").executable, "-m", "core.services.instruction_server"]
    assert command[3:] == [
        "--host",
        "192.168.1.20",
        "--public-host",
        "192.168.1.20",
        "--port",
        "9010",
        "--log-dir",
        str(tmp_path / "instruction"),
    ]
    assert mock_popen.call_args.kwargs["shell"] is False
    assert mock_get.call_args.args[0] == "http://192.168.1.20:9010/health"
    assert status.to_dict() == {
        "state": "running",
        "host": "192.168.1.20",
        "port": 9010,
        "pid": 321,
        "uptime_seconds": 0,
        "message": "Instruction server is running",
        "health_url": "http://192.168.1.20:9010/health",
        "last_error": None,
    }


@patch("core.services.host_listener.requests.get", return_value=_healthy_response())
@patch("core.services.host_listener.subprocess.Popen")
def test_ensure_started_is_idempotent(mock_popen: Mock, _mock_get: Mock, tmp_path) -> None:
    mock_popen.return_value = _process()
    controller = _controller(tmp_path)
    config = HostListenerConfig()

    controller.ensure_started(config)
    controller.ensure_started(config)

    mock_popen.assert_called_once()


@patch("core.services.host_listener.subprocess.Popen")
def test_start_rejects_unassigned_local_ip(mock_popen: Mock, tmp_path) -> None:
    controller = HostListenerController(tmp_path, local_addresses=lambda: {"127.0.0.1"})

    with pytest.raises(HostListenerError, match="not assigned to a local network interface"):
        controller.start(HostListenerConfig("192.168.1.20", 9000))

    mock_popen.assert_not_called()


@patch("core.services.host_listener.requests.get", return_value=_healthy_response())
@patch("core.services.host_listener.subprocess.Popen")
def test_restart_rolls_back_when_listener_log_directory_cannot_be_created(
    mock_popen: Mock, _mock_get: Mock, tmp_path, monkeypatch
) -> None:
    first, restored = _process(111), _process(222)
    mock_popen.side_effect = [first, restored]
    controller = _controller(tmp_path)
    previous = HostListenerConfig()
    controller.start(previous)
    mkdir = Mock(side_effect=[OSError("read-only filesystem"), None])
    monkeypatch.setattr("core.services.host_listener.Path.mkdir", mkdir)

    with pytest.raises(HostListenerRestartError, match="Previous instruction server was restored.") as exc:
        controller.restart(HostListenerConfig("192.168.1.20", 9010), rollback_config=previous)

    assert exc.value.rollback_succeeded is True
    assert controller.status().host == previous.host
    assert mock_popen.call_count == 2


def test_local_address_discovery_uses_interface_addresses_only(monkeypatch) -> None:
    result = Mock(stdout="inet 192.168.1.20 netmask 0xffffff00\n")
    monkeypatch.setattr("core.services.host_listener.shutil.which", lambda command: "/sbin/ifconfig")
    monkeypatch.setattr("core.services.host_listener.subprocess.run", lambda *args, **kwargs: result)

    from core.services.host_listener import local_ipv4_addresses

    assert local_ipv4_addresses() == {"127.0.0.1", "192.168.1.20"}


def test_local_address_discovery_uses_windows_interface_addresses(monkeypatch) -> None:
    result = Mock(stdout="   IPv4 Address. . . . . . . . . . . : 10.0.0.42\n")
    monkeypatch.setattr(
        "core.services.host_listener.shutil.which",
        lambda command: r"C:\Windows\System32\ipconfig.exe" if command == "ipconfig" else None,
    )
    monkeypatch.setattr("core.services.host_listener.subprocess.run", lambda *args, **kwargs: result)

    from core.services.host_listener import local_ipv4_addresses

    assert local_ipv4_addresses() == {"127.0.0.1", "10.0.0.42"}


@patch("core.services.host_listener.subprocess.Popen")
def test_start_detects_child_exit_and_closes_log(mock_popen: Mock, tmp_path) -> None:
    process = _process()
    process.poll.return_value = 7
    process.returncode = 7
    mock_popen.return_value = process
    controller = _controller(tmp_path)

    with pytest.raises(HostListenerError, match="exited with code 7"):
        controller.start(HostListenerConfig())

    assert controller.status().state == "failed"
    assert controller._log_handle is None


@patch("core.services.host_listener.requests.get", return_value=_healthy_response())
@patch("core.services.host_listener.subprocess.Popen")
def test_stop_gracefully_stops_only_owned_process_and_closes_log(mock_popen: Mock, _mock_get: Mock, tmp_path) -> None:
    process = _process()
    mock_popen.return_value = process
    controller = _controller(tmp_path)
    controller.start(HostListenerConfig())
    log_handle = controller._log_handle

    status = controller.stop()

    process.terminate.assert_called_once()
    process.kill.assert_not_called()
    assert log_handle is not None and log_handle.closed
    assert status.state == "stopped"
    assert status.pid is None


@patch("core.services.host_listener.requests.get", return_value=_healthy_response())
@patch("core.services.host_listener.subprocess.Popen")
def test_stop_forces_kill_after_bounded_grace_period(mock_popen: Mock, _mock_get: Mock, tmp_path) -> None:
    process = _process()
    process.wait.side_effect = [subprocess.TimeoutExpired("instruction", 0.01), None]
    mock_popen.return_value = process
    controller = _controller(tmp_path, stop_timeout=0.01)
    controller.start(HostListenerConfig())

    controller.stop()

    process.terminate.assert_called_once()
    process.kill.assert_called_once()


@patch("core.services.host_listener.requests.get", return_value=_healthy_response())
@patch("core.services.host_listener.subprocess.Popen")
def test_restart_replaces_owned_listener_after_health(mock_popen: Mock, _mock_get: Mock, tmp_path) -> None:
    first, second = _process(111), _process(222)
    mock_popen.side_effect = [first, second]
    controller = _controller(tmp_path)
    controller.start(HostListenerConfig())

    status = controller.restart(HostListenerConfig("192.168.1.20", 9010))

    first.terminate.assert_called_once()
    assert status.host == "192.168.1.20"
    assert status.port == 9010
    assert status.pid == 222


@patch("core.services.host_listener.requests.get", return_value=_healthy_response())
@patch("core.services.host_listener.subprocess.Popen")
def test_restart_failure_restores_previous_listener(mock_popen: Mock, _mock_get: Mock, tmp_path) -> None:
    first, failed, restored = _process(111), _process(222), _process(333)
    failed.poll.return_value = 2
    failed.returncode = 2
    mock_popen.side_effect = [first, failed, restored]
    controller = _controller(tmp_path)
    old = HostListenerConfig()
    controller.start(old)

    with pytest.raises(HostListenerRestartError, match="was restored") as exc:
        controller.restart(HostListenerConfig("192.168.1.20", 9010), rollback_config=old)

    assert exc.value.rollback_succeeded is True
    assert controller.status().state == "running"
    assert controller.status().host == "127.0.0.1"
    assert controller.status().last_error is not None
    assert mock_popen.call_count == 3


@patch("core.services.host_listener.requests.get", return_value=_healthy_response())
@patch("core.services.host_listener.subprocess.Popen")
def test_restart_failure_reports_failed_rollback(mock_popen: Mock, _mock_get: Mock, tmp_path) -> None:
    first, failed, rollback_failed = _process(111), _process(222), _process(333)
    failed.poll.return_value = 2
    failed.returncode = 2
    rollback_failed.poll.return_value = 3
    rollback_failed.returncode = 3
    mock_popen.side_effect = [first, failed, rollback_failed]
    controller = _controller(tmp_path)
    old = HostListenerConfig()
    controller.start(old)

    with pytest.raises(HostListenerRestartError, match="Rollback failed") as exc:
        controller.restart(HostListenerConfig("192.168.1.20", 9010), rollback_config=old)

    assert exc.value.rollback_succeeded is False
    assert controller.status().state == "failed"
