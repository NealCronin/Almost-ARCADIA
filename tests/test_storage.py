from __future__ import annotations

from core.storage import state_child, state_directory
from web.uploads import UploadStore


def test_state_directory_keeps_browser_uploads_outside_project(monkeypatch, tmp_path):
    monkeypatch.setenv("ARCADIA_STATE_DIR", str(tmp_path / "state"))

    assert state_directory() == tmp_path / "state"
    assert state_child("uploads") == tmp_path / "state" / "uploads"
    assert UploadStore().root == tmp_path / "state" / "uploads"


def test_state_directory_creates_logs_and_outputs(monkeypatch, tmp_path):
    monkeypatch.setenv("ARCADIA_STATE_DIR", str(tmp_path / "state"))

    assert state_child("logs").is_dir()
    assert state_child("outputs").is_dir()
