from __future__ import annotations

from pathlib import Path

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

from core.errors import AnalysisError
from web.uploads import UploadStore


def upload(name: str, content: bytes = b"image") -> SimpleUploadedFile:
    return SimpleUploadedFile(name, content, content_type="image/jpeg")


def test_folder_upload_preserves_relative_paths_and_returns_local_directory(tmp_path: Path) -> None:
    store = UploadStore(tmp_path / "workspace" / "uploads")
    manifest = store.create([upload("first.jpg"), upload("second.jpg")], ["mission/first.jpg", "mission/second.jpg"])

    assert manifest["source_type"] == "folder"
    assert [entry["relative_path"] for entry in manifest["files"]] == ["mission/first.jpg", "mission/second.jpg"]
    assert store.input_path(manifest["id"]).name == "files"
    assert store.list()[0]["id"] == manifest["id"]


def test_upload_rejects_traversal_and_absolute_paths_without_writes(tmp_path: Path) -> None:
    store = UploadStore(tmp_path / "uploads")
    for path in ("../escape.jpg", "/tmp/escape.jpg", "C:\\escape.jpg", "folder//escape.jpg"):
        with pytest.raises(AnalysisError):
            store.create([upload("escape.jpg")], [path])
    assert not store.root.exists()


def test_single_upload_returns_stored_file_and_delete_removes_manifest(tmp_path: Path) -> None:
    store = UploadStore(tmp_path / "uploads")
    manifest = store.create([upload("one.jpg", b"one")], ["one.jpg"])
    input_path = store.input_path(manifest["id"])

    assert input_path.is_file()
    assert input_path.read_bytes() == b"one"
    store.delete(manifest["id"])
    assert store.list() == []
