from __future__ import annotations

import pytest

from core.services.sam_checkpoint import SAMCheckpointStore


def test_checkpoint_store_writes_into_huggingface_models(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCADIA_MODELS_DIR", str(tmp_path / "huggingface"))
    payload = SAMCheckpointStore.save_chunks([b"sam", b"3"], "sam3.pt", expected_size=4)

    path = tmp_path / "huggingface" / "models" / "sam3.pt"
    assert path.read_bytes() == b"sam3"
    assert payload["path"] == str(path.resolve())
    assert payload["size_bytes"] == 4


def test_checkpoint_store_rejects_non_pt_files(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCADIA_MODELS_DIR", str(tmp_path / "huggingface"))
    with pytest.raises(ValueError, match=r"\.pt extension"):
        SAMCheckpointStore.save_chunks([b"bad"], "sam3.bin", expected_size=3)


def test_checkpoint_write_is_atomic_when_upload_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCADIA_HUGGINGFACE_DIR", str(tmp_path / "huggingface"))
    original = SAMCheckpointStore.save_chunks([b"old"], "sam3.pt", expected_size=3)

    with pytest.raises(ValueError, match="incomplete"):
        SAMCheckpointStore.save_chunks([b"new"], "sam3.pt", expected_size=4)

    target = tmp_path / "huggingface" / "models" / "sam3.pt"
    assert target.read_bytes() == b"old"
    assert original["path"] == str(target.resolve())
    assert not list(target.parent.glob("*.part"))


def test_checkpoint_upload_enforces_size_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCADIA_HUGGINGFACE_DIR", str(tmp_path / "huggingface"))
    monkeypatch.setenv("ARCADIA_SAM_CHECKPOINT_MAX_BYTES", "3")

    with pytest.raises(ValueError, match="upload limit"):
        SAMCheckpointStore.save_chunks([b"four"], "sam3.pt")
