from __future__ import annotations

from pathlib import Path

import pytest

from core.errors import AnalysisError
from web.artifacts import ArtifactStore


def test_artifact_store_lists_regular_files_and_rejects_escapes(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    run = root / "run-1"
    run.mkdir(parents=True)
    (run / "preview.jpg").write_bytes(b"jpg")
    (run / "analysis.log").write_text("log", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (run / "escape.txt").symlink_to(outside)

    store = ArtifactStore(root, "run-1")
    assert [record.path for record in store.list()] == ["analysis.log", "preview.jpg"]
    assert store.resolve("preview.jpg").read_bytes() == b"jpg"
    for path in ("../outside.txt", "/tmp/outside.txt", "escape.txt"):
        with pytest.raises(AnalysisError):
            store.resolve(path)
