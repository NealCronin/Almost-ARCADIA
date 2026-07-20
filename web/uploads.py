from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from core.storage import state_child


class UploadStore:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else state_child("uploads")
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_relative(value: str, fallback: str) -> Path:
        normalized = value.replace("\\", "/").strip().lstrip("/") or fallback
        path = PurePosixPath(normalized)
        if ".." in path.parts or path.is_absolute():
            raise ValueError(f"Unsafe upload path: {value!r}")
        return Path(*path.parts)

    def create(self, files: Iterable[Any], relative_paths: list[str]) -> dict[str, Any]:
        uploaded = list(files)
        if not uploaded:
            raise ValueError("Choose at least one file.")
        upload_id = uuid.uuid4().hex
        final_directory = self.root / upload_id
        temporary = Path(tempfile.mkdtemp(prefix=f".{upload_id}.", dir=self.root))
        size = 0
        names: list[str] = []
        try:
            data_directory = temporary / "data"
            data_directory.mkdir(parents=True)
            for index, item in enumerate(uploaded):
                relative = (
                    relative_paths[index] if index < len(relative_paths) else getattr(item, "name", f"file-{index}")
                )
                target_relative = self._safe_relative(relative, f"file-{index}")
                target = data_directory / target_relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as handle:
                    for chunk in item.chunks() if hasattr(item, "chunks") else [item.read()]:
                        handle.write(chunk)
                        size += len(chunk)
                names.append(target_relative.as_posix())
            source_type = "folder" if len(uploaded) > 1 or any("/" in name for name in names) else "file"
            manifest = {
                "id": upload_id,
                "source_type": source_type,
                "file_count": len(uploaded),
                "size_bytes": size,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "files": names,
            }
            (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            temporary.replace(final_directory)
            return manifest
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def _manifest(self, upload_id: str) -> dict[str, Any]:
        if not upload_id.isalnum():
            raise FileNotFoundError(upload_id)
        path = self.root / upload_id / "manifest.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FileNotFoundError(upload_id) from exc

    def list(self) -> list[dict[str, Any]]:
        manifests: list[dict[str, Any]] = []
        for directory in self.root.iterdir():
            if not directory.is_dir():
                continue
            try:
                manifests.append(self._manifest(directory.name))
            except FileNotFoundError:
                continue
        return sorted(manifests, key=lambda item: item.get("created_at", ""), reverse=True)

    def input_path(self, upload_id: str) -> Path:
        manifest = self._manifest(upload_id)
        data = self.root / upload_id / "data"
        files = manifest.get("files", [])
        if manifest.get("source_type") == "file" and len(files) == 1:
            return data / files[0]
        return data

    def delete(self, upload_id: str) -> None:
        self._manifest(upload_id)
        shutil.rmtree(self.root / upload_id)
