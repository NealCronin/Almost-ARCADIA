from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from django.core.files.uploadedfile import UploadedFile
from django.utils.text import get_valid_filename

from core.errors import AnalysisError

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


class UploadStore:
    """Filesystem-backed client uploads retained beneath one workspace root."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def create(self, files: Iterable[UploadedFile], relative_paths: Iterable[str]) -> dict[str, Any]:
        file_list = list(files)
        paths = list(relative_paths)
        if not file_list or len(file_list) != len(paths):
            raise AnalysisError("Upload files and relative paths must be supplied in matching non-empty lists.")
        normalized = [self._normalize_relative_path(value) for value in paths]
        source_type = self._source_type(normalized)
        upload_id = uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{upload_id}.", dir=self.root))
        try:
            files_root = temporary / "files"
            records: list[dict[str, Any]] = []
            total_size = 0
            used_paths: set[Path] = set()
            for upload, relative_path in zip(file_list, normalized, strict=True):
                stored_path = self._stored_path(relative_path, used_paths)
                destination = files_root / stored_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                size = 0
                with destination.open("xb") as handle:
                    for chunk in upload.chunks():
                        handle.write(chunk)
                        size += len(chunk)
                total_size += size
                records.append(
                    {"relative_path": relative_path.as_posix(), "path": stored_path.as_posix(), "size": size}
                )
            manifest = {
                "id": upload_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_type": source_type,
                "file_count": len(records),
                "size_bytes": total_size,
                "files": records,
            }
            (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, self.root / upload_id)
            return manifest
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def list(self) -> list[dict[str, Any]]:
        manifests: list[dict[str, Any]] = []
        if not self.root.exists():
            return manifests
        for directory in self.root.iterdir():
            if directory.name.startswith(".") or not directory.is_dir() or directory.is_symlink():
                continue
            try:
                manifests.append(self._manifest(directory.name))
            except AnalysisError:
                continue
        return sorted(manifests, key=lambda manifest: str(manifest["created_at"]), reverse=True)

    def delete(self, upload_id: str) -> None:
        directory = self._directory(upload_id)
        if not directory.exists():
            raise FileNotFoundError(upload_id)
        shutil.rmtree(directory)

    def input_path(self, upload_id: str) -> Path:
        manifest = self._manifest(upload_id)
        directory = self._directory(upload_id)
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise AnalysisError("Upload manifest has no files.")
        if manifest.get("file_count") == 1:
            stored_path = files[0].get("path") if isinstance(files[0], dict) else None
            if not isinstance(stored_path, str):
                raise AnalysisError("Upload manifest is invalid.")
            candidate = directory / "files" / self._normalize_relative_path(stored_path)
        else:
            candidate = directory / "files"
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(directory.resolve(strict=True)):
            raise AnalysisError("Upload path escapes its retained directory.")
        return resolved

    def _manifest(self, upload_id: str) -> dict[str, Any]:
        directory = self._directory(upload_id)
        if not directory.exists():
            raise FileNotFoundError(upload_id)
        try:
            payload = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AnalysisError("Upload manifest is unavailable.") from exc
        if not isinstance(payload, dict) or payload.get("id") != upload_id:
            raise AnalysisError("Upload manifest is invalid.")
        return payload

    def _directory(self, upload_id: str) -> Path:
        try:
            normalized_id = uuid.UUID(hex=upload_id).hex
        except (ValueError, AttributeError) as exc:
            raise AnalysisError("Invalid upload ID.") from exc
        directory = self.root / normalized_id
        resolved_root = self.root.resolve()
        resolved_directory = directory.resolve(strict=False)
        if not resolved_directory.is_relative_to(resolved_root):
            raise AnalysisError("Invalid upload directory.")
        return directory

    @staticmethod
    def _normalize_relative_path(value: str) -> Path:
        if not isinstance(value, str) or "\x00" in value:
            raise AnalysisError("Upload path is invalid.")
        normalized = value.replace("\\", "/")
        if normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized) or Path(normalized).is_absolute():
            raise AnalysisError("Upload path must be relative.")
        raw_parts = normalized.split("/")
        if not raw_parts or any(part in ("", ".", "..") for part in raw_parts):
            raise AnalysisError("Upload path contains an invalid component.")
        if any(not get_valid_filename(part) for part in raw_parts):
            raise AnalysisError("Upload path contains an invalid filename.")
        return Path(*raw_parts)

    @staticmethod
    def _stored_path(relative_path: Path, used_paths: set[Path]) -> Path:
        sanitized = Path(*(get_valid_filename(part) for part in relative_path.parts))
        candidate = sanitized
        stem, suffix = candidate.stem, candidate.suffix
        suffix_index = 1
        while candidate in used_paths:
            candidate = candidate.with_name(f"{stem}-{suffix_index}{suffix}")
            suffix_index += 1
        used_paths.add(candidate)
        return candidate

    @staticmethod
    def _source_type(paths: Sequence[Path]) -> str:
        suffixes = {path.suffix.lower() for path in paths}
        if not suffixes or not suffixes <= (_IMAGE_EXTENSIONS | _VIDEO_EXTENSIONS):
            raise AnalysisError("Uploads must be supported image or video files.")
        nested = any(len(path.parts) > 1 for path in paths)
        if nested:
            if not suffixes <= _IMAGE_EXTENSIONS:
                raise AnalysisError("Folder uploads must contain only images.")
            return "folder"
        if len(paths) == 1 and next(iter(suffixes)) in _VIDEO_EXTENSIONS:
            return "video"
        if suffixes <= _IMAGE_EXTENSIONS:
            return "image" if len(paths) == 1 else "images"
        raise AnalysisError("Uploads cannot mix images and videos.")
