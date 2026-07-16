from __future__ import annotations

import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path

from core.errors import AnalysisError

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*$")


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    path: str
    size_bytes: int
    content_type: str


class ArtifactStore:
    """Resolve only regular non-symlink artifacts below one run directory."""

    def __init__(self, output_root: Path, run_id: str) -> None:
        if not _RUN_ID.fullmatch(run_id):
            raise AnalysisError("Invalid run ID.")
        self.output_root = Path(output_root).resolve()
        self.run_id = run_id
        requested_directory = self.output_root / run_id
        if requested_directory.is_symlink():
            raise AnalysisError("Run artifacts are unavailable.")
        self.run_directory = requested_directory.resolve(strict=True)
        if not self.run_directory.is_relative_to(self.output_root) or not self.run_directory.is_dir():
            raise AnalysisError("Run artifacts are unavailable.")

    def list(self) -> list[ArtifactRecord]:
        records: list[ArtifactRecord] = []
        for candidate in self.run_directory.rglob("*"):
            if candidate.is_symlink() or not candidate.is_file():
                continue
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(self.run_directory):
                continue
            relative = candidate.relative_to(self.run_directory).as_posix()
            records.append(
                ArtifactRecord(
                    path=relative,
                    size_bytes=candidate.stat().st_size,
                    content_type=mimetypes.guess_type(candidate.name)[0] or "application/octet-stream",
                )
            )
        return sorted(records, key=lambda record: record.path)

    def resolve(self, relative_path: str) -> Path:
        if not isinstance(relative_path, str) or "\x00" in relative_path:
            raise AnalysisError("Invalid artifact path.")
        normalized = relative_path.replace("\\", "/")
        parts = normalized.split("/")
        if not normalized or normalized.startswith("/") or any(part in ("", ".", "..") for part in parts):
            raise AnalysisError("Invalid artifact path.")
        candidate = self.run_directory.joinpath(*parts)
        relative_candidate = candidate.relative_to(self.run_directory)
        for index in range(1, len(relative_candidate.parts) + 1):
            if (self.run_directory / Path(*relative_candidate.parts[:index])).is_symlink():
                raise AnalysisError("Artifact is unavailable.")
        if not candidate.is_file():
            raise AnalysisError("Artifact is unavailable.")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(self.run_directory) or not resolved.is_file():
            raise AnalysisError("Artifact is unavailable.")
        return resolved
