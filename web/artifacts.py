from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path

from core.errors import ArcadiaError


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    path: str
    size_bytes: int
    content_type: str


class ArtifactStore:
    def __init__(self, root: str | Path, run_id: str) -> None:
        if not run_id or "/" in run_id or "\\" in run_id or run_id in (".", ".."):
            raise ArcadiaError("Invalid run identifier.")
        self.root = Path(root).resolve()
        self.run_directory = (self.root / run_id).resolve()
        if self.root not in self.run_directory.parents:
            raise ArcadiaError("Invalid run directory.")

    def list(self) -> list[ArtifactRecord]:
        if not self.run_directory.is_dir():
            raise ArcadiaError("Run output was not found.")
        records: list[ArtifactRecord] = []
        for path in sorted(self.run_directory.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(self.run_directory).as_posix()
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            records.append(ArtifactRecord(relative, path.stat().st_size, content_type))
        return records

    def resolve(self, artifact_path: str) -> Path:
        candidate = (self.run_directory / artifact_path).resolve()
        if self.run_directory not in candidate.parents or not candidate.is_file():
            raise ArcadiaError("Artifact was not found.")
        return candidate
