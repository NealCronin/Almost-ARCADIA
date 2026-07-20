from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterable, Iterable
from pathlib import Path
from typing import Any

from core.services.llm_runtime import LLMRuntime

_DEFAULT_MAX_BYTES = 12 * 1024 * 1024 * 1024


class SAMCheckpointStore:
    """Atomically store user-selected SAM checkpoints on a compute node."""

    @staticmethod
    def max_bytes() -> int:
        raw = os.environ.get("ARCADIA_SAM_CHECKPOINT_MAX_BYTES", str(_DEFAULT_MAX_BYTES))
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError("ARCADIA_SAM_CHECKPOINT_MAX_BYTES must be an integer.") from exc
        if value < 1:
            raise ValueError("ARCADIA_SAM_CHECKPOINT_MAX_BYTES must be positive.")
        return value

    @classmethod
    def directory(cls) -> Path:
        return LLMRuntime.cache_directory("models")

    @staticmethod
    def safe_filename(filename: str) -> str:
        normalized = str(filename).replace("\\", "/").strip()
        name = Path(normalized).name
        if not name or name in {".", ".."}:
            raise ValueError("A checkpoint filename is required.")
        if Path(name).suffix.lower() != ".pt":
            raise ValueError("SAM3 checkpoints must use the .pt extension.")
        return name

    @classmethod
    def validate_checkpoint_path(cls, value: str | Path) -> Path:
        path = Path(value).expanduser().resolve()
        root = cls.directory().resolve()
        if path.suffix.lower() != ".pt":
            raise ValueError("SAM3 checkpoint must use the .pt extension.")
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"SAM3 checkpoint must be stored under {root}.") from exc
        if not path.is_file():
            raise FileNotFoundError(f"SAM3 checkpoint does not exist: {path}")
        return path

    @classmethod
    def _paths(cls, filename: str) -> tuple[Path, Path]:
        target = cls.directory() / cls.safe_filename(filename)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
        return target, temporary

    @classmethod
    def _validate_expected_size(cls, expected_size: int | None) -> None:
        if expected_size is None:
            return
        if isinstance(expected_size, bool) or expected_size < 0:
            raise ValueError("Checkpoint size must be a non-negative integer.")
        if expected_size > cls.max_bytes():
            raise ValueError(f"Checkpoint exceeds the {cls.max_bytes()}-byte upload limit.")

    @classmethod
    def save_chunks(
        cls,
        chunks: Iterable[bytes],
        filename: str,
        *,
        expected_size: int | None = None,
    ) -> dict[str, Any]:
        cls._validate_expected_size(expected_size)
        target, temporary = cls._paths(filename)
        total = 0
        try:
            with temporary.open("wb") as handle:
                for chunk in chunks:
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > cls.max_bytes():
                        raise ValueError(f"Checkpoint exceeds the {cls.max_bytes()}-byte upload limit.")
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if expected_size is not None and total != expected_size:
                raise ValueError(f"Checkpoint upload was incomplete: expected {expected_size} bytes, received {total}.")
            os.replace(temporary, target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return {"path": str(target.resolve()), "filename": target.name, "size_bytes": total}

    @classmethod
    async def save_async_chunks(
        cls,
        chunks: AsyncIterable[bytes],
        filename: str,
        *,
        expected_size: int | None = None,
    ) -> dict[str, Any]:
        cls._validate_expected_size(expected_size)
        target, temporary = cls._paths(filename)
        total = 0
        try:
            with temporary.open("wb") as handle:
                async for chunk in chunks:
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > cls.max_bytes():
                        raise ValueError(f"Checkpoint exceeds the {cls.max_bytes()}-byte upload limit.")
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if expected_size is not None and total != expected_size:
                raise ValueError(f"Checkpoint upload was incomplete: expected {expected_size} bytes, received {total}.")
            os.replace(temporary, target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return {"path": str(target.resolve()), "filename": target.name, "size_bytes": total}
