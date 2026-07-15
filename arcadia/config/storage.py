"""JSON configuration file I/O."""

import json
import os
import tempfile
from pathlib import Path

from arcadia.config.models import AppConfig, ConfigError


class JsonConfigRepository:
    """Loads and saves AppConfig instances to a JSON file on disk.

    Writes are atomic: data is written to a temporary file in the same
    directory, then os.replace() is used to swap it into place. This
    prevents partial overwrites of a previously valid configuration.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> AppConfig:
        """Load configuration from disk.

        Returns a default AppConfig if the file does not exist.
        Raises ConfigError for invalid JSON or malformed structure.
        """
        if not self.path.exists():
            return AppConfig()

        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"Cannot read configuration file '{self.path}': {exc}") from exc

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Invalid JSON in configuration file '{self.path}': {exc}") from exc

        try:
            return AppConfig.from_dict(data)
        except ConfigError:
            raise
        except Exception as exc:
            raise ConfigError(f"Invalid configuration structure in '{self.path}': {exc}") from exc

    def save(self, config: AppConfig) -> None:
        """Save configuration to disk atomically.

        Creates parent directories if needed. Writes to a temporary file
        first, then atomically replaces the destination with os.replace().
        Does not modify the supplied AppConfig.
        """
        try:
            data = config.to_dict()
            json_text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        except Exception as exc:
            raise ConfigError(f"Failed to serialize configuration: {exc}") from exc

        parent = self.path.parent
        if parent.exists() and not parent.is_dir():
            raise ConfigError(f"Cannot write configuration: '{parent}' exists and is not a directory")

        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

        tmp = None
        try:
            # Write to a temp file in the same directory for atomic replace
            fd, tmp_path = tempfile.mkstemp(dir=str(parent), prefix=".arcadia-config-", suffix=".tmp")
            tmp = Path(tmp_path)
            try:
                os.write(fd, json_text.encode("utf-8"))
            finally:
                os.close(fd)

            os.replace(str(tmp), str(self.path))
        except Exception:
            # Clean up temp file on any failure
            if tmp is not None and tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            raise ConfigError(f"Failed to write configuration file '{self.path}': "
                              f"write or atomic replace failed") from None
