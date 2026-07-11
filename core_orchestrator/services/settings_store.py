"""
settings_store.py

Persistent, versioned JSON settings for Almost ARCADIA.

Provides atomic writes, schema migration, malformed-JSON recovery,
and a thread-safe API.  The on-disk file is the authoritative source;
in-memory dataclass mirrors reflect the current state.

Settings directory precedence
-----------------------------
1. ``ALMOST_ARCADIA_CONFIG_DIR`` env var (portable / testing override)
2. Platform default:
   - Windows: ``%APPDATA%\\AlmostARCADIA\\``
   - macOS:   ``~/Library/Application Support/AlmostARCADIA/``
   - Linux:   ``$XDG_CONFIG_HOME/almost-arcadia/`` or ``~/.config/almost-arcadia/``
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------
CURRENT_VERSION = 1


# ---------------------------------------------------------------------------
# Helper: platform config directory
# ---------------------------------------------------------------------------
def _default_config_dir() -> Path:
    """Return the platform-appropriate config directory."""
    system = platform.system()
    if system == "Windows":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "AlmostARCADIA"
        # fallback
        return Path.home() / "AppData" / "Roaming" / "AlmostARCADIA"
    elif system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "AlmostARCADIA"
    else:
        # Linux / BSD
        xdg = os.environ.get("XDG_CONFIG_HOME")
        if xdg:
            return Path(xdg) / "almost-arcadia"
        return Path.home() / ".config" / "almost-arcadia"


def config_dir() -> Path:
    """
    Resolve the config directory.
    ``ALMOST_ARCADIA_CONFIG_DIR`` env var overrides platform default.
    """
    override = os.environ.get("ALMOST_ARCADIA_CONFIG_DIR")
    if override:
        return Path(override).resolve()
    return _default_config_dir()


# ---------------------------------------------------------------------------
# Dataclass models
# ---------------------------------------------------------------------------
@dataclass
class RemoteHostSettings:
    host: str = "127.0.0.1"
    port: int = 8080
    scheme: str = "http"


@dataclass
class LLMServiceSettings:
    service_mode: str = "managed"  # "managed" | "external"
    executable: str = ""
    model_path: str = ""
    model_id: str = ""
    base_url: str = ""
    api_format: str = "llama-completion"
    host: str = "127.0.0.1"
    port: int = 8081
    arguments: list[str] = field(default_factory=list)
    startup_timeout_seconds: int = 60
    request_timeout_seconds: int = 120
    auto_start: bool = False


@dataclass
class SAMServiceSettings:
    service_mode: str = "managed"  # "managed" | "external"
    weights_path: str = ""
    base_url: str = ""
    arguments: list[str] = field(default_factory=list)
    startup_timeout_seconds: int = 60
    request_timeout_seconds: int = 120
    auto_start: bool = False


@dataclass
class ClientSettings:
    llm_mode: str = "local"  # "local" | "remote"
    sam3_mode: str = "local"  # "local" | "remote"
    dataset_path: str = ""
    remote_host: RemoteHostSettings = field(default_factory=RemoteHostSettings)
    local_llm: LLMServiceSettings = field(default_factory=LLMServiceSettings)
    local_sam3: SAMServiceSettings = field(default_factory=SAMServiceSettings)


@dataclass
class HostSettings:
    listen_ip: str = "0.0.0.0"
    listen_port: int = 8080
    llm: LLMServiceSettings = field(default_factory=LLMServiceSettings)
    sam3: SAMServiceSettings = field(default_factory=SAMServiceSettings)


@dataclass
class UISettings:
    """Non-critical display preferences that survive restarts."""
    last_tab: str = "dashboard"
    theme: str = "dark"


@dataclass
class AppSettings:
    version: int = CURRENT_VERSION
    client: ClientSettings = field(default_factory=ClientSettings)
    host: HostSettings = field(default_factory=HostSettings)
    ui: UISettings = field(default_factory=UISettings)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------
def _dataclass_to_dict(obj: Any) -> dict:
    return json.loads(json.dumps(asdict(obj), default=str))


def _dict_to_appsettings(data: dict) -> AppSettings:
    """Build an AppSettings from a dict, filling missing fields with defaults."""
    version = data.get("version", CURRENT_VERSION)

    def _build_remote_host(d: Any) -> RemoteHostSettings:
        if not isinstance(d, dict):
            return RemoteHostSettings()
        return RemoteHostSettings(
            host=str(d.get("host", "127.0.0.1")),
            port=int(d.get("port", 8080)),
            scheme=str(d.get("scheme", "http")),
        )

    def _build_llm(d: Any) -> LLMServiceSettings:
        if not isinstance(d, dict):
            return LLMServiceSettings()
        return LLMServiceSettings(
            service_mode=str(d.get("service_mode", "managed")),
            executable=str(d.get("executable", "")),
            model_path=str(d.get("model_path", "")),
            model_id=str(d.get("model_id", "")),
            base_url=str(d.get("base_url", "")),
            api_format=str(d.get("api_format", "llama-completion")),
            host=str(d.get("host", "127.0.0.1")),
            port=int(d.get("port", 8081)),
            arguments=list(d.get("arguments", [])),
            startup_timeout_seconds=int(d.get("startup_timeout_seconds", 60)),
            request_timeout_seconds=int(d.get("request_timeout_seconds", 120)),
            auto_start=bool(d.get("auto_start", False)),
        )

    def _build_sam(d: Any) -> SAMServiceSettings:
        if not isinstance(d, dict):
            return SAMServiceSettings()
        return SAMServiceSettings(
            service_mode=str(d.get("service_mode", "managed")),
            weights_path=str(d.get("weights_path", "")),
            base_url=str(d.get("base_url", "")),
            arguments=list(d.get("arguments", [])),
            startup_timeout_seconds=int(d.get("startup_timeout_seconds", 60)),
            request_timeout_seconds=int(d.get("request_timeout_seconds", 120)),
            auto_start=bool(d.get("auto_start", False)),
        )

    client_data = data.get("client", {})
    host_data = data.get("host", {})
    ui_data = data.get("ui", {})

    client = ClientSettings(
        llm_mode=str(client_data.get("llm_mode", "local")),
        sam3_mode=str(client_data.get("sam3_mode", "local")),
        dataset_path=str(client_data.get("dataset_path", "")),
        remote_host=_build_remote_host(client_data.get("remote_host", {})),
        local_llm=_build_llm(client_data.get("local_llm", {})),
        local_sam3=_build_sam(client_data.get("local_sam3", {})),
    )

    host = HostSettings(
        listen_ip=str(host_data.get("listen_ip", "0.0.0.0")),
        listen_port=int(host_data.get("listen_port", 8080)),
        llm=_build_llm(host_data.get("llm", {})),
        sam3=_build_sam(host_data.get("sam3", {})),
    )

    ui = UISettings(
        last_tab=str(ui_data.get("last_tab", "dashboard")),
        theme=str(ui_data.get("theme", "dark")),
    )

    return AppSettings(version=version, client=client, host=host, ui=ui)


def _apply_legacy_migration(data: dict) -> dict:
    """
    Migrate from older schema versions to the current one.
    Handles the legacy ``routing_mode`` field -> independent llm_mode/sam3_mode.
    """
    version = data.get("version", 0)

    if version < 1:
        # v0 -> v1: handle legacy routing_mode
        if "client" not in data:
            data["client"] = {}

        legacy_mode = data.get("routing_mode") or data.get("client", {}).get(
            "routing_mode", "local"
        )
        if legacy_mode in ("local", "remote"):
            client = data.setdefault("client", {})
            client.setdefault("llm_mode", legacy_mode)
            client.setdefault("sam3_mode", legacy_mode)

        # Migrate old host_api_config shape if present
        if "listen_ip" in data and "host" not in data:
            data["host"] = {
                "listen_ip": data.pop("listen_ip", "0.0.0.0"),
                "listen_port": data.pop("listen_port", 8080),
            }

        data["version"] = 1

    return data


# ---------------------------------------------------------------------------
# SettingsStore
# ---------------------------------------------------------------------------
class SettingsStore:
    """
    Thread-safe persistent settings store backed by a JSON file.

    Usage::

        store = SettingsStore()
        settings = store.load()
        settings.client.llm_mode = "remote"
        store.save(settings)
    """

    def __init__(self, directory: Optional[Path | str] = None) -> None:
        self._lock = threading.Lock()
        self._directory = Path(directory or config_dir()).resolve()
        self._directory.mkdir(parents=True, exist_ok=True)
        self._file_path = self._directory / "settings.json"
        self._bak_path = self._directory / "settings.json.bak"
        self._cached: Optional[AppSettings] = None

    # -- Public API ---------------------------------------------------------

    def load(self) -> AppSettings:
        """
        Load settings from disk.

        * Missing file -> create and return defaults.
        * Malformed JSON -> attempt backup, else defaults; malformed file preserved.
        * Backup also malformed -> log warning, return defaults.
        """
        with self._lock:
            return self._load_with_lock()

    def save(self, settings: AppSettings) -> None:
        """Atomically save settings to disk (thread-safe)."""
        with self._lock:
            self._save_with_lock(settings)

    def update(self, updates: dict) -> AppSettings:
        """
        Merge ``updates`` into current settings, persist, and return the result.

        ``updates`` is a partial dict following the same structure as
        the full JSON schema.  Unknown keys are preserved at the top level
        for forward compatibility.
        """
        with self._lock:
            current = self._load_with_lock()
            merged = self._merge_dicts(_dataclass_to_dict(current), updates)

            # Preserve unknown top-level keys from updates for forward compat
            for k in updates:
                if k not in ("version", "client", "host", "ui"):
                    merged[k] = updates[k]
            # Preserve any existing unknown top-level keys from current settings
            current_dict = _dataclass_to_dict(current)
            for k in current_dict:
                if k not in ("version", "client", "host", "ui"):
                    merged.setdefault(k, current_dict[k])

            new_settings = _dict_to_appsettings(merged)
            self._save_with_lock(new_settings, extra_data=merged)
            return new_settings

    def reload(self) -> AppSettings:
        """Force a fresh read from disk."""
        with self._lock:
            self._cached = None
            return self._load_with_lock()

    def reset_to_defaults(self) -> AppSettings:
        """Replace saved config with factory defaults and persist it."""
        defaults = AppSettings()
        with self._lock:
            self._save_with_lock(defaults)
            return defaults

    @property
    def file_path(self) -> Path:
        return self._file_path

    @property
    def directory(self) -> Path:
        return self._directory

    # -- Internal (caller must hold self._lock) -----------------------------

    def _load_with_lock(self) -> AppSettings:
        if self._cached is not None:
            return self._cached

        path = self._file_path
        if not path.exists():
            logger.info("No settings file at %s, creating defaults", path)
            defaults = AppSettings()
            self._save_with_lock(defaults)
            return defaults

        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            logger.warning("Failed to parse settings file: %s", exc)
            data = self._attempt_backup_recovery(path)
            if data is None:
                logger.warning("No valid backup; loading defaults")
                defaults = AppSettings()
                self._save_with_lock(defaults)
                return defaults

        # Migrate legacy schema
        data = _apply_legacy_migration(data)

        settings = _dict_to_appsettings(data)
        settings.version = CURRENT_VERSION
        self._cached = settings
        return settings

    def _save_with_lock(self, settings: AppSettings, extra_data: Optional[dict] = None) -> None:
        """Atomic write: temp file -> fsync -> replace.

        If *extra_data* is provided, its top-level keys are merged into the
        serialised output (preserving unknown/advanced fields).
        """
        # Backup current file if it exists
        if self._file_path.exists():
            try:
                shutil.copy2(self._file_path, self._bak_path)
            except OSError as exc:
                logger.warning("Failed to create backup: %s", exc)

        data = _dataclass_to_dict(settings)
        data["version"] = CURRENT_VERSION

        # Merge extra/unknown fields if provided
        if extra_data:
            for k in extra_data:
                if k not in ("version", "client", "host", "ui"):
                    data[k] = extra_data[k]
            # Re-read version from extra_data if present
            if "version" in extra_data:
                data["version"] = extra_data["version"]

        fd, tmp_path = tempfile.mkstemp(
            suffix=".json",
            prefix="settings_",
            dir=str(self._directory),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(self._file_path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        self._cached = settings

    def _attempt_backup_recovery(self, path: Path) -> Optional[dict]:
        """Try to load the backup file.  Rename malformed file for inspection."""
        # Rename malformed file so it's not silently lost
        malformed = path.with_suffix(".json.malformed")
        try:
            path.rename(malformed)
            logger.info("Preserved malformed settings as %s", malformed)
        except OSError:
            pass

        bak = self._bak_path
        if bak.exists():
            try:
                raw = bak.read_text(encoding="utf-8")
                return json.loads(raw)
            except (json.JSONDecodeError, OSError) as exc2:
                logger.warning("Backup also malformed: %s", exc2)
        return None

    @staticmethod
    def _merge_dicts(base: dict, overlay: dict) -> dict:
        """Recursive dict merge.  overlay values win; lists replaced wholesale."""
        result = base.copy()
        for key, value in overlay.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = SettingsStore._merge_dicts(result[key], value)
            else:
                result[key] = value
        return result


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------
_store: Optional[SettingsStore] = None
_store_lock = threading.Lock()


def get_settings_store() -> SettingsStore:
    """Return the application-global SettingsStore (created on first call)."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = SettingsStore()
    return _store