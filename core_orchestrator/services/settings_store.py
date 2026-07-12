"""
settings_store.py

Persistent, versioned JSON settings for Almost ARCADIA.

Provides atomic writes, schema migration, malformed-JSON recovery,
and a thread-safe API.  The on-disk file is the authoritative source;
in-memory dataclass mirrors reflect the current state.

Key guarantees
--------------
* ``load()`` returns a deep copy — callers cannot mutate the internal cache.
* ``save()`` deep-copies the caller's settings before caching.
* Unknown nested fields are preserved through round-trips.
* Malformed primary file is preserved with a unique timestamp; backup is
  restored to the active file so the next restart is clean.
* All coercions use explicit validation — no ``int("abc")`` crashes.
"""

from __future__ import annotations

import copy
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

# Known top-level keys — everything else is preserved as unknown
_KNOWN_TOP_KEYS = frozenset({"version", "client", "host", "ui"})


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
class SettingsValidationError(ValueError):
    """Raised when settings data fails validation."""

    def __init__(self, message: str, field_path: str = ""):
        super().__init__(message)
        self.field_path = field_path


def _validate_enum(value: Any, allowed: frozenset[str], field_path: str) -> str:
    if value is None:
        raise SettingsValidationError(f"{field_path} is required", field_path)
    s = str(value)
    if s not in allowed:
        raise SettingsValidationError(
            f"{field_path} must be one of {sorted(allowed)}, got '{s}'", field_path
        )
    return s


def _validate_int(value: Any, field_path: str, minimum: int = 1, maximum: int = 65535) -> int:
    if value is None:
        raise SettingsValidationError(f"{field_path} is required", field_path)
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise SettingsValidationError(f"{field_path} must be an integer, got {value!r}", field_path)
    if n < minimum or n > maximum:
        raise SettingsValidationError(
            f"{field_path} must be between {minimum} and {maximum}, got {n}", field_path
        )
    return n


def _validate_port(value: Any, field_path: str) -> int:
    return _validate_int(value, field_path, minimum=1, maximum=65535)


def _validate_bool(value: Any, field_path: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in ("true", "1"):
            return True
        if value.lower() in ("false", "0"):
            return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise SettingsValidationError(f"{field_path} must be a boolean, got {value!r}", field_path)


def _validate_string(value: Any, field_path: str, default: str = "") -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        return str(value)
    return value


def _validate_string_list(value: Any, field_path: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise SettingsValidationError(f"{field_path} must be an array of strings", field_path)
    result = []
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise SettingsValidationError(
                f"{field_path}[{i}] must be a string, got {type(item).__name__}", field_path
            )
        if item:
            result.append(item)
    return result


def _validate_timeout(value: Any, field_path: str, default: int = 60) -> int:
    if value is None:
        return default
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise SettingsValidationError(f"{field_path} must be a positive integer", field_path)
    if n < 1:
        raise SettingsValidationError(f"{field_path} must be > 0, got {n}", field_path)
    return n


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
        return Path.home() / "AppData" / "Roaming" / "AlmostARCADIA"
    elif system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "AlmostARCADIA"
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        if xdg:
            return Path(xdg) / "almost-arcadia"
        return Path.home() / ".config" / "almost-arcadia"


def config_dir() -> Path:
    """Resolve the config directory (ALMOST_ARCADIA_CONFIG_DIR overrides)."""
    override = os.environ.get("ALMOST_ARCADIA_CONFIG_DIR")
    if override:
        return Path(override)
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
    """Convert a dataclass tree to a plain dict (JSON-safe)."""
    return json.loads(json.dumps(asdict(obj), default=str))


def _build_remote_host(d: Any) -> RemoteHostSettings:
    if not isinstance(d, dict):
        return RemoteHostSettings()
    return RemoteHostSettings(
        host=_validate_string(d.get("host"), "client.remote_host.host", "127.0.0.1"),
        port=_validate_port(d.get("port", 8080), "client.remote_host.port"),
        scheme=_validate_enum(
            d.get("scheme", "http"), frozenset({"http", "https"}), "client.remote_host.scheme"
        ),
    )


def _build_llm(d: Any, prefix: str) -> LLMServiceSettings:
    if not isinstance(d, dict):
        return LLMServiceSettings()
    return LLMServiceSettings(
        service_mode=_validate_enum(
            d.get("service_mode", "managed"),
            frozenset({"managed", "external"}),
            f"{prefix}.service_mode",
        ),
        executable=_validate_string(d.get("executable"), f"{prefix}.executable"),
        model_path=_validate_string(d.get("model_path"), f"{prefix}.model_path"),
        model_id=_validate_string(d.get("model_id"), f"{prefix}.model_id"),
        base_url=_validate_string(d.get("base_url"), f"{prefix}.base_url"),
        api_format=_validate_enum(
            d.get("api_format", "llama-completion"),
            frozenset({"llama-completion", "openai-chat", "openai-responses"}),
            f"{prefix}.api_format",
        ),
        host=_validate_string(d.get("host"), f"{prefix}.host", "127.0.0.1"),
        port=_validate_port(d.get("port", 8081), f"{prefix}.port"),
        arguments=_validate_string_list(d.get("arguments"), f"{prefix}.arguments"),
        startup_timeout_seconds=_validate_timeout(
            d.get("startup_timeout_seconds", 60), f"{prefix}.startup_timeout_seconds"
        ),
        request_timeout_seconds=_validate_timeout(
            d.get("request_timeout_seconds", 120), f"{prefix}.request_timeout_seconds"
        ),
        auto_start=_validate_bool(d.get("auto_start", False), f"{prefix}.auto_start"),
    )


def _build_sam(d: Any, prefix: str) -> SAMServiceSettings:
    if not isinstance(d, dict):
        return SAMServiceSettings()
    return SAMServiceSettings(
        service_mode=_validate_enum(
            d.get("service_mode", "managed"),
            frozenset({"managed", "external"}),
            f"{prefix}.service_mode",
        ),
        weights_path=_validate_string(d.get("weights_path"), f"{prefix}.weights_path"),
        base_url=_validate_string(d.get("base_url"), f"{prefix}.base_url"),
        arguments=_validate_string_list(d.get("arguments"), f"{prefix}.arguments"),
        startup_timeout_seconds=_validate_timeout(
            d.get("startup_timeout_seconds", 60), f"{prefix}.startup_timeout_seconds"
        ),
        request_timeout_seconds=_validate_timeout(
            d.get("request_timeout_seconds", 120), f"{prefix}.request_timeout_seconds"
        ),
        auto_start=_validate_bool(d.get("auto_start", False), f"{prefix}.auto_start"),
    )


def _dict_to_appsettings(data: dict) -> AppSettings:
    """Build an AppSettings from a dict with explicit validation."""
    version = data.get("version", CURRENT_VERSION)

    client_data = data.get("client", {})
    host_data = data.get("host", {})
    ui_data = data.get("ui", {})

    client = ClientSettings(
        llm_mode=_validate_enum(
            client_data.get("llm_mode", "local"),
            frozenset({"local", "remote"}),
            "client.llm_mode",
        ),
        sam3_mode=_validate_enum(
            client_data.get("sam3_mode", "local"),
            frozenset({"local", "remote"}),
            "client.sam3_mode",
        ),
        dataset_path=_validate_string(client_data.get("dataset_path", ""), "client.dataset_path"),
        remote_host=_build_remote_host(client_data.get("remote_host", {})),
        local_llm=_build_llm(client_data.get("local_llm", {}), "client.local_llm"),
        local_sam3=_build_sam(client_data.get("local_sam3", {}), "client.local_sam3"),
    )

    host = HostSettings(
        listen_ip=_validate_string(host_data.get("listen_ip", "0.0.0.0"), "host.listen_ip", "0.0.0.0"),
        listen_port=_validate_port(host_data.get("listen_port", 8080), "host.listen_port"),
        llm=_build_llm(host_data.get("llm", {}), "host.llm"),
        sam3=_build_sam(host_data.get("sam3", {}), "host.sam3"),
    )

    ui = UISettings(
        last_tab=_validate_string(ui_data.get("last_tab", "dashboard"), "ui.last_tab", "dashboard"),
        theme=_validate_string(ui_data.get("theme", "dark"), "ui.theme", "dark"),
    )

    return AppSettings(version=version, client=client, host=host, ui=ui)


def _apply_legacy_migration(data: dict) -> dict:
    """Migrate from older schema versions to the current one."""
    version = data.get("version", 0)

    if version < 1:
        if "client" not in data:
            data["client"] = {}

        legacy_mode = data.get("routing_mode") or data.get("client", {}).get("routing_mode", "local")
        if legacy_mode in ("local", "remote"):
            client = data.setdefault("client", {})
            client.setdefault("llm_mode", legacy_mode)
            client.setdefault("sam3_mode", legacy_mode)

        if "listen_ip" in data and "host" not in data:
            data["host"] = {
                "listen_ip": data.pop("listen_ip", "0.0.0.0"),
                "listen_port": data.pop("listen_port", 8080),
            }

        data["version"] = 1

    return data


# ---------------------------------------------------------------------------
# Deep-merge helper preserving unknown keys at every level
# ---------------------------------------------------------------------------
def _deep_merge_preserving(base: dict, overlay: dict, known_keys: frozenset[str] | None = None) -> dict:
    """
    Deep-merge *overlay* into *base*.

    For keys in *known_keys*, overlay values replace base values (after validation).
    For unknown keys, both base and overlay values are preserved.
    Dict values are merged recursively.
    """
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge_preserving(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _build_raw_from_typed(typed: dict, raw: dict) -> dict:
    """
    Deep-merge typed (known) fields into the raw document, preserving unknown
    keys at every nesting level.  Typed values win over raw for known keys.
    """
    result = copy.deepcopy(raw)
    for key, value in typed.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _build_raw_from_typed(value, result[key])
        else:
            result[key] = copy.deepcopy(value)
    return result


# ---------------------------------------------------------------------------
# SettingsStore
# ---------------------------------------------------------------------------
class SettingsStore:
    """
    Thread-safe persistent settings store backed by a JSON file.

    The on-disk file is authoritative.  ``load()`` returns a deep copy so
    callers cannot mutate the internal cache without saving.
    """

    def __init__(self, directory: Path | str | None = None) -> None:
        self._lock = threading.Lock()
        self._directory = Path(directory or config_dir()).resolve()
        self._directory.mkdir(parents=True, exist_ok=True)
        self._file_path = self._directory / "settings.json"
        self._bak_path = self._directory / "settings.json.bak"
        self._cached: Optional[AppSettings] = None
        self._cached_raw: Optional[dict] = None
        self.last_warning: Optional[str] = None

    # -- Public API ---------------------------------------------------------

    def load(self) -> AppSettings:
        """Load settings from disk.  Returns a deep copy of the cached settings."""
        with self._lock:
            internal = self._load_with_lock()
            return copy.deepcopy(internal)

    def save(self, settings: AppSettings) -> None:
        """Atomically save settings to disk (thread-safe)."""
        with self._lock:
            self._save_with_lock(copy.deepcopy(settings))

    def update(self, updates: dict) -> AppSettings:
        """
        Merge *updates* into current settings, persist, and return the result.

        Unknown nested fields are preserved at all levels.
        Returns a deep copy.
        """
        with self._lock:
            # Load current raw + typed
            current_typed = self._load_with_lock()
            current_raw = self._cached_raw if self._cached_raw is not None else _dataclass_to_dict(current_typed)

            # Deep-merge the update into the raw document
            merged_raw = _deep_merge_preserving(current_raw, updates)

            # Validate known fields from the merged raw
            migrated = _apply_legacy_migration(merged_raw)
            new_settings = _dict_to_appsettings(migrated)

            # Rebuild raw from typed to normalize known fields while preserving unknowns
            typed_dict = _dataclass_to_dict(new_settings)
            final_raw = _build_raw_from_typed(typed_dict, migrated)
            final_raw["version"] = CURRENT_VERSION

            self._save_raw_with_lock(final_raw, new_settings)
            return copy.deepcopy(new_settings)

    def reload(self) -> AppSettings:
        """Force a fresh read from disk."""
        with self._lock:
            self._cached = None
            self._cached_raw = None
            return copy.deepcopy(self._load_with_lock())

    def reset_to_defaults(self) -> AppSettings:
        """Replace saved config with factory defaults and persist it."""
        defaults = AppSettings()
        with self._lock:
            self._save_with_lock(copy.deepcopy(defaults))
            return copy.deepcopy(defaults)

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
            raw_text = path.read_text(encoding="utf-8")
            data = json.loads(raw_text)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            logger.warning("Failed to parse settings file: %s", exc)
            data = self._attempt_backup_recovery(path)
            if data is None:
                logger.warning("No valid backup; loading defaults")
                defaults = AppSettings()
                self._save_with_lock(defaults)
                return defaults
            self.last_warning = "settings_recovered_from_backup"

        # Migrate legacy schema
        data = _apply_legacy_migration(data)

        settings = _dict_to_appsettings(data)
        settings.version = CURRENT_VERSION
        self._cached = settings
        self._cached_raw = data
        return settings

    def _save_with_lock(self, settings: AppSettings) -> None:
        """Save typed settings, preserving unknown fields from current raw."""
        # Get the current raw document to preserve unknowns
        current_raw = self._cached_raw if self._cached_raw is not None else _dataclass_to_dict(settings)
        typed_dict = _dataclass_to_dict(settings)
        final_raw = _build_raw_from_typed(typed_dict, current_raw)
        final_raw["version"] = CURRENT_VERSION
        self._save_raw_with_lock(final_raw, settings)

    def _save_raw_with_lock(self, raw: dict, settings: AppSettings) -> None:
        """Write the raw dict to disk atomically and update the cache."""
        # Backup current file if it exists
        if self._file_path.exists():
            try:
                shutil.copy2(self._file_path, self._bak_path)
            except OSError as exc:
                logger.warning("Failed to create backup: %s", exc)

        fd, tmp_path = tempfile.mkstemp(
            suffix=".json",
            prefix="settings_",
            dir=str(self._directory),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(raw, f, indent=2, ensure_ascii=False)
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
        self._cached_raw = raw

    def _attempt_backup_recovery(self, path: Path) -> Optional[dict]:
        """Try to load the backup file.  Preserve malformed file with timestamp."""
        # Preserve malformed file with unique timestamp name
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        malformed = path.parent / f"settings.json.malformed-{ts}"
        try:
            path.rename(malformed)
            logger.info("Preserved malformed settings as %s", malformed)
        except OSError:
            # If rename fails, try the old fixed name
            malformed_old = path.with_suffix(".json.malformed")
            try:
                path.rename(malformed_old)
                logger.info("Preserved malformed settings as %s", malformed_old)
            except OSError:
                pass

        bak = self._bak_path
        if bak.exists():
            try:
                raw = bak.read_text(encoding="utf-8")
                data = json.loads(raw)
                # Restore backup to active file immediately
                logger.info("Restoring backup to active settings file")
                fd, tmp_path = tempfile.mkstemp(
                    suffix=".json", prefix="settings_", dir=str(self._directory)
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
                return data
            except (json.JSONDecodeError, OSError) as exc2:
                logger.warning("Backup also malformed: %s", exc2)
        return None


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


def reset_settings_store_for_tests() -> None:
    """Reset the global SettingsStore singleton.  Intended for test isolation."""
    global _store
    with _store_lock:
        _store = None