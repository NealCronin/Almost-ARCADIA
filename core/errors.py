from __future__ import annotations


class ArcadiaError(Exception):
    """Base exception for expected Almost ARCADIA failures."""


class ConfigurationError(ArcadiaError):
    """The application configuration is missing or invalid."""


class ServiceError(ArcadiaError):
    """A service could not be controlled or reached."""


class ServiceStartupError(ServiceError):
    """A launched service did not become ready."""


class ServiceNotRunningError(ServiceError):
    """A requested owned service is not running."""


class InstructionError(ServiceError):
    """The remote instruction server rejected or could not process a request."""


class InferenceError(ArcadiaError):
    """An inference request failed or returned an invalid response."""


class AnalysisError(ArcadiaError):
    """An analysis could not complete."""
