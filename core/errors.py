from __future__ import annotations


class ArcadiaError(Exception):
    """Base error surfaced by the application."""


class ConfigurationError(ArcadiaError):
    pass


class ServiceError(ArcadiaError):
    pass


class ServiceStartupError(ServiceError):
    pass


class ServiceNotRunningError(ServiceError):
    pass


class InstructionError(ServiceError):
    pass


class InferenceError(ArcadiaError):
    def __init__(self, message: str, *, service_type: str | None = None) -> None:
        super().__init__(message)
        self.service_type = service_type


class AnalysisError(ArcadiaError):
    pass
