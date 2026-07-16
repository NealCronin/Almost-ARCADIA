from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from core.errors import ServiceError
from core.services.controller import ServiceController
from core.services.llm_settings import validate_llm_settings
from core.services.specs import ServiceSpec


class StartServiceRequest(BaseModel):
    service_type: Literal["llm", "sam3"]
    port: int = Field(ge=1, le=65535)
    settings: dict[str, Any] = Field(default_factory=dict)


class StopServiceRequest(BaseModel):
    port: int = Field(ge=1, le=65535)


def _reject_remote_commands(settings: dict[str, Any], *, service_type: str, public_host: str) -> dict[str, Any]:
    forbidden = {"command", "shell", "executable", "python_executable", "server_module", "model_path", "hf_cache_dir"}
    present = forbidden.intersection(settings)
    if present:
        raise HTTPException(status_code=422, detail=f"Remote settings cannot include: {', '.join(sorted(present))}")
    if service_type != "llm":
        return settings
    if settings.get("models_cache_subdir", "huggingface") != "huggingface":
        raise HTTPException(status_code=422, detail="Remote LLM cache must use the local huggingface cache.")
    if settings.get("bind_host") != public_host:
        raise HTTPException(
            status_code=422, detail="Remote LLM bind host must equal the instruction server public host."
        )
    try:
        validated = validate_llm_settings(settings, remote=True)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    validated["bind_host"] = public_host
    return validated


def create_app(controller: ServiceController | None = None, *, public_host: str | None = None) -> FastAPI:
    service_controller = controller or ServiceController(public_host=public_host or "127.0.0.1")
    controller_public_host = getattr(service_controller, "public_host", public_host or "127.0.0.1")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            service_controller.stop_all()

    app = FastAPI(title="Almost ARCADIA Instruction Server", lifespan=lifespan)
    app.state.controller = service_controller

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "instruction"}

    @app.post("/services/start", response_model=dict)
    def start_service(request: StartServiceRequest) -> dict[str, Any]:
        settings = _reject_remote_commands(
            request.settings, service_type=request.service_type, public_host=controller_public_host
        )
        try:
            endpoint = service_controller.start(
                ServiceSpec(service_type=request.service_type, port=request.port, settings=settings)
            )
        except (ServiceError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return endpoint.to_dict()

    @app.post("/services/stop")
    def stop_service(request: StopServiceRequest) -> dict[str, Any]:
        service_controller.stop(request.port)
        return {"stopped": True, "port": request.port}

    @app.get("/services")
    def list_services() -> list[dict[str, Any]]:
        return [status.to_dict() for status in service_controller.list_services()]

    @app.get("/services/{port}/logs", response_class=PlainTextResponse)
    def get_logs(port: int, tail: int = 200) -> str:
        if not 1 <= port <= 65535:
            raise HTTPException(status_code=422, detail="port must be between 1 and 65535")
        try:
            return service_controller.get_logs(port, tail=tail)
        except ServiceError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Almost ARCADIA instruction server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--public-host", default=None, help="Address clients should use for inference")
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()
    public_host = args.public_host or args.host
    controller = ServiceController(public_host=public_host, log_dir=args.log_dir)
    uvicorn.run(create_app(controller, public_host=public_host), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
