from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from starlette.requests import Request

from core.errors import ArcadiaError, ServiceStartupError
from core.networking import validate_ipv4
from core.services.controller import ServiceController
from core.services.llm_settings import validate_additional_server_arguments, validate_llm_settings
from core.services.sam_checkpoint import SAMCheckpointStore
from core.services.specs import ServiceSpec
from core.storage import state_child


def _validate_remote_sam_settings(settings: dict[str, Any]) -> dict[str, Any]:
    allowed = {"checkpoint", "bind_host", "confidence", "extra_args", "startup_timeout"}
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError(f"Unknown remote SAM3 settings: {', '.join(sorted(unknown))}.")
    checkpoint = str(settings.get("checkpoint", "")).strip()
    if not checkpoint:
        raise ValueError("SAM3 checkpoint is required.")
    checkpoint = str(SAMCheckpointStore.validate_checkpoint_path(checkpoint))
    confidence = float(settings.get("confidence", 0.25))
    if not 0 <= confidence <= 1:
        raise ValueError("SAM3 confidence must be between 0 and 1.")
    return {
        "checkpoint": checkpoint,
        "bind_host": validate_ipv4(str(settings.get("bind_host", "127.0.0.1")), label="SAM3 bind host"),
        "confidence": confidence,
        "extra_args": validate_additional_server_arguments(settings.get("extra_args", [])),
        **({"startup_timeout": float(settings["startup_timeout"])} if "startup_timeout" in settings else {}),
    }


def create_app(controller: ServiceController | None = None, *, public_host: str = "127.0.0.1"):
    from fastapi import FastAPI, Header, HTTPException, Query

    service_controller = controller or ServiceController(
        public_host=public_host, log_dir=state_child("logs") / "instruction"
    )
    app = FastAPI(title="Almost ARCADIA instruction server")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "instruction"}

    @app.get("/services")
    def services() -> list[dict[str, Any]]:
        return [status.to_dict() for status in service_controller.list_services()]

    @app.post("/services/start")
    def start_service(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            spec = ServiceSpec.from_dict(payload)
            if spec.service_type in ("llm", "visual_llm"):
                spec.settings = validate_llm_settings(spec.settings, remote=True)
            else:
                spec.settings = _validate_remote_sam_settings(spec.settings)
            endpoint = service_controller.start(spec)
            return endpoint.to_dict()
        except ServiceStartupError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except (ArcadiaError, ValueError, OSError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/services/stop")
    def stop_service(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            port = int(payload["port"])
            service_controller.stop(port)
            return {"stopped": port}
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="A valid port is required.") from exc

    @app.get("/services/{port}/logs")
    def logs(port: int, tail: int = Query(default=200, ge=1, le=5000)) -> str:
        try:
            return service_controller.get_logs(port, tail=tail)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/artifacts/sam3/checkpoint")
    async def upload_sam_checkpoint(
        request: Request,
        filename: str = Header(..., alias="X-Arcadia-Filename"),
    ) -> dict[str, Any]:
        try:
            raw_size = request.headers.get("content-length")
            expected_size = int(raw_size) if raw_size is not None else None
            return await SAMCheckpointStore.save_async_chunks(
                request.stream(),
                filename,
                expected_size=expected_size,
            )
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    app.state.controller = service_controller
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Almost ARCADIA instruction server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--public-host")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--log-dir", default=str(state_child("logs") / "instruction"))
    args = parser.parse_args()

    host = validate_ipv4(args.host, label="Instruction host")
    public_host = validate_ipv4(args.public_host or host, label="Public host")
    controller = ServiceController(public_host=public_host, log_dir=Path(args.log_dir))
    import uvicorn

    uvicorn.run(create_app(controller, public_host=public_host), host=host, port=args.port)


if __name__ == "__main__":
    main()
