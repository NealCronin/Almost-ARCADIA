from __future__ import annotations

import argparse

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .controller import ServiceController
from .specs import ServiceSpec


class StartServiceRequest(BaseModel):
    service_type: str
    port: int = Field(ge=1, le=65535)
    settings: dict = Field(default_factory=dict)


class StopServiceRequest(BaseModel):
    port: int = Field(ge=1, le=65535)


def create_app(controller: ServiceController | None = None) -> FastAPI:
    app = FastAPI(title="Almost ARCADIA Instruction Server")
    service_controller = controller or ServiceController(public_host="127.0.0.1")

    @app.post("/services/start")
    def start_service(request: StartServiceRequest) -> dict:
        try:
            endpoint = service_controller.start(
                ServiceSpec(
                    service_type=request.service_type,  # type: ignore[arg-type]
                    port=request.port,
                    settings=request.settings,
                )
            )
            return endpoint.to_dict()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/services/stop")
    def stop_service(request: StopServiceRequest) -> dict[str, object]:
        service_controller.stop(request.port)
        return {"stopped": True, "port": request.port}

    @app.get("/services")
    def list_services() -> list[dict[str, object]]:
        return service_controller.list_services()

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    controller = ServiceController(public_host=args.host)
    uvicorn.run(create_app(controller), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
