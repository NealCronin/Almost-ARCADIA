from __future__ import annotations

import copy
import threading
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote

from django.contrib import messages
from django.http import FileResponse, HttpRequest, HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from core.config import ConfiguredService
from core.errors import ArcadiaError
from core.inference.llm_client import LLMClient
from core.services.host_listener import HostListenerError, HostListenerRestartError
from core.services.instruction_client import InstructionClient
from core.services.specs import ServiceEndpoint
from web.artifacts import ArtifactStore
from web.forms import (
    AnalysisForm,
    EndpointTestForm,
    HostListenerForm,
    LLMServiceForm,
    PipelineForm,
    SAMServiceForm,
)
from web.runtime import get_runtime
from web.tools import TOOLS

_host_listener_save_lock = threading.RLock()


def _service_form(name: str, config: Any, data: Any = None):
    form_classes = {"llm": LLMServiceForm, "sam3": SAMServiceForm}
    if name not in TOOLS["priority-map"].required_services:
        raise ValueError(f"Unknown Priority Map service {name}.")
    form_class = form_classes[name]
    form = (
        form_class(data=data, nodes=config.nodes, auto_id=f"{name}_%s")
        if data is not None
        else form_class(nodes=config.nodes, auto_id=f"{name}_%s")
    )
    form.initial_from(config.priority_map.services.get(name))
    return form


def home(request: HttpRequest) -> HttpResponse:
    runtime = get_runtime()
    config = runtime.config_store.load()
    return render(
        request,
        "web/home.html",
        {"config": config, "services": runtime.controller.list_services(), "analysis": runtime.analysis.status()},
    )


@require_GET
def client_portal(request: HttpRequest) -> HttpResponse:
    runtime = get_runtime()
    return render(
        request,
        "web/client.html",
        {"analysis": runtime.analysis.status()},
    )


@require_GET
def services(request: HttpRequest) -> HttpResponse:
    runtime = get_runtime()
    config = runtime.config_store.load()
    context = {
        "config": config,
        "llm_form": _service_form("llm", config),
        "sam_form": _service_form("sam3", config),
        "services": runtime.controller.list_services(),
        "log_port": request.GET.get("log_port"),
        "log_text": None,
    }
    if request.GET.get("log_port"):
        try:
            port = int(request.GET["log_port"])
            context["log_text"] = runtime.controller.get_logs(port)
        except (ValueError, ArcadiaError) as exc:
            context["log_text"] = str(exc)
    return render(request, "web/services.html", context)


@require_POST
def start_service(request: HttpRequest, service_name: str) -> HttpResponse:
    if service_name not in ("llm", "sam3"):
        return HttpResponse("Unknown service", status=404)
    runtime = get_runtime()
    try:
        runtime.analysis.assert_configuration_mutable()
        config = runtime.config_store.load()
        form = _service_form(service_name, config, request.POST)
        if not form.is_valid():
            return render(
                request,
                "web/services.html",
                {
                    "config": config,
                    "llm_form": form if service_name == "llm" else _service_form("llm", config),
                    "sam_form": form if service_name == "sam3" else _service_form("sam3", config),
                    "services": runtime.controller.list_services(),
                },
                status=400,
            )
        spec = form.to_spec()
        previous = config.priority_map.services.get(service_name)
        config.priority_map.services[service_name] = ConfiguredService(
            node=form.cleaned_data["node"], spec=spec, extra=previous.extra if previous else {}
        )
        runtime.config_store.save(config)
        messages.success(
            request, f"Saved {service_name.upper()} settings. Priority Map starts configured services for a run."
        )
    except (ArcadiaError, ValueError, OSError) as exc:
        messages.error(request, str(exc))
    return redirect("services")


@require_POST
def stop_service(request: HttpRequest, service_name: str) -> HttpResponse:
    runtime = get_runtime()
    try:
        runtime.analysis.assert_configuration_mutable()
        config = runtime.config_store.load()
        configured = config.priority_map.services.get(service_name)
        if configured is None:
            raise ArcadiaError(f"No configuration exists for {service_name}.")
        node = config.nodes[configured.node]
        if node.mode == "local":
            runtime.controller.stop(configured.port)
        else:
            InstructionClient(node.host, node.instruction_port or 9000).stop_service(configured.port)
        messages.success(request, f"Stopped {service_name} on port {configured.port}.")
    except (ArcadiaError, ValueError, OSError) as exc:
        messages.error(request, str(exc))
    return redirect("services")


@require_GET
def service_logs(request: HttpRequest, port: int) -> HttpResponse:
    runtime = get_runtime()
    try:
        config = runtime.config_store.load()
        configured = next((service for service in config.priority_map.services.values() if service.port == port), None)
        if configured is None:
            return HttpResponse(f"No configured service on port {port}.", status=404)
        node = config.nodes[configured.node]
        if node.mode == "local":
            logs = runtime.controller.get_logs(port)
        else:
            logs = InstructionClient(node.host, node.instruction_port or 9000).get_logs(port)
        return HttpResponse(logs, content_type="text/plain")
    except (ArcadiaError, ValueError, OSError) as exc:
        return HttpResponse(str(exc), status=502, content_type="text/plain")


@require_GET
def nodes(request: HttpRequest) -> HttpResponse:
    runtime = get_runtime()
    config = runtime.config_store.load()
    return render(
        request,
        "web/nodes.html",
        {
            "form": HostListenerForm(initial=config.host_listener.to_dict()),
            "listener": runtime.host_listener.status(),
        },
    )


@require_POST
def save_host_listener(request: HttpRequest) -> HttpResponse:
    runtime = get_runtime()
    form = HostListenerForm(request.POST)
    if not form.is_valid():
        return render(
            request,
            "web/nodes.html",
            {"form": form, "listener": runtime.host_listener.status()},
            status=400,
        )
    with _host_listener_save_lock:
        config = runtime.config_store.load()
        previous = config.host_listener
        replacement = form.to_config()
        replacement.extra = copy.deepcopy(previous.extra)
        try:
            runtime.host_listener.restart(replacement, rollback_config=previous)
        except HostListenerRestartError as exc:
            form.add_error(None, str(exc))
            return render(
                request,
                "web/nodes.html",
                {"form": form, "listener": runtime.host_listener.status()},
                status=502,
            )
        except HostListenerError as exc:
            form.add_error(None, f"Instruction server was not changed: {exc}")
            return render(
                request,
                "web/nodes.html",
                {"form": form, "listener": runtime.host_listener.status()},
                status=400,
            )
        config.host_listener = replacement
        try:
            runtime.config_store.save(config)
        except OSError as exc:
            try:
                runtime.host_listener.restart(previous, rollback_config=replacement)
            except HostListenerRestartError as rollback_exc:
                form.add_error(None, f"Configuration was not saved: {exc}. Listener rollback failed: {rollback_exc}")
            else:
                form.add_error(None, f"Configuration was not saved: {exc}. The previous listener was restored.")
            return render(
                request,
                "web/nodes.html",
                {"form": form, "listener": runtime.host_listener.status()},
                status=500,
            )
    messages.success(request, f"Instruction server is listening on {replacement.host}:{replacement.port}.")
    return redirect("host")


@require_GET
def host_listener_status(request: HttpRequest) -> JsonResponse:
    return JsonResponse(get_runtime().host_listener.status().to_dict())


@require_GET
def analysis_page(request: HttpRequest) -> HttpResponse:
    runtime = get_runtime()
    config = runtime.config_store.load()
    return render(
        request,
        "web/analysis.html",
        {
            "config": config,
            "pipeline_form": PipelineForm.from_config(config.priority_map.pipeline),
            "analysis_form": AnalysisForm(),
            "analysis": runtime.analysis.status(),
        },
    )


@require_POST
def save_pipeline(request: HttpRequest) -> HttpResponse:
    runtime = get_runtime()
    try:
        runtime.analysis.assert_configuration_mutable()
        config = runtime.config_store.load()
        form = PipelineForm(request.POST)
        if not form.is_valid():
            return render(
                request,
                "web/analysis.html",
                {
                    "config": config,
                    "pipeline_form": form,
                    "analysis_form": AnalysisForm(),
                    "analysis": runtime.analysis.status(),
                },
                status=400,
            )
        config.priority_map.pipeline = form.to_config()
        runtime.config_store.save(config)
        messages.success(request, "Saved pipeline settings.")
    except (ArcadiaError, ValueError) as exc:
        messages.error(request, str(exc))
    return redirect("analysis")


@require_POST
def start_analysis(request: HttpRequest) -> HttpResponse:
    runtime = get_runtime()
    form = AnalysisForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Choose exactly one input path or retained upload before starting the analysis.")
        return redirect("analysis")
    try:
        if form.cleaned_data["upload_id"]:
            input_path = runtime.uploads.input_path(form.cleaned_data["upload_id"])
        else:
            input_path = Path(form.cleaned_data["input_path"]).expanduser().resolve(strict=True)
            if not input_path.is_file() and not input_path.is_dir():
                raise ValueError("Analysis input must be a regular file or directory.")
        config = runtime.config_store.load()
        runtime.analysis.start(input_path, config)
        messages.success(request, "Analysis started.")
    except (ArcadiaError, ValueError, OSError) as exc:
        messages.error(request, str(exc))
    return redirect("results")


@require_http_methods(["GET", "POST"])
def uploads(request: HttpRequest) -> JsonResponse:
    runtime = get_runtime()
    if request.method == "GET":
        return JsonResponse({"uploads": [_upload_payload(manifest) for manifest in runtime.uploads.list()]})
    try:
        manifest = runtime.uploads.create(request.FILES.getlist("files"), request.POST.getlist("relative_paths"))
    except (ArcadiaError, ValueError, OSError) as exc:
        return JsonResponse({"detail": str(exc)}, status=400)
    return JsonResponse({"upload": _upload_payload(manifest)}, status=201)


@require_POST
def delete_upload(request: HttpRequest, upload_id: str) -> JsonResponse:
    runtime = get_runtime()
    try:
        input_path = runtime.uploads.input_path(upload_id)
        analysis = runtime.analysis
        active_input = analysis.status().input_path
        if analysis.is_active() and active_input and input_path.resolve() == Path(active_input).resolve():
            return JsonResponse({"detail": "Cannot delete the upload used by the active run."}, status=409)
        runtime.uploads.delete(upload_id)
    except FileNotFoundError:
        return JsonResponse({"detail": "Upload not found."}, status=404)
    except (ArcadiaError, ValueError, OSError) as exc:
        return JsonResponse({"detail": str(exc)}, status=400)
    return JsonResponse({"deleted": upload_id})


def _upload_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    upload_id = str(manifest["id"])
    return {
        "id": upload_id,
        "source_type": manifest["source_type"],
        "file_count": manifest["file_count"],
        "size_bytes": manifest["size_bytes"],
        "created_at": manifest["created_at"],
        "delete_url": f"/client/priority-map/uploads/{upload_id}/delete/",
    }


@require_GET
def analysis_status(request: HttpRequest) -> JsonResponse:
    return JsonResponse(get_runtime().analysis.status().to_dict())


@require_POST
def cancel_run(request: HttpRequest, run_id: str) -> JsonResponse:
    analysis = get_runtime().analysis
    status = analysis.status()
    if run_id != status.run_id or not analysis.is_active():
        return JsonResponse({"detail": "Active run not found."}, status=404)
    try:
        return JsonResponse(analysis.cancel_after_current_frame().to_dict())
    except ArcadiaError as exc:
        return JsonResponse({"detail": str(exc)}, status=409)


@require_GET
def run_stream(request: HttpRequest, run_id: str) -> HttpResponse:
    analysis = get_runtime().analysis
    status = analysis.status()
    if run_id != status.run_id or status.state == "idle":
        return HttpResponse("Run stream not found.", status=404)

    def frames() -> Iterator[bytes]:
        version = 0
        while True:
            jpeg, newest_version, state = analysis.preview(version)
            if jpeg is not None and newest_version > version:
                version = newest_version
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n" + f"Content-Length: {len(jpeg)}\r\n\r\n".encode() + jpeg + b"\r\n"
                )
                continue
            if state in ("completed", "cancelled", "failed"):
                return

    response = StreamingHttpResponse(
        frames(),
        content_type=f"{TOOLS['priority-map'].presentation.stream_content_type}; boundary=frame",
    )
    response["Cache-Control"] = "no-store"
    return response


@require_GET
def run_artifacts(request: HttpRequest, run_id: str) -> JsonResponse:
    runtime = get_runtime()
    try:
        store = ArtifactStore(runtime.config_store.load().priority_map.output.root, run_id)
        artifacts = [
            {
                "path": record.path,
                "size_bytes": record.size_bytes,
                "content_type": record.content_type,
                "inline_url": f"/client/priority-map/runs/{run_id}/artifacts/{quote(record.path)}/",
                "download_url": f"/client/priority-map/runs/{run_id}/artifacts/{quote(record.path)}/?download=1",
            }
            for record in store.list()
        ]
    except (ArcadiaError, OSError) as exc:
        return JsonResponse({"detail": str(exc)}, status=404)
    payload: dict[str, Any] = {"run_id": run_id, "artifacts": artifacts}
    for artifact in artifacts:
        if artifact["path"] == "effective_settings.json":
            payload["effective_settings"] = artifact
        elif artifact["path"] == "analysis.log":
            payload["log"] = artifact
    return JsonResponse(payload)


@require_GET
def run_artifact(request: HttpRequest, run_id: str, artifact_path: str) -> HttpResponse:
    runtime = get_runtime()
    try:
        store = ArtifactStore(runtime.config_store.load().priority_map.output.root, run_id)
        artifact = store.resolve(artifact_path)
    except (ArcadiaError, OSError) as exc:
        return HttpResponse(str(exc), status=404)
    extension = artifact.suffix.lower()
    inline = extension in TOOLS["priority-map"].presentation.inline_artifact_extensions
    download = request.GET.get("download") == "1"
    return FileResponse(
        artifact.open("rb"),
        as_attachment=download or not inline,
        filename=artifact.name,
        content_type=next((record.content_type for record in store.list() if record.path == artifact_path), None),
    )


@require_GET
def results(request: HttpRequest) -> HttpResponse:
    return render(request, "web/results.html", {"analysis": get_runtime().analysis.status()})


@require_GET
def endpoint_test(request: HttpRequest) -> HttpResponse:
    return render(request, "web/endpoint_test.html", {"form": EndpointTestForm()})


@require_POST
def run_endpoint_test(request: HttpRequest) -> HttpResponse:
    form = EndpointTestForm(request.POST)
    response_text = None
    if form.is_valid():
        try:
            endpoint = ServiceEndpoint(form.cleaned_data["endpoint_host"], form.cleaned_data["endpoint_port"], "llm")
            response_text = LLMClient(endpoint).chat(form.cleaned_data["prompt"]).text
        except (ArcadiaError, ValueError, OSError) as exc:
            form.add_error(None, str(exc))
    return render(request, "web/endpoint_test.html", {"form": form, "response_text": response_text})
