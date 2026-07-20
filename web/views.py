from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
from typing import Any, Iterator, cast
from urllib.parse import quote

import cv2
import numpy as np
from django.contrib import messages
from django.http import FileResponse, HttpRequest, HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from core.config import AppConfig, ConfiguredService, NodeConfig
from core.errors import ArcadiaError, ConfigurationError, InferenceError, ServiceError
from core.inference.llm_client import LLMClient
from core.inference.sam_client import SAMClient
from core.services.host_listener import HostListenerError, HostListenerRestartError
from core.services.instruction_client import InstructionClient
from core.services.llm_runtime import LLMRuntime
from core.services.llm_settings import PROJECTOR_RE, generation_settings, parse_hf_source, validate_llm_settings
from core.services.sam_checkpoint import SAMCheckpointStore
from core.services.specs import ServiceEndpoint, ServiceSpec, ServiceType
from web.artifacts import ArtifactStore
from web.forms import (
    AnalysisForm,
    EndpointTestForm,
    HostListenerForm,
    HostSAMCheckpointForm,
    LLMServiceForm,
    PipelineForm,
    RemoteNodeForm,
    SAMServiceForm,
    VisualLLMServiceForm,
)
from web.runtime import get_runtime
from web.tools import TOOLS

_host_listener_save_lock = threading.RLock()
_models_node_lock = threading.RLock()
_SERVICE_LABELS = {"llm": "Logical LLM", "visual_llm": "Visual LLM", "sam3": "SAM3"}
_ALLOWED_IMAGES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
_ALLOWED_SAM_IMAGES = {"image/jpeg", "image/png", "image/webp"}
_IMAGE_LIMIT = 10 * 1024 * 1024


def _label(name: str) -> str:
    return _SERVICE_LABELS.get(name, name.replace("_", " ").title())


def _service_form(name: str, config: AppConfig, data: Any = None):
    classes = {"llm": LLMServiceForm, "visual_llm": VisualLLMServiceForm, "sam3": SAMServiceForm}
    if name not in classes:
        raise ValueError(f"Unknown Priority Map service {name!r}.")
    form = (
        classes[name](data=data, nodes=config.nodes, auto_id=f"{name}_%s")
        if data is not None
        else classes[name](nodes=config.nodes, auto_id=f"{name}_%s")
    )
    form.initial_from(config.priority_map.services.get(name))
    if name == "sam3" and "checkpoint" not in form.initial and config.host_listener.sam3_checkpoint:
        form.initial["checkpoint"] = config.host_listener.sam3_checkpoint
    return form


def _models_context(
    request: HttpRequest,
    config: AppConfig,
    *,
    llm_form: LLMServiceForm | None = None,
    visual_llm_form: VisualLLMServiceForm | None = None,
    sam_form: SAMServiceForm | None = None,
    node_form: RemoteNodeForm | None = None,
    editing_node: str | None = None,
    edit_node_form: RemoteNodeForm | None = None,
    allow_save_anyway: bool = False,
    node_error: str | None = None,
) -> dict[str, Any]:
    runtime = get_runtime()
    context: dict[str, Any] = {
        "config": config,
        "llm_form": llm_form or _service_form("llm", config),
        "visual_llm_form": visual_llm_form or _service_form("visual_llm", config),
        "sam_form": sam_form or _service_form("sam3", config),
        "visual_llm_mode": config.priority_map.visual_llm_mode,
        "visual_llm_readonly": config.priority_map.visual_llm_mode == "same_as_logical",
        "nodes": sorted(config.nodes.items(), key=lambda item: (item[0] != "local", item[0].lower())),
        "node_hosts": {name: node.host for name, node in config.nodes.items()},
        "node_form": node_form or RemoteNodeForm(),
        "editing_node": editing_node,
        "edit_node_form": edit_node_form,
        "allow_save_anyway": allow_save_anyway,
        "node_error": node_error,
        "services": runtime.controller.list_services(),
        "log_port": request.GET.get("log_port"),
        "log_text": None,
    }
    if request.GET.get("log_port"):
        try:
            context["log_text"] = runtime.controller.get_logs(int(request.GET["log_port"]))
        except (ValueError, ArcadiaError) as exc:
            context["log_text"] = str(exc)
    return context


def _render_models(request: HttpRequest, config: AppConfig, *, status: int = 200, **kwargs: Any) -> HttpResponse:
    return render(request, "web/services.html", _models_context(request, config, **kwargs), status=status)


def home(request: HttpRequest) -> HttpResponse:
    runtime = get_runtime()
    return render(
        request,
        "web/home.html",
        {
            "config": runtime.config_store.load(),
            "services": runtime.controller.list_services(),
            "analysis": runtime.analysis.status(),
        },
    )


@require_GET
def client_portal(request: HttpRequest) -> HttpResponse:
    return render(request, "web/client.html", {"analysis": get_runtime().analysis.status()})


@require_GET
def services(request: HttpRequest) -> HttpResponse:
    runtime = get_runtime()
    config = runtime.config_store.load()
    editing = request.GET.get("edit")
    edit_form = None
    if editing and editing != "local" and editing in config.nodes:
        node = config.nodes[editing]
        edit_form = RemoteNodeForm(
            initial={"name": editing, "host": node.host, "instruction_port": node.instruction_port}
        )
    return _render_models(request, config, editing_node=editing, edit_node_form=edit_form)


@require_POST
def inspect_llm_repository(request: HttpRequest) -> JsonResponse:
    try:
        payload = json.loads(request.body)
        if not isinstance(payload, dict):
            raise ValueError("JSON object required.")
        source = parse_hf_source(str(payload.get("source", "")))
        files = LLMRuntime.list_repository_files(source.repo_id, revision=source.revision)
        candidates = sorted(
            [
                item
                for item in files
                if item.lower().endswith(".gguf")
                and bool(PROJECTOR_RE.search(Path(item).name)) == bool(payload.get("projector", False))
            ],
            key=str.casefold,
        )
        if source.filename:
            exact_matches = [item for item in candidates if item.casefold() == source.filename.casefold()]
            if len(exact_matches) != 1:
                kind = "projector" if payload.get("projector", False) else "model"
                raise ValueError(
                    f"{source.filename!r} is not a usable {kind} GGUF in {source.repo_id}@{source.revision}."
                )
            message = f"Exact file found: {exact_matches[0]}"
        elif len(candidates) == 1:
            message = f"One usable GGUF found: {candidates[0]}"
        elif not candidates:
            message = "No usable GGUF files were found."
        else:
            preview = ", ".join(candidates[:8])
            message = f"{len(candidates)} usable GGUF files found. Paste an exact file link. Candidates: {preview}"
        return JsonResponse({"files": candidates[:50], "message": message, "ambiguous": len(candidates) != 1})
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return JsonResponse({"error": str(exc)}, status=400)


def _remote_node_reachable(node: NodeConfig) -> bool:
    return InstructionClient(node.host, node.instruction_port or 9000, timeout=2.0, retries=0).health()


def _node_failure(
    request: HttpRequest,
    config: AppConfig,
    form: RemoteNodeForm,
    *,
    editing: str | None = None,
    save_anyway: bool = False,
    status: int = 400,
) -> HttpResponse:
    return _render_models(
        request,
        config,
        node_form=form if editing is None else None,
        editing_node=editing,
        edit_node_form=form if editing is not None else None,
        allow_save_anyway=save_anyway,
        status=status,
    )


@require_POST
def add_remote_node(request: HttpRequest) -> HttpResponse:
    runtime = get_runtime()
    form = RemoteNodeForm(request.POST)
    if not form.is_valid():
        return _node_failure(request, runtime.config_store.load(), form)
    with runtime.config_lock, _models_node_lock:
        config = runtime.config_store.load()
        try:
            runtime.analysis.assert_configuration_mutable()
            name = form.cleaned_data["name"]
            if name in config.nodes:
                form.add_error("name", f"A compute node named '{name}' already exists.")
                return _node_failure(request, config, form)
            node = form.to_config()
            reachable = _remote_node_reachable(node)
            if not reachable and request.POST.get("save_anyway") != "1":
                form.add_error(None, "Instruction server is unreachable. Test the address or choose Save anyway.")
                return _node_failure(request, config, form, save_anyway=True)
            config.nodes[name] = node
            runtime.config_store.save(config)
        except (ArcadiaError, OSError) as exc:
            form.add_error(None, f"Could not save remote computer: {exc}")
            return _node_failure(request, config, form, status=500)
    messages.success(request, f"Saved remote computer '{name}'.")
    return redirect("priority_map_models")


@require_POST
def edit_remote_node(request: HttpRequest, node_name: str) -> HttpResponse:
    runtime = get_runtime()
    form = RemoteNodeForm(request.POST)
    with runtime.config_lock, _models_node_lock:
        config = runtime.config_store.load()
        previous = config.nodes.get(node_name)
        if previous is None:
            return HttpResponse("Unknown compute node.", status=404)
        if node_name == "local" or previous.mode != "remote":
            return _node_failure(request, config, form, editing=node_name, status=400)
        if not form.is_valid():
            return _node_failure(request, config, form, editing=node_name)
        try:
            runtime.analysis.assert_configuration_mutable()
            name = form.cleaned_data["name"]
            if name in config.nodes and name != node_name:
                form.add_error("name", f"A compute node named '{name}' already exists.")
                return _node_failure(request, config, form, editing=node_name)
            replacement = form.to_config(extra=copy.deepcopy(previous.extra))
            reachable = _remote_node_reachable(replacement)
            if not reachable and request.POST.get("save_anyway") != "1":
                form.add_error(None, "Instruction server is unreachable. Test the address or choose Save anyway.")
                return _node_failure(request, config, form, editing=node_name, save_anyway=True)
            del config.nodes[node_name]
            config.nodes[name] = replacement
            if name != node_name:
                for configured in config.priority_map.services.values():
                    if configured.node == node_name:
                        configured.node = name
            runtime.config_store.save(config)
        except (ArcadiaError, OSError) as exc:
            form.add_error(None, f"Could not save remote computer: {exc}")
            return _node_failure(request, config, form, editing=node_name, status=500)
    messages.success(request, f"Saved remote computer '{name}'.")
    return redirect("priority_map_models")


@require_POST
def delete_remote_node(request: HttpRequest, node_name: str) -> HttpResponse:
    runtime = get_runtime()
    with runtime.config_lock, _models_node_lock:
        config = runtime.config_store.load()
        try:
            runtime.analysis.assert_configuration_mutable()
            node = config.nodes.get(node_name)
            if node is None:
                return HttpResponse("Unknown compute node.", status=404)
            if node_name == "local":
                return _render_models(request, config, node_error="This computer cannot be deleted.", status=400)
            references = [_label(name) for name, item in config.priority_map.services.items() if item.node == node_name]
            if references:
                return _render_models(
                    request,
                    config,
                    node_error=f"Cannot delete '{node_name}'; it is used by {', '.join(references)}.",
                    status=409,
                )
            del config.nodes[node_name]
            runtime.config_store.save(config)
        except (ArcadiaError, OSError) as exc:
            return _render_models(request, config, node_error=str(exc), status=500)
    messages.success(request, f"Deleted remote computer '{node_name}'.")
    return redirect("priority_map_models")


@require_POST
def test_remote_node(request: HttpRequest, node_name: str) -> JsonResponse:
    config = get_runtime().config_store.load()
    node = config.nodes.get(node_name)
    if node is None:
        return JsonResponse({"state": "unknown", "message": "Unknown compute node."}, status=404)
    if node.mode == "local":
        return JsonResponse({"state": "local", "message": "This computer does not need remote testing."})
    reachable = _remote_node_reachable(node)
    return JsonResponse(
        {
            "state": "reachable" if reachable else "unreachable",
            "message": "Instruction server is reachable." if reachable else "Instruction server is unreachable.",
        }
    )


def _collision(config: AppConfig, name: str, candidate: ConfiguredService) -> str | None:
    for other_name, other in config.priority_map.services.items():
        if other_name == name:
            continue
        if other.node == candidate.node and other.port == candidate.port:
            if {name, other_name} == {"llm", "visual_llm"} and config.priority_map.visual_llm_mode == "same_as_logical":
                continue
            return f"{_label(name)} and {_label(other_name)} cannot use the same port on the same compute node."
    return None


@require_POST
def start_service(request: HttpRequest, service_name: str) -> HttpResponse:
    if service_name not in ("llm", "visual_llm", "sam3"):
        return HttpResponse("Unknown service.", status=404)
    runtime = get_runtime()
    with runtime.config_lock:
        config = runtime.config_store.load()
        try:
            runtime.analysis.assert_configuration_mutable()
            if service_name == "visual_llm" and "visual_llm_mode" in request.POST:
                mode = request.POST.get("visual_llm_mode")
                if mode not in ("same_as_logical", "separate"):
                    raise ValueError("Unknown Visual LLM mode.")
                if mode == "separate":
                    logical = config.priority_map.services.get("llm")
                    visual = config.priority_map.services.get("visual_llm")
                    if logical and visual and logical.node == visual.node and logical.port == visual.port:
                        raise ValueError(
                            "Logical and separate Visual LLM cannot use the same port on the same compute node."
                        )
                config.priority_map.visual_llm_mode = mode
                runtime.config_store.save(config)
                messages.success(request, "Saved Visual LLM mode.")
                return redirect("priority_map_models")

            form = _service_form(service_name, config, request.POST)
            if not form.is_valid():
                return _render_models(
                    request,
                    config,
                    llm_form=form if service_name == "llm" else None,
                    visual_llm_form=form if service_name == "visual_llm" else None,
                    sam_form=form if service_name == "sam3" else None,
                    status=400,
                )
            try:
                spec = form.to_spec()
            except (ConfigurationError, ValueError) as exc:
                form.add_error(None, str(exc))
                return _render_models(
                    request,
                    config,
                    llm_form=form if service_name == "llm" else None,
                    visual_llm_form=form if service_name == "visual_llm" else None,
                    sam_form=form if service_name == "sam3" else None,
                    status=400,
                )
            if spec.service_type != service_name:
                spec = ServiceSpec(cast(ServiceType, service_name), spec.port, spec.settings)
            previous = config.priority_map.services.get(service_name)
            candidate = ConfiguredService(
                node=form.cleaned_data["node"],
                spec=spec,
                extra=copy.deepcopy(previous.extra) if previous else {},
            )
            collision = _collision(config, service_name, candidate)
            if collision:
                form.add_error(None, collision)
                return _render_models(
                    request,
                    config,
                    llm_form=form if service_name == "llm" else None,
                    visual_llm_form=form if service_name == "visual_llm" else None,
                    sam_form=form if service_name == "sam3" else None,
                    status=409,
                )
            config.priority_map.services[service_name] = candidate
            runtime.config_store.save(config)
            messages.success(request, f"Saved {_label(service_name)} settings.")
        except (ArcadiaError, ValueError, OSError) as exc:
            messages.error(request, str(exc))
    return redirect("priority_map_models")


@require_POST
def stop_service(request: HttpRequest, service_name: str) -> HttpResponse:
    runtime = get_runtime()
    try:
        runtime.analysis.assert_configuration_mutable()
        config = runtime.config_store.load()
        backing = (
            "llm"
            if service_name == "visual_llm" and config.priority_map.visual_llm_mode == "same_as_logical"
            else service_name
        )
        configured = config.priority_map.services.get(backing)
        if configured is None:
            raise ArcadiaError(f"No configuration exists for {_label(backing)}.")
        node = config.nodes[configured.node]
        if node.mode == "local":
            runtime.controller.stop(configured.port)
        else:
            InstructionClient(node.host, node.instruction_port or 9000).stop_service(configured.port)
        messages.success(request, f"Stopped {_label(backing)} on port {configured.port}.")
    except (ArcadiaError, ValueError, OSError) as exc:
        messages.error(request, str(exc))
    return redirect("priority_map_models")


@require_GET
def service_logs(request: HttpRequest, port: int) -> HttpResponse:
    runtime = get_runtime()
    try:
        if runtime.controller.is_running(port):
            return HttpResponse(runtime.controller.get_logs(port), content_type="text/plain")
        config = runtime.config_store.load()
        matches = [item for item in config.priority_map.services.values() if item.port == port]
        if len(matches) != 1:
            return HttpResponse("Configured service log is missing or ambiguous.", status=404)
        configured = matches[0]
        node = config.nodes[configured.node]
        logs = (
            runtime.controller.get_logs(port)
            if node.mode == "local"
            else InstructionClient(node.host, node.instruction_port or 9000).get_logs(port)
        )
        return HttpResponse(logs, content_type="text/plain")
    except (ArcadiaError, ValueError, OSError) as exc:
        return HttpResponse(str(exc), status=502, content_type="text/plain")


@require_GET
def nodes(request: HttpRequest) -> HttpResponse:
    runtime = get_runtime()
    config = runtime.config_store.load()
    checkpoint = config.host_listener.sam3_checkpoint
    checkpoint_path = Path(checkpoint).expanduser() if checkpoint else None
    checkpoint_status = (
        "No SAM3 checkpoint is configured."
        if checkpoint_path is None
        else "Configured checkpoint is ready."
        if checkpoint_path.is_file() and checkpoint_path.suffix.lower() == ".pt"
        else "Configured checkpoint is missing or is not a .pt file."
    )
    return render(
        request,
        "web/nodes.html",
        {
            "form": HostListenerForm(initial=config.host_listener.to_dict()),
            "sam_checkpoint_form": HostSAMCheckpointForm(initial={"checkpoint": checkpoint}),
            "sam_checkpoint_status": checkpoint_status,
            "sam_checkpoint_ready": checkpoint_path is not None
            and checkpoint_path.is_file()
            and checkpoint_path.suffix.lower() == ".pt",
            "listener": runtime.host_listener.status(),
        },
    )


@require_POST
def save_host_listener(request: HttpRequest) -> HttpResponse:
    runtime = get_runtime()
    form = HostListenerForm(request.POST)
    if not form.is_valid():
        return render(request, "web/nodes.html", {"form": form, "listener": runtime.host_listener.status()}, status=400)
    with runtime.config_lock, _host_listener_save_lock:
        config = runtime.config_store.load()
        previous = config.host_listener
        replacement = form.to_config()
        replacement.sam3_checkpoint = previous.sam3_checkpoint
        replacement.extra = copy.deepcopy(previous.extra)
        try:
            runtime.host_listener.restart(replacement, rollback_config=previous)
            config.host_listener = replacement
            runtime.config_store.save(config)
        except (HostListenerError, HostListenerRestartError, OSError) as exc:
            form.add_error(None, str(exc))
            return render(
                request, "web/nodes.html", {"form": form, "listener": runtime.host_listener.status()}, status=502
            )
    messages.success(request, f"Instruction server is listening on {replacement.host}:{replacement.port}.")
    return redirect("host")


@require_POST
def save_host_sam3_checkpoint(request: HttpRequest) -> HttpResponse:
    runtime = get_runtime()
    form = HostSAMCheckpointForm(request.POST)
    if not form.is_valid():
        config = runtime.config_store.load()
        return render(
            request,
            "web/nodes.html",
            {
                "form": HostListenerForm(initial=config.host_listener.to_dict()),
                "sam_checkpoint_form": form,
                "sam_checkpoint_status": "Checkpoint configuration is invalid.",
                "sam_checkpoint_ready": False,
                "listener": runtime.host_listener.status(),
            },
            status=400,
        )
    with runtime.config_lock:
        config = runtime.config_store.load()
        config.host_listener.sam3_checkpoint = form.cleaned_data["checkpoint"]
        runtime.config_store.save(config)
    messages.success(request, "SAM3 checkpoint configuration saved.")
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
            "uploads": runtime.uploads.list(),
        },
    )


@require_POST
def save_pipeline(request: HttpRequest) -> HttpResponse:
    runtime = get_runtime()
    with runtime.config_lock:
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
                    "uploads": runtime.uploads.list(),
                },
                status=400,
            )
        try:
            runtime.analysis.assert_configuration_mutable()
            config.priority_map.pipeline = form.to_config()
            runtime.config_store.save(config)
            messages.success(request, "Saved pipeline settings.")
        except (ArcadiaError, ValueError, OSError) as exc:
            messages.error(request, str(exc))
    return redirect("analysis")


@require_POST
def start_analysis(request: HttpRequest) -> HttpResponse:
    runtime = get_runtime()
    form = AnalysisForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Choose exactly one input path or retained upload.")
        return redirect("analysis")
    try:
        if form.cleaned_data["upload_id"]:
            input_path = runtime.uploads.input_path(form.cleaned_data["upload_id"])
        else:
            input_path = Path(form.cleaned_data["input_path"]).expanduser().resolve(strict=True)
            if not input_path.is_file() and not input_path.is_dir():
                raise ValueError("Analysis input must be a regular file or directory.")
        runtime.analysis.start(input_path, runtime.config_store.load())
        messages.success(request, "Analysis started.")
    except (ArcadiaError, ValueError, OSError) as exc:
        messages.error(request, str(exc))
    return redirect("results")


@require_http_methods(["GET", "POST"])
def uploads(request: HttpRequest) -> JsonResponse:
    runtime = get_runtime()
    if request.method == "GET":
        return JsonResponse({"uploads": [_upload_payload(item) for item in runtime.uploads.list()]})
    try:
        item = runtime.uploads.create(request.FILES.getlist("files"), request.POST.getlist("relative_paths"))
        return JsonResponse({"upload": _upload_payload(item)}, status=201)
    except (ArcadiaError, ValueError, OSError) as exc:
        return JsonResponse({"detail": str(exc)}, status=400)


@require_POST
def delete_upload(request: HttpRequest, upload_id: str) -> JsonResponse:
    runtime = get_runtime()
    try:
        active = runtime.analysis.status()
        input_path = runtime.uploads.input_path(upload_id)
        if (
            runtime.analysis.is_active()
            and active.input_path
            and Path(active.input_path).resolve() == input_path.resolve()
        ):
            return JsonResponse({"detail": "Cannot delete the upload used by the active run."}, status=409)
        runtime.uploads.delete(upload_id)
        return JsonResponse({"deleted": upload_id})
    except FileNotFoundError:
        return JsonResponse({"detail": "Upload not found."}, status=404)
    except (ArcadiaError, ValueError, OSError) as exc:
        return JsonResponse({"detail": str(exc)}, status=400)


def _upload_payload(item: dict[str, Any]) -> dict[str, Any]:
    upload_id = str(item["id"])
    return {
        "id": upload_id,
        "source_type": item["source_type"],
        "file_count": item["file_count"],
        "size_bytes": item["size_bytes"],
        "created_at": item["created_at"],
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
            jpeg, newest, state = analysis.preview(version)
            if jpeg is not None and newest > version:
                version = newest
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    + jpeg
                    + b"\r\n"
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
        return JsonResponse({"run_id": run_id, "artifacts": artifacts})
    except (ArcadiaError, OSError) as exc:
        return JsonResponse({"detail": str(exc)}, status=404)


@require_GET
def run_artifact(request: HttpRequest, run_id: str, artifact_path: str) -> HttpResponse:
    runtime = get_runtime()
    try:
        store = ArtifactStore(runtime.config_store.load().priority_map.output.root, run_id)
        artifact = store.resolve(artifact_path)
        extension = artifact.suffix.lower()
        inline = extension in TOOLS["priority-map"].presentation.inline_artifact_extensions
        return FileResponse(
            artifact.open("rb"),
            as_attachment=request.GET.get("download") == "1" or not inline,
            filename=artifact.name,
        )
    except (ArcadiaError, OSError) as exc:
        return HttpResponse(str(exc), status=404)


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


def _resolve_sam(config: AppConfig) -> tuple[ConfiguredService, NodeConfig, ServiceSpec]:
    configured = config.priority_map.services.get("sam3")
    if configured is None:
        raise ValueError("No saved SAM3 configuration exists.")
    node = config.nodes.get(configured.node)
    if node is None:
        raise ValueError(f"SAM3 references unknown node {configured.node!r}.")
    settings = copy.deepcopy(configured.settings)
    spec = ServiceSpec("sam3", configured.port, settings)
    return configured, node, spec


def _mask_overlay(
    image: np.ndarray,
    masks: list[Any],
    labels: list[str],
    confidences: list[float],
    boxes: list[Any],
) -> tuple[np.ndarray, int]:
    """Render SAM masks, contours, labels, and boxes onto a BGR image."""
    height, width = image.shape[:2]
    rendered = image.copy()
    colors = (
        np.array((44, 200, 255), dtype=np.float32),
        np.array((255, 168, 71), dtype=np.float32),
        np.array((120, 220, 120), dtype=np.float32),
        np.array((220, 120, 220), dtype=np.float32),
    )
    accepted = 0

    for index, raw_mask in enumerate(masks):
        mask = np.asarray(raw_mask)
        mask = np.squeeze(mask)
        if mask.ndim != 2 or mask.size == 0:
            continue
        if mask.shape != (height, width):
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        selected = mask > 0
        if not np.any(selected):
            continue

        color = colors[index % len(colors)]
        pixels = rendered[selected].astype(np.float32)
        rendered[selected] = np.clip((pixels * 0.52) + (color * 0.48), 0, 255).astype(np.uint8)

        binary = selected.astype(np.uint8) * 255
        contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rendered, contours, -1, tuple(int(value) for value in color), 2)

        label = labels[index] if index < len(labels) and labels[index].strip() else "segment"
        score = confidences[index] if index < len(confidences) else None
        caption = f"{label} {score:.2f}" if score is not None else label
        text_x, text_y = 8, 24 + (accepted * 24)

        if index < len(boxes):
            try:
                values = [float(value) for value in boxes[index][:4]]
            except (TypeError, ValueError, IndexError):
                values = []
            if len(values) == 4:
                x1, y1, x2, y2 = values
                x1_i = max(0, min(width - 1, int(round(x1))))
                y1_i = max(0, min(height - 1, int(round(y1))))
                x2_i = max(0, min(width - 1, int(round(x2))))
                y2_i = max(0, min(height - 1, int(round(y2))))
                cv2.rectangle(rendered, (x1_i, y1_i), (x2_i, y2_i), tuple(int(value) for value in color), 2)
                text_x = x1_i
                text_y = max(22, y1_i - 7)

        (text_width, text_height), _baseline = cv2.getTextSize(
            caption,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            1,
        )
        cv2.rectangle(
            rendered,
            (text_x, max(0, text_y - text_height - 7)),
            (min(width - 1, text_x + text_width + 8), min(height - 1, text_y + 4)),
            (12, 17, 27),
            -1,
        )
        cv2.putText(
            rendered,
            caption,
            (text_x + 4, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 248, 255),
            1,
            cv2.LINE_AA,
        )
        accepted += 1

    return rendered, accepted


@require_POST
def upload_sam3_checkpoint(request: HttpRequest) -> JsonResponse:
    runtime = get_runtime()
    node_name = ""
    try:
        runtime.analysis.assert_configuration_mutable()
        config = runtime.config_store.load()
        node_name = request.POST.get("node", "").strip()
        node = config.nodes.get(node_name)
        if node is None:
            return JsonResponse({"detail": "Choose a valid compute node."}, status=400)

        uploaded = request.FILES.get("checkpoint")
        if uploaded is None:
            return JsonResponse({"detail": "Choose a SAM3 .pt checkpoint."}, status=400)
        filename = str(getattr(uploaded, "name", "sam3.pt"))
        size = int(getattr(uploaded, "size", 0))

        if node.mode == "local":
            payload = SAMCheckpointStore.save_chunks(uploaded.chunks(), filename, expected_size=size)
        else:
            client = InstructionClient(node.host, node.instruction_port or 9000)
            payload = client.upload_sam_checkpoint(uploaded.file, filename=filename, size=size)

        return JsonResponse(
            {
                "node": node_name,
                "checkpoint": payload["path"],
                "filename": payload.get("filename", Path(payload["path"]).name),
                "size_bytes": payload.get("size_bytes", size),
            },
            status=201,
        )
    except (ArcadiaError, ValueError, OSError) as exc:
        return JsonResponse({"detail": str(exc)}, status=502 if node_name and node_name != "local" else 400)


@require_POST
def test_sam3(request: HttpRequest) -> HttpResponse:
    try:
        runtime = get_runtime()
        runtime.analysis.assert_configuration_mutable()
        _configured, node, spec = _resolve_sam(runtime.config_store.load())

        search_term = request.POST.get("search_term", "").strip()
        if not search_term:
            return _plain_text("Enter a SAM3 search term.", status=400)
        if len(search_term) > 200:
            return _plain_text("SAM3 search term must be 200 characters or fewer.", status=400)

        uploaded = request.FILES.get("image")
        if uploaded is None:
            return _plain_text("Choose an image to test SAM3.", status=400)
        if uploaded.size <= 0 or uploaded.size > _IMAGE_LIMIT:
            return _plain_text("Image must be non-empty and at most 10 MB.", status=400)
        content_type = (uploaded.content_type or "").lower()
        if content_type not in _ALLOWED_SAM_IMAGES:
            return _plain_text("Use a JPEG, PNG, or WebP image.", status=400)

        encoded_input: Any = np.frombuffer(uploaded.read(), dtype=np.uint8)
        image = cv2.imdecode(encoded_input, cv2.IMREAD_COLOR)
        if image is None:
            return _plain_text("The uploaded image could not be decoded.", status=400)

        endpoint = _ensure_endpoint(node, spec)
        confidence = float(spec.settings.get("confidence", 0.25))
        result = SAMClient(endpoint).segment(image, [search_term], confidence=confidence)
        rendered, segment_count = _mask_overlay(
            image,
            result.masks,
            result.labels,
            result.confidences,
            result.bounding_boxes,
        )
        if segment_count == 0:
            return _plain_text(
                f"SAM3 found no segments for {search_term!r} at confidence {confidence:.2f}.",
                status=422,
            )

        success, encoded_output = cv2.imencode(".png", rendered)
        if not success:
            raise InferenceError("Could not encode the SAM3 test result.", service_type="sam3")
        response = HttpResponse(encoded_output.tobytes(), content_type="image/png")
        response["Content-Disposition"] = 'inline; filename="sam3-test.png"'
        response["Cache-Control"] = "no-store"
        response["X-Arcadia-Segment-Count"] = str(segment_count)
        return response
    except ValueError as exc:
        return _plain_text(str(exc), status=400)
    except (InferenceError, ServiceError, ArcadiaError, OSError) as exc:
        return _plain_text(str(exc), status=502)


def _resolve_llm(config: AppConfig, role: str) -> tuple[str, ConfiguredService, NodeConfig, ServiceSpec]:
    backing = "llm" if role == "visual_llm" and config.priority_map.visual_llm_mode == "same_as_logical" else role
    configured = config.priority_map.services.get(backing)
    if configured is None:
        raise ValueError(f"No saved {_label(backing)} configuration exists.")
    node = config.nodes.get(configured.node)
    if node is None:
        raise ValueError(f"{_label(backing)} references unknown node {configured.node!r}.")
    spec = ServiceSpec(
        cast(ServiceType, backing),
        configured.port,
        validate_llm_settings(configured.settings, remote=node.mode == "remote"),
    )
    return backing, configured, node, spec


def _ensure_endpoint(node: NodeConfig, spec: ServiceSpec) -> ServiceEndpoint:
    runtime = get_runtime()
    if node.mode == "local":
        if runtime.controller.matches(spec):
            return runtime.controller.endpoint_for(spec.port)
        return runtime.controller.start(spec)
    if node.instruction_port is None:
        raise ValueError("Remote compute node has no instruction port.")
    return InstructionClient(node.host, node.instruction_port).ensure_service(spec)


def _plain_text(text: str, *, status: int = 200) -> HttpResponse:
    return HttpResponse(
        text,
        status=status,
        content_type="text/plain; charset=utf-8",
    )


def _test_chat_response_text(result: Any) -> str:
    """Return the model's text directly, with a reasoning fallback for test chat only."""
    text = result.text if isinstance(result.text, str) else str(result.text)
    if text.strip():
        return text

    raw = result.raw if isinstance(result.raw, dict) else {}
    choices = raw.get("choices")
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    message = first_choice.get("message") if isinstance(first_choice, dict) else {}
    if isinstance(message, dict):
        for key in ("reasoning_content", "reasoning"):
            fallback = message.get(key)
            if isinstance(fallback, str) and fallback.strip():
                return fallback

    finish_reason = first_choice.get("finish_reason") if isinstance(first_choice, dict) else None
    suffix = f" Finish reason: {finish_reason}." if finish_reason else ""
    raise InferenceError(
        "llama-server returned an empty assistant response." + suffix,
        service_type="llm",
    )


@require_POST
def test_llm_chat(request: HttpRequest) -> HttpResponse:
    return _test_chat(request, "llm", False)


@require_POST
def test_visual_llm_chat(request: HttpRequest) -> HttpResponse:
    return _test_chat(request, "visual_llm", True)


def _test_chat(request: HttpRequest, role: str, require_vision: bool) -> HttpResponse:
    try:
        runtime = get_runtime()
        runtime.analysis.assert_configuration_mutable()
        backing, _configured, node, spec = _resolve_llm(runtime.config_store.load(), role)
        if require_vision and not spec.settings.get("vision_enabled"):
            return _plain_text(f"{_label(backing)} does not have vision enabled.", status=400)
        images = None
        if require_vision:
            uploaded = request.FILES.get("image")
            if uploaded is None:
                return _plain_text("An image is required.", status=400)
            if uploaded.size <= 0 or uploaded.size > _IMAGE_LIMIT:
                return _plain_text("Image must be non-empty and at most 10 MB.", status=400)
            content_type = (uploaded.content_type or "").lower()
            if content_type not in _ALLOWED_IMAGES:
                return _plain_text("Use JPEG, PNG, WebP, or GIF.", status=400)
            images = [(content_type, uploaded.read())]
        prompt = request.POST.get("prompt", "").strip() or (
            "Describe this image." if require_vision else "Confirm that chat is working."
        )
        endpoint = _ensure_endpoint(node, spec)
        defaults = generation_settings(spec.settings)
        result = LLMClient(endpoint, role_defaults=defaults).chat(prompt, images=images)
        return _plain_text(_test_chat_response_text(result))
    except ValueError as exc:
        return _plain_text(str(exc), status=400)
    except (InferenceError, ServiceError, ArcadiaError, OSError) as exc:
        return _plain_text(str(exc), status=502)
