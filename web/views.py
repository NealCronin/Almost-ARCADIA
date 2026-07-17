from __future__ import annotations

import copy
import os
import tempfile
import json
import threading
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote

from django.contrib import messages
from django.http import FileResponse, HttpRequest, HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from core.config import ConfiguredService, NodeConfig
from core.errors import ArcadiaError
from core.inference.llm_client import LLMClient
from core.services.host_listener import HostListenerError, HostListenerRestartError
from core.services.instruction_client import InstructionClient
from core.services.llm_runtime import LLMRuntime
from core.services.llm_settings import PROJECTOR_RE, SPLIT_GGUF_RE, validate_hf_repository
from core.services.specs import ServiceEndpoint
from web.artifacts import ArtifactStore
from web.forms import (
    AnalysisForm,
    EndpointTestForm,
    HostListenerForm,
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


def _service_form(name: str, config: Any, data: Any = None):
    form_classes = {"llm": LLMServiceForm, "visual_llm": VisualLLMServiceForm, "sam3": SAMServiceForm}
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


def _models_context(
    request: HttpRequest,
    config: Any,
    *,
    llm_form: LLMServiceForm | None = None,
    sam_form: SAMServiceForm | None = None,
    visual_llm_form: VisualLLMServiceForm | None = None,
    node_form: RemoteNodeForm | None = None,
    editing_node: str | None = None,
    edit_node_form: RemoteNodeForm | None = None,
    allow_save_anyway: bool = False,
    node_error: str | None = None,
) -> dict[str, Any]:
    runtime = get_runtime()
    visual_llm_configured = config.priority_map.services.get("visual_llm")
    visual_llm_form = visual_llm_form or _service_form("visual_llm", config)
    if visual_llm_configured:
        visual_llm_form.initial_from(visual_llm_configured)
    context: dict[str, Any] = {
        "config": config,
        "llm_form": llm_form or _service_form("llm", config),
        "sam_form": sam_form or _service_form("sam3", config),
        "visual_llm_form": visual_llm_form,
        "visual_llm_mode": config.priority_map.visual_llm_mode,
        "visual_llm_readonly": config.priority_map.visual_llm_mode == "same_as_logical",
        "node_form": node_form or RemoteNodeForm(),
        "editing_node": editing_node,
        "edit_node_form": edit_node_form,
        "allow_save_anyway": allow_save_anyway,
        "node_error": node_error,
        "nodes": sorted(config.nodes.items(), key=lambda item: (item[0] != "local", item[0])),
        "node_reachability": {
            name: "local" if node.mode == "local" else "not_checked" for name, node in config.nodes.items()
        },
        "services": runtime.controller.list_services(),
        "log_port": request.GET.get("log_port"),
        "log_text": None,
        "llm_advanced_open": bool(
            (llm_form and (llm_form.errors or llm_form.legacy_local_model))
            or (llm_form and llm_form.initial.get("vision_enabled"))
            or (
                llm_form
                and any(
                    llm_form.initial.get(key) not in (None, "", default)
                    for key, default in {
                        "model_file_pattern": "",
                        "model_alias": "local-model",
                        "chat_format": "",
                        "n_gpu_layers": -1,
                        "n_batch": 2048,
                        "n_ubatch": 512,
                    }.items()
                )
            )
        ),
        "node_hosts": {name: node.host for name, node in config.nodes.items()},
    }
    if request.GET.get("log_port"):
        try:
            context["log_text"] = runtime.controller.get_logs(int(request.GET["log_port"]))
        except (ValueError, ArcadiaError) as exc:
            context["log_text"] = str(exc)
    return context


def _render_models(
    request: HttpRequest,
    config: Any,
    *,
    llm_form: LLMServiceForm | None = None,
    sam_form: SAMServiceForm | None = None,
    visual_llm_form: VisualLLMServiceForm | None = None,
    node_form: RemoteNodeForm | None = None,
    editing_node: str | None = None,
    edit_node_form: RemoteNodeForm | None = None,
    allow_save_anyway: bool = False,
    node_error: str | None = None,
    status: int = 200,
) -> HttpResponse:
    return render(
        request,
        "web/services.html",
        _models_context(
            request,
            config,
            llm_form=llm_form,
            sam_form=sam_form,
            visual_llm_form=visual_llm_form,
            node_form=node_form,
            editing_node=editing_node,
            edit_node_form=edit_node_form,
            allow_save_anyway=allow_save_anyway,
            node_error=node_error,
        ),
        status=status,
    )


@require_GET
def services(request: HttpRequest) -> HttpResponse:
    runtime = get_runtime()
    config = runtime.config_store.load()
    editing_node = request.GET.get("edit")
    edit_node_form = None
    if editing_node and editing_node in config.nodes and editing_node != "local":
        node = config.nodes[editing_node]
        edit_node_form = RemoteNodeForm(
            initial={"name": editing_node, "host": node.host, "instruction_port": node.instruction_port}
        )
    return _render_models(request, config, editing_node=editing_node, edit_node_form=edit_node_form)


@require_POST
def inspect_llm_repository(request: HttpRequest) -> JsonResponse:
    try:
        payload = json.loads(request.body)
        if not isinstance(payload, dict):
            raise ValueError("JSON object required.")
        repository = validate_hf_repository(str(payload.get("hf_repo", "")))
        mmproj_repository = payload.get("mmproj_repo")
        if mmproj_repository:
            mmproj_repository = validate_hf_repository(str(mmproj_repository))
        files = LLMRuntime.list_repository_files(repository)
        models = [
            Path(item).name
            for item in files
            if item.lower().endswith(".gguf")
            and not PROJECTOR_RE.search(Path(item).name)
            and not SPLIT_GGUF_RE.search(Path(item).name)
        ]
        projector_files = files if not mmproj_repository else LLMRuntime.list_repository_files(mmproj_repository)
        projectors = [
            Path(item).name
            for item in projector_files
            if item.lower().endswith(".gguf")
            and PROJECTOR_RE.search(Path(item).name)
            and not SPLIT_GGUF_RE.search(Path(item).name)
        ]
        return JsonResponse(
            {
                "models": models[:50],
                "mmproj": projectors[:50],
                "model_ambiguous": len(models) != 1,
                "mmproj_ambiguous": bool(projectors) and len(projectors) != 1,
                "message": "Repository inspected without downloading model files.",
            }
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return JsonResponse({"error": str(exc)}, status=400)


def _remote_node_reachable(node: NodeConfig) -> bool:
    return InstructionClient(node.host, node.instruction_port or 9000, timeout=2.0, retries=0).health()


def _save_remote_node(
    config: Any,
    *,
    old_name: str | None,
    form: RemoteNodeForm,
) -> tuple[str, NodeConfig]:
    name = form.cleaned_data["name"]
    if name in config.nodes and name != old_name:
        form.add_error("name", f"A compute node named '{name}' already exists.")
        raise ValueError("duplicate node name")
    previous = config.nodes.get(old_name) if old_name else None
    node = form.to_config(extra=copy.deepcopy(previous.extra) if previous else {})
    if old_name is not None:
        del config.nodes[old_name]
    config.nodes[name] = node
    if old_name is not None and name != old_name:
        for configured in config.priority_map.services.values():
            if configured.node == old_name:
                configured.node = name
    return name, node


def _node_form_failure(
    request: HttpRequest,
    config: Any,
    form: RemoteNodeForm,
    *,
    editing_node: str | None = None,
    allow_save_anyway: bool = False,
    status: int = 400,
) -> HttpResponse:
    return _render_models(
        request,
        config,
        node_form=form if editing_node is None else None,
        editing_node=editing_node,
        edit_node_form=form if editing_node is not None else None,
        allow_save_anyway=allow_save_anyway,
        status=status,
    )


@require_POST
def add_remote_node(request: HttpRequest) -> HttpResponse:
    runtime = get_runtime()
    form = RemoteNodeForm(request.POST)
    if not form.is_valid():
        return _node_form_failure(request, runtime.config_store.load(), form)
    with runtime.config_lock, _models_node_lock:
        try:
            runtime.analysis.assert_configuration_mutable()
            config = runtime.config_store.load()
            name = form.cleaned_data["name"]
            if name in config.nodes:
                form.add_error("name", f"A compute node named '{name}' already exists.")
                return _node_form_failure(request, config, form)
            node = form.to_config()
            reachable = _remote_node_reachable(node)
            if not reachable and request.POST.get("save_anyway") != "1":
                form.add_error(None, "Instruction server is unreachable. Test the address or choose Save anyway.")
                return _node_form_failure(request, config, form, allow_save_anyway=True)
            config.nodes[name] = node
            runtime.config_store.save(config)
        except (ArcadiaError, OSError) as exc:
            form.add_error(None, f"Could not save remote computer: {exc}")
            return _node_form_failure(request, runtime.config_store.load(), form, status=500)
    messages.success(
        request,
        f"Saved remote computer '{name}'{' without a reachable instruction server' if not reachable else ''}.",
    )
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
            return _node_form_failure(request, config, form, editing_node=node_name, status=400)
        if not form.is_valid():
            return _node_form_failure(request, config, form, editing_node=node_name)
        try:
            runtime.analysis.assert_configuration_mutable()
            name = form.cleaned_data["name"]
            if name in config.nodes and name != node_name:
                form.add_error("name", f"A compute node named '{name}' already exists.")
                return _node_form_failure(request, config, form, editing_node=node_name)
            replacement = form.to_config(extra=copy.deepcopy(previous.extra))
            reachable = _remote_node_reachable(replacement)
            if not reachable and request.POST.get("save_anyway") != "1":
                form.add_error(None, "Instruction server is unreachable. Test the address or choose Save anyway.")
                return _node_form_failure(
                    request,
                    config,
                    form,
                    editing_node=node_name,
                    allow_save_anyway=True,
                )
            _save_remote_node(config, old_name=node_name, form=form)
            runtime.config_store.save(config)
        except (ArcadiaError, OSError) as exc:
            form.add_error(None, f"Could not save remote computer: {exc}")
            return _node_form_failure(request, runtime.config_store.load(), form, editing_node=node_name, status=500)
    messages.success(
        request,
        f"Saved remote computer '{name}'{' without a reachable instruction server' if not reachable else ''}.",
    )
    return redirect("priority_map_models")


@require_POST
def delete_remote_node(request: HttpRequest, node_name: str) -> HttpResponse:
    runtime = get_runtime()
    with runtime.config_lock, _models_node_lock:
        try:
            runtime.analysis.assert_configuration_mutable()
            config = runtime.config_store.load()
            node = config.nodes.get(node_name)
            if node is None:
                return HttpResponse("Unknown compute node.", status=404)
            if node_name == "local" or node.mode != "remote":
                return _render_models(request, config, node_error="This computer cannot be deleted.", status=400)
            references = [
                name.upper()
                for name, configured in config.priority_map.services.items()
                if configured.node == node_name
            ]
            if references:
                return _render_models(
                    request,
                    config,
                    node_error=(
                        f"Cannot delete '{node_name}'; it is used by {', '.join(references)}. Move that service first."
                    ),
                    status=409,
                )
            del config.nodes[node_name]
            runtime.config_store.save(config)
        except (ArcadiaError, OSError) as exc:
            return _render_models(
                request, runtime.config_store.load(), node_error=f"Could not delete remote computer: {exc}", status=500
            )
    messages.success(request, f"Deleted remote computer '{node_name}'.")
    return redirect("priority_map_models")


@require_POST
def test_remote_node(request: HttpRequest, node_name: str) -> JsonResponse:
    runtime = get_runtime()
    with runtime.config_lock:
        config = runtime.config_store.load()
        node = config.nodes.get(node_name)
    if node is None:
        return JsonResponse({"state": "unknown", "message": "Unknown compute node."}, status=404)
    if node_name == "local" or node.mode != "remote":
        return JsonResponse(
            {"state": "local", "message": "This computer does not require remote health testing."},
            status=400,
        )
    if _remote_node_reachable(node):
        return JsonResponse({"state": "reachable", "message": "Instruction server is reachable."})
    return JsonResponse({"state": "unreachable", "message": "Instruction server is unreachable."})


@require_POST
def start_service(request: HttpRequest, service_name: str) -> HttpResponse:
    if service_name not in ("llm", "visual_llm", "sam3"):
        return HttpResponse("Unknown service", status=404)
    runtime = get_runtime()
    with runtime.config_lock:
        try:
            runtime.analysis.assert_configuration_mutable()
            config = runtime.config_store.load()
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
    return redirect("priority_map_models")


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
    return redirect("priority_map_models")


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
    with runtime.config_lock, _host_listener_save_lock:
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
    with runtime.config_lock:
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
        except (ArcadiaError, ValueError, OSError) as exc:
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



@require_POST
def test_llm_chat(request: HttpRequest) -> JsonResponse:
    """Start Logical LLM, wait for readiness, send text prompt, return response."""
    return _test_chat(request, role="llm", require_vision=False)


@require_POST
def test_visual_llm_chat(request: HttpRequest) -> JsonResponse:
    """Start Visual LLM (or reuse Logical), wait for readiness, send prompt+image, return response."""
    return _test_chat(request, role="visual_llm", require_vision=True)


def _test_chat(request: HttpRequest, role: str, require_vision: bool) -> JsonResponse:
    """Validate saved settings, ensure service running, send chat request, return JSON."""
    from core.errors import InferenceError, ServiceError

    runtime = get_runtime()
    try:
        config = runtime.config_store.load()
    except ArcadiaError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    # Validate saved settings exist for the role
    configured = config.priority_map.services.get(role)
    if configured is None:
        return JsonResponse(
            {"error": f"No configuration saved for {role.upper()}. Save settings first."},
            status=400,
        )

    # If require_vision, validate vision is enabled and projector exists
    if require_vision:
        if not configured.settings.get("vision_enabled"):
            return JsonResponse(
                {"error": "Vision is not enabled for this service. Enable vision and save a projector file."},
                status=400,
            )
        if not configured.settings.get("mmproj_repo") and not configured.settings.get("mmproj_file_pattern"):
            return JsonResponse(
                {"error": "No projector (MMProj) configured. Visual chat requires a multimodal projector."},
                status=400,
            )

    # Handle image upload for visual chat
    image_data: list[bytes] = []
    if require_vision and request.FILES.get("image"):
        uploaded = request.FILES["image"]
        if uploaded.size > 10 * 1024 * 1024:
            return JsonResponse({"error": "Image must be under 10 MB."}, status=400)
        allowed_types = {"image/jpeg", "image/png", "image/webp", "image/gif"}
        if uploaded.content_type not in allowed_types:
            return JsonResponse(
                {"error": f"Unsupported image type {uploaded.content_type}. Use JPEG, PNG, WebP, or GIF."},
                status=400,
            )
        image_data = [uploaded.read()]

    # Get the prompt from the request
    prompt = request.POST.get("prompt", "").strip()
    if not prompt:
        prompt = "Describe the contents of this image." if require_vision else "Hello, what model are you?"

    try:
        # Ensure the service is running
        node = config.nodes.get(configured.node)
        if node is None:
            return JsonResponse({"error": f"Service {role} references unknown node {configured.node!r}."}, status=400)

        if node.mode == "local":
            # Check if already running
            running_services = runtime.controller.list_services()
            running_ports = {s.port for s in running_services if s.running}
            if configured.port not in running_ports:
                # Start the service (reuse same settings)
                endpoint = runtime.controller.start(configured.spec)
            else:
                # Build endpoint from existing service info
                for svc in running_services:
                    if svc.port == configured.port:
                        endpoint = ServiceEndpoint(
                            host="127.0.0.1",
                            port=svc.port,
                            service_type=svc.service_type,
                        )
                        break
                else:
                    endpoint = ServiceEndpoint(host="127.0.0.1", port=configured.port, service_type="llm")
        else:
            # Remote node - use instruction client
            if node.instruction_port is None:
                return JsonResponse(
                    {"error": f"Remote node {configured.node!r} has no instruction port."}, status=400
                )
            client = InstructionClient(node.host, node.instruction_port)
            endpoint = client.start_service(configured.spec)

        # Build LLMClient and send chat request
        role_defaults = {}
        for key in ("temperature", "top_k", "min_p", "top_p", "max_tokens",
                     "repeat_penalty", "presence_penalty", "frequency_penalty", "seed"):
            if key in configured.settings:
                role_defaults[key] = configured.settings[key]

        llm_client = LLMClient(endpoint, role_defaults=role_defaults)
        result = llm_client.chat(
            prompt,
            images=image_data if image_data else None,
            model=configured.settings.get("model_alias", "local-model"),
        )
        return JsonResponse({"response": result.text})

    except (InferenceError, ServiceError, ArcadiaError, ValueError, OSError) as exc:
        return JsonResponse({"error": str(exc)}, status=500)
    except Exception as exc:
        return JsonResponse({"error": f"Unexpected error: {exc}"}, status=500)