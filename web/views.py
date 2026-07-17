from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote

from django.contrib import messages
from django.http import (
    FileResponse,
    HttpRequest,
    HttpResponse,
    JsonResponse,
    StreamingHttpResponse,
)
from django.shortcuts import redirect, render
from django.views.decorators.http import (
    require_GET,
    require_http_methods,
    require_POST,
)

from core.config import ConfiguredService, NodeConfig
from core.errors import ArcadiaError, ConfigurationError
from core.inference.llm_client import LLMClient
from core.services.host_listener import (
    HostListenerError,
    HostListenerRestartError,
)
from core.services.instruction_client import InstructionClient
from core.services.llm_runtime import LLMRuntime
from core.services.llm_settings import (
    PROJECTOR_RE,
    SPLIT_GGUF_RE,
    generation_settings,
    validate_hf_repository,
    validate_llm_settings,
)
from core.services.specs import ServiceEndpoint, ServiceSpec
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

_LLM_SERVICE_NAMES = {
    "llm",
    "visual_llm",
}

_SERVICE_LABELS = {
    "llm": "Logical LLM",
    "visual_llm": "Visual LLM",
    "sam3": "SAM3",
}

_IMAGE_UPLOAD_LIMIT = 10 * 1024 * 1024
_ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
}


# ============================================================
# Model-page helpers
# ============================================================


def _service_label(service_name: str) -> str:
    return _SERVICE_LABELS.get(
        service_name,
        service_name.replace("_", " ").title(),
    )


def _service_form(
    name: str,
    config: Any,
    data: Any = None,
):
    form_classes = {
        "llm": LLMServiceForm,
        "visual_llm": VisualLLMServiceForm,
        "sam3": SAMServiceForm,
    }

    if name not in TOOLS["priority-map"].required_services:
        raise ValueError(
            f"Unknown Priority Map service {name}."
        )

    form_class = form_classes[name]

    if data is None:
        form = form_class(
            nodes=config.nodes,
            auto_id=f"{name}_%s",
        )
    else:
        form = form_class(
            data=data,
            nodes=config.nodes,
            auto_id=f"{name}_%s",
        )

    form.initial_from(
        config.priority_map.services.get(name)
    )

    return form


def _llm_advanced_should_open(
    form: LLMServiceForm | None,
) -> bool:
    if form is None:
        return False

    if form.errors or form.legacy_local_model:
        return True

    advanced_defaults: dict[str, Any] = {
        "model_file_pattern": "",
        "model_alias": "logical-model",
        "chat_format": "",
        "mmproj_file_pattern": "",
        "draft_enabled": False,
        "n_gpu_layers": "all",
        "n_threads": None,
        "n_batch": 2048,
        "n_ubatch": 512,
        "flash_attn": "auto",
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "use_mmap": True,
        "use_mlock": False,
        "max_tokens": 1024,
        "repeat_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "seed": None,
        "additional_arguments": "",
    }

    for key, default in advanced_defaults.items():
        value = form.initial.get(key)

        if value not in (
            None,
            "",
            default,
        ):
            return True

    return False


def _resolve_llm_backing_service(
    config: Any,
    requested_role: str,
) -> tuple[str, ConfiguredService]:
    if requested_role not in _LLM_SERVICE_NAMES:
        raise ValueError(
            f"Unknown LLM role {requested_role!r}."
        )

    backing_role = requested_role

    if (
        requested_role == "visual_llm"
        and config.priority_map.visual_llm_mode
        == "same_as_logical"
    ):
        backing_role = "llm"

    configured = config.priority_map.services.get(
        backing_role
    )

    if configured is None:
        if backing_role == "llm":
            raise ValueError(
                "No Logical LLM configuration is saved. "
                "Save Logical LLM settings first."
            )

        raise ValueError(
            "No separate Visual LLM configuration is saved. "
            "Save Visual LLM settings first."
        )

    return backing_role, configured


def _normalized_llm_spec(
    configured: ConfiguredService,
    *,
    backing_role: str,
    node: NodeConfig,
) -> ServiceSpec:
    settings = validate_llm_settings(
        configured.settings,
        remote=node.mode == "remote",
    )

    return ServiceSpec(
        service_type=backing_role,
        port=configured.port,
        settings=settings,
    )


def _validate_separate_llm_port_collision(
    config: Any,
    service_name: str,
    candidate: ConfiguredService,
) -> None:
    if (
        config.priority_map.visual_llm_mode
        != "separate"
    ):
        return

    if service_name not in _LLM_SERVICE_NAMES:
        return

    other_name = (
        "visual_llm"
        if service_name == "llm"
        else "llm"
    )
    other = config.priority_map.services.get(
        other_name
    )

    if other is None:
        return

    if (
        other.node == candidate.node
        and other.port == candidate.port
    ):
        raise ValueError(
            "Logical LLM and a separate Visual LLM "
            "cannot use the same port on the same compute "
            "computer. Choose another inference port or node."
        )


def _validate_existing_separate_llm_collision(
    config: Any,
) -> None:
    logical = config.priority_map.services.get("llm")
    visual = config.priority_map.services.get(
        "visual_llm"
    )

    if logical is None or visual is None:
        return

    if (
        logical.node == visual.node
        and logical.port == visual.port
    ):
        raise ValueError(
            "The saved Logical and Visual LLM settings "
            "use the same port on the same compute computer. "
            "Change one before enabling separate Visual mode."
        )


def _inspection_candidates(
    files: list[str],
    *,
    projector: bool,
) -> tuple[list[str], list[str]]:
    names: list[str] = []
    duplicate_names: set[str] = set()
    seen_paths_by_name: dict[str, str] = {}

    for repository_path in files:
        if not repository_path.lower().endswith(".gguf"):
            continue

        basename = Path(repository_path).name
        is_projector = bool(
            PROJECTOR_RE.search(basename)
        )

        if is_projector != projector:
            continue

        split_match = SPLIT_GGUF_RE.search(basename)

        if split_match is not None:
            shard_number = int(split_match.group(1))

            # Offer only shard one. The runtime downloads the full set.
            if shard_number != 1:
                continue

        normalized_name = basename.casefold()
        previous_path = seen_paths_by_name.get(
            normalized_name
        )

        if (
            previous_path is not None
            and previous_path != repository_path
        ):
            duplicate_names.add(basename)
        else:
            seen_paths_by_name[
                normalized_name
            ] = repository_path

        names.append(basename)

    unique_names = sorted(
        set(names),
        key=str.casefold,
    )
    duplicate_list = sorted(
        duplicate_names,
        key=str.casefold,
    )

    return unique_names, duplicate_list


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

    logical_form = (
        llm_form
        or _service_form(
            "llm",
            config,
        )
    )
    separate_visual_form = (
        visual_llm_form
        or _service_form(
            "visual_llm",
            config,
        )
    )

    context: dict[str, Any] = {
        "config": config,
        "llm_form": logical_form,
        "sam_form": (
            sam_form
            or _service_form(
                "sam3",
                config,
            )
        ),
        "visual_llm_form": separate_visual_form,
        "visual_llm_mode": (
            config.priority_map.visual_llm_mode
        ),
        "visual_llm_readonly": (
            config.priority_map.visual_llm_mode
            == "same_as_logical"
        ),
        "node_form": node_form or RemoteNodeForm(),
        "editing_node": editing_node,
        "edit_node_form": edit_node_form,
        "allow_save_anyway": allow_save_anyway,
        "node_error": node_error,
        "nodes": sorted(
            config.nodes.items(),
            key=lambda item: (
                item[0] != "local",
                item[0].casefold(),
            ),
        ),
        "node_reachability": {
            name: (
                "local"
                if node.mode == "local"
                else "not_checked"
            )
            for name, node in config.nodes.items()
        },
        "services": runtime.controller.list_services(),
        "log_port": request.GET.get("log_port"),
        "log_text": None,
        "llm_advanced_open": (
            _llm_advanced_should_open(logical_form)
        ),
        "visual_llm_advanced_open": (
            _llm_advanced_should_open(
                separate_visual_form
            )
        ),
        "node_hosts": {
            name: node.host
            for name, node in config.nodes.items()
        },
    }

    if request.GET.get("log_port"):
        try:
            context["log_text"] = (
                runtime.controller.get_logs(
                    int(request.GET["log_port"])
                )
            )
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


# ============================================================
# Main pages
# ============================================================


def home(request: HttpRequest) -> HttpResponse:
    runtime = get_runtime()
    config = runtime.config_store.load()

    return render(
        request,
        "web/home.html",
        {
            "config": config,
            "services": (
                runtime.controller.list_services()
            ),
            "analysis": runtime.analysis.status(),
        },
    )


@require_GET
def client_portal(
    request: HttpRequest,
) -> HttpResponse:
    runtime = get_runtime()

    return render(
        request,
        "web/client.html",
        {
            "analysis": runtime.analysis.status(),
        },
    )


@require_GET
def services(request: HttpRequest) -> HttpResponse:
    runtime = get_runtime()
    config = runtime.config_store.load()

    editing_node = request.GET.get("edit")
    edit_node_form = None

    if (
        editing_node
        and editing_node in config.nodes
        and editing_node != "local"
    ):
        node = config.nodes[editing_node]

        edit_node_form = RemoteNodeForm(
            initial={
                "name": editing_node,
                "host": node.host,
                "instruction_port": (
                    node.instruction_port
                ),
            }
        )

    return _render_models(
        request,
        config,
        editing_node=editing_node,
        edit_node_form=edit_node_form,
    )


# ============================================================
# Repository inspection
# ============================================================


@require_POST
def inspect_llm_repository(
    request: HttpRequest,
) -> JsonResponse:
    try:
        payload = json.loads(request.body)

        if not isinstance(payload, dict):
            raise ValueError(
                "JSON object required."
            )

        repository = validate_hf_repository(
            str(payload.get("hf_repo", ""))
        )

        raw_mmproj_repository = payload.get(
            "mmproj_repo"
        )
        raw_draft_repository = payload.get(
            "draft_repo"
        )

        mmproj_repository = (
            validate_hf_repository(
                str(raw_mmproj_repository)
            )
            if raw_mmproj_repository
            else repository
        )
        draft_repository = (
            validate_hf_repository(
                str(raw_draft_repository)
            )
            if raw_draft_repository
            else repository
        )

        files_by_repository: dict[str, list[str]] = {}

        for repository_id in {
            repository,
            mmproj_repository,
            draft_repository,
        }:
            files_by_repository[repository_id] = (
                LLMRuntime.list_repository_files(
                    repository_id
                )
            )

        models, model_duplicates = (
            _inspection_candidates(
                files_by_repository[repository],
                projector=False,
            )
        )
        projectors, projector_duplicates = (
            _inspection_candidates(
                files_by_repository[
                    mmproj_repository
                ],
                projector=True,
            )
        )
        drafts, draft_duplicates = (
            _inspection_candidates(
                files_by_repository[
                    draft_repository
                ],
                projector=False,
            )
        )

        warnings: list[str] = []

        duplicate_groups = {
            "model": model_duplicates,
            "projector": projector_duplicates,
            "draft": draft_duplicates,
        }

        for label, duplicates in duplicate_groups.items():
            if duplicates:
                warnings.append(
                    f"Duplicate {label} basenames exist in "
                    f"different repository directories: "
                    f"{', '.join(duplicates[:10])}."
                )

        message = (
            "Repositories inspected without downloading "
            "model files."
        )

        if warnings:
            message = (
                f"{message} {' '.join(warnings)}"
            )

        return JsonResponse(
            {
                "models": models[:50],
                "mmproj": projectors[:50],
                "drafts": drafts[:50],
                "model_ambiguous": len(models) != 1,
                "mmproj_ambiguous": (
                    len(projectors) != 1
                ),
                "draft_ambiguous": len(drafts) != 1,
                "duplicate_model_basenames": (
                    model_duplicates[:50]
                ),
                "duplicate_mmproj_basenames": (
                    projector_duplicates[:50]
                ),
                "duplicate_draft_basenames": (
                    draft_duplicates[:50]
                ),
                "message": message,
            }
        )

    except (
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        return JsonResponse(
            {
                "error": str(exc),
            },
            status=400,
        )


# ============================================================
# Compute nodes
# ============================================================


def _remote_node_reachable(
    node: NodeConfig,
) -> bool:
    return InstructionClient(
        node.host,
        node.instruction_port or 9000,
        timeout=2.0,
        retries=0,
    ).health()


def _save_remote_node(
    config: Any,
    *,
    old_name: str | None,
    form: RemoteNodeForm,
) -> tuple[str, NodeConfig]:
    name = form.cleaned_data["name"]

    if (
        name in config.nodes
        and name != old_name
    ):
        form.add_error(
            "name",
            f"A compute node named '{name}' already exists.",
        )
        raise ValueError(
            "duplicate node name"
        )

    previous = (
        config.nodes.get(old_name)
        if old_name
        else None
    )

    node = form.to_config(
        extra=(
            copy.deepcopy(previous.extra)
            if previous
            else {}
        )
    )

    if old_name is not None:
        del config.nodes[old_name]

    config.nodes[name] = node

    if (
        old_name is not None
        and name != old_name
    ):
        for configured in (
            config.priority_map.services.values()
        ):
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
        node_form=(
            form
            if editing_node is None
            else None
        ),
        editing_node=editing_node,
        edit_node_form=(
            form
            if editing_node is not None
            else None
        ),
        allow_save_anyway=allow_save_anyway,
        status=status,
    )


@require_POST
def add_remote_node(
    request: HttpRequest,
) -> HttpResponse:
    runtime = get_runtime()
    form = RemoteNodeForm(request.POST)

    if not form.is_valid():
        return _node_form_failure(
            request,
            runtime.config_store.load(),
            form,
        )

    with runtime.config_lock, _models_node_lock:
        try:
            runtime.analysis.assert_configuration_mutable()
            config = runtime.config_store.load()
            name = form.cleaned_data["name"]

            if name in config.nodes:
                form.add_error(
                    "name",
                    f"A compute node named '{name}' "
                    "already exists.",
                )
                return _node_form_failure(
                    request,
                    config,
                    form,
                )

            node = form.to_config()
            reachable = _remote_node_reachable(node)

            if (
                not reachable
                and request.POST.get("save_anyway")
                != "1"
            ):
                form.add_error(
                    None,
                    "Instruction server is unreachable. "
                    "Test the address or choose Save anyway.",
                )
                return _node_form_failure(
                    request,
                    config,
                    form,
                    allow_save_anyway=True,
                )

            config.nodes[name] = node
            runtime.config_store.save(config)

        except (ArcadiaError, OSError) as exc:
            form.add_error(
                None,
                f"Could not save remote computer: {exc}",
            )
            return _node_form_failure(
                request,
                runtime.config_store.load(),
                form,
                status=500,
            )

    suffix = (
        " without a reachable instruction server"
        if not reachable
        else ""
    )

    messages.success(
        request,
        f"Saved remote computer '{name}'{suffix}.",
    )

    return redirect("priority_map_models")


@require_POST
def edit_remote_node(
    request: HttpRequest,
    node_name: str,
) -> HttpResponse:
    runtime = get_runtime()
    form = RemoteNodeForm(request.POST)

    with runtime.config_lock, _models_node_lock:
        config = runtime.config_store.load()
        previous = config.nodes.get(node_name)

        if previous is None:
            return HttpResponse(
                "Unknown compute node.",
                status=404,
            )

        if (
            node_name == "local"
            or previous.mode != "remote"
        ):
            return _node_form_failure(
                request,
                config,
                form,
                editing_node=node_name,
                status=400,
            )

        if not form.is_valid():
            return _node_form_failure(
                request,
                config,
                form,
                editing_node=node_name,
            )

        try:
            runtime.analysis.assert_configuration_mutable()
            name = form.cleaned_data["name"]

            if (
                name in config.nodes
                and name != node_name
            ):
                form.add_error(
                    "name",
                    f"A compute node named '{name}' "
                    "already exists.",
                )
                return _node_form_failure(
                    request,
                    config,
                    form,
                    editing_node=node_name,
                )

            replacement = form.to_config(
                extra=copy.deepcopy(previous.extra)
            )
            reachable = _remote_node_reachable(
                replacement
            )

            if (
                not reachable
                and request.POST.get("save_anyway")
                != "1"
            ):
                form.add_error(
                    None,
                    "Instruction server is unreachable. "
                    "Test the address or choose Save anyway.",
                )
                return _node_form_failure(
                    request,
                    config,
                    form,
                    editing_node=node_name,
                    allow_save_anyway=True,
                )

            _save_remote_node(
                config,
                old_name=node_name,
                form=form,
            )
            runtime.config_store.save(config)

        except (ArcadiaError, OSError) as exc:
            form.add_error(
                None,
                f"Could not save remote computer: {exc}",
            )
            return _node_form_failure(
                request,
                runtime.config_store.load(),
                form,
                editing_node=node_name,
                status=500,
            )

    suffix = (
        " without a reachable instruction server"
        if not reachable
        else ""
    )

    messages.success(
        request,
        f"Saved remote computer '{name}'{suffix}.",
    )

    return redirect("priority_map_models")


@require_POST
def delete_remote_node(
    request: HttpRequest,
    node_name: str,
) -> HttpResponse:
    runtime = get_runtime()

    with runtime.config_lock, _models_node_lock:
        try:
            runtime.analysis.assert_configuration_mutable()
            config = runtime.config_store.load()
            node = config.nodes.get(node_name)

            if node is None:
                return HttpResponse(
                    "Unknown compute node.",
                    status=404,
                )

            if (
                node_name == "local"
                or node.mode != "remote"
            ):
                return _render_models(
                    request,
                    config,
                    node_error=(
                        "This computer cannot be deleted."
                    ),
                    status=400,
                )

            references = [
                _service_label(name)
                for name, configured in (
                    config.priority_map.services.items()
                )
                if configured.node == node_name
            ]

            if references:
                return _render_models(
                    request,
                    config,
                    node_error=(
                        f"Cannot delete '{node_name}'; "
                        f"it is used by "
                        f"{', '.join(references)}. "
                        "Move that service first."
                    ),
                    status=409,
                )

            del config.nodes[node_name]
            runtime.config_store.save(config)

        except (ArcadiaError, OSError) as exc:
            return _render_models(
                request,
                runtime.config_store.load(),
                node_error=(
                    "Could not delete remote computer: "
                    f"{exc}"
                ),
                status=500,
            )

    messages.success(
        request,
        f"Deleted remote computer '{node_name}'.",
    )

    return redirect("priority_map_models")


@require_POST
def test_remote_node(
    request: HttpRequest,
    node_name: str,
) -> JsonResponse:
    runtime = get_runtime()

    with runtime.config_lock:
        config = runtime.config_store.load()
        node = config.nodes.get(node_name)

    if node is None:
        return JsonResponse(
            {
                "state": "unknown",
                "message": "Unknown compute node.",
            },
            status=404,
        )

    if (
        node_name == "local"
        or node.mode != "remote"
    ):
        return JsonResponse(
            {
                "state": "local",
                "message": (
                    "This computer does not require "
                    "remote health testing."
                ),
            },
            status=400,
        )

    if _remote_node_reachable(node):
        return JsonResponse(
            {
                "state": "reachable",
                "message": (
                    "Instruction server is reachable."
                ),
            }
        )

    return JsonResponse(
        {
            "state": "unreachable",
            "message": (
                "Instruction server is unreachable."
            ),
        }
    )


# ============================================================
# Model and service configuration
# ============================================================


@require_POST
def start_service(
    request: HttpRequest,
    service_name: str,
) -> HttpResponse:
    if service_name not in (
        "llm",
        "visual_llm",
        "sam3",
    ):
        return HttpResponse(
            "Unknown service",
            status=404,
        )

    runtime = get_runtime()

    with runtime.config_lock:
        try:
            runtime.analysis.assert_configuration_mutable()
            config = runtime.config_store.load()

            # The Visual mode selector intentionally shares this
            # endpoint with the Visual LLM settings form.
            if (
                service_name == "visual_llm"
                and "visual_llm_mode" in request.POST
            ):
                mode = request.POST[
                    "visual_llm_mode"
                ]

                if mode not in (
                    "same_as_logical",
                    "separate",
                ):
                    messages.error(
                        request,
                        "Visual LLM mode must be "
                        "Same as Logical or Separate.",
                    )
                    return redirect(
                        "priority_map_models"
                    )

                if mode == "separate":
                    original_mode = (
                        config.priority_map.visual_llm_mode
                    )
                    config.priority_map.visual_llm_mode = (
                        "separate"
                    )

                    try:
                        _validate_existing_separate_llm_collision(
                            config
                        )
                    except ValueError:
                        config.priority_map.visual_llm_mode = (
                            original_mode
                        )
                        raise

                config.priority_map.visual_llm_mode = mode
                runtime.config_store.save(config)

                label = (
                    "Same as Logical LLM"
                    if mode == "same_as_logical"
                    else "Separate Visual LLM"
                )

                messages.success(
                    request,
                    f"Visual LLM mode set to {label}.",
                )

                return redirect(
                    "priority_map_models"
                )

            form = _service_form(
                service_name,
                config,
                request.POST,
            )

            if not form.is_valid():
                return _render_models(
                    request,
                    config,
                    llm_form=(
                        form
                        if service_name == "llm"
                        else None
                    ),
                    visual_llm_form=(
                        form
                        if service_name
                        == "visual_llm"
                        else None
                    ),
                    sam_form=(
                        form
                        if service_name == "sam3"
                        else None
                    ),
                    status=400,
                )

            try:
                spec = form.to_spec()
            except (
                ConfigurationError,
                ValueError,
            ) as exc:
                form.add_error(None, str(exc))

                return _render_models(
                    request,
                    config,
                    llm_form=(
                        form
                        if service_name == "llm"
                        else None
                    ),
                    visual_llm_form=(
                        form
                        if service_name
                        == "visual_llm"
                        else None
                    ),
                    sam_form=(
                        form
                        if service_name == "sam3"
                        else None
                    ),
                    status=400,
                )

            # Defend against a form accidentally returning the
            # wrong embedded role.
            if spec.service_type != service_name:
                spec = ServiceSpec(
                    service_type=service_name,
                    port=spec.port,
                    settings=spec.settings,
                )

            previous = (
                config.priority_map.services.get(
                    service_name
                )
            )

            candidate = ConfiguredService(
                node=form.cleaned_data["node"],
                spec=spec,
                extra=(
                    copy.deepcopy(previous.extra)
                    if previous
                    else {}
                ),
            )

            try:
                _validate_separate_llm_port_collision(
                    config,
                    service_name,
                    candidate,
                )
            except ValueError as exc:
                form.add_error(None, str(exc))

                return _render_models(
                    request,
                    config,
                    llm_form=(
                        form
                        if service_name == "llm"
                        else None
                    ),
                    visual_llm_form=(
                        form
                        if service_name
                        == "visual_llm"
                        else None
                    ),
                    sam_form=(
                        form
                        if service_name == "sam3"
                        else None
                    ),
                    status=409,
                )

            config.priority_map.services[
                service_name
            ] = candidate
            runtime.config_store.save(config)

            messages.success(
                request,
                f"Saved {_service_label(service_name)} "
                "settings. Priority Map starts the "
                "service when needed.",
            )

        except (
            ArcadiaError,
            ValueError,
            OSError,
        ) as exc:
            messages.error(
                request,
                str(exc),
            )

    return redirect("priority_map_models")


@require_POST
def stop_service(
    request: HttpRequest,
    service_name: str,
) -> HttpResponse:
    if service_name not in (
        "llm",
        "visual_llm",
        "sam3",
    ):
        return HttpResponse(
            "Unknown service",
            status=404,
        )

    runtime = get_runtime()

    try:
        runtime.analysis.assert_configuration_mutable()
        config = runtime.config_store.load()

        backing_name = service_name

        if (
            service_name == "visual_llm"
            and config.priority_map.visual_llm_mode
            == "same_as_logical"
        ):
            backing_name = "llm"

        configured = (
            config.priority_map.services.get(
                backing_name
            )
        )

        if configured is None:
            raise ArcadiaError(
                "No configuration exists for "
                f"{_service_label(backing_name)}."
            )

        node = config.nodes.get(configured.node)

        if node is None:
            raise ArcadiaError(
                f"{_service_label(backing_name)} "
                f"references unknown node "
                f"{configured.node!r}."
            )

        if node.mode == "local":
            runtime.controller.stop(configured.port)
        else:
            InstructionClient(
                node.host,
                node.instruction_port or 9000,
            ).stop_service(configured.port)

        messages.success(
            request,
            f"Stopped {_service_label(backing_name)} "
            f"on port {configured.port}.",
        )

    except (
        ArcadiaError,
        ValueError,
        OSError,
    ) as exc:
        messages.error(
            request,
            str(exc),
        )

    return redirect("priority_map_models")


@require_GET
def service_logs(
    request: HttpRequest,
    port: int,
) -> HttpResponse:
    runtime = get_runtime()

    try:
        local_status = next(
            (
                status
                for status in (
                    runtime.controller.list_services()
                )
                if status.port == port
            ),
            None,
        )

        if local_status is not None:
            logs = runtime.controller.get_logs(port)

            return HttpResponse(
                logs,
                content_type="text/plain",
            )

        config = runtime.config_store.load()
        matches = [
            configured
            for configured in (
                config.priority_map.services.values()
            )
            if configured.port == port
        ]

        if not matches:
            return HttpResponse(
                f"No configured service on port {port}.",
                status=404,
            )

        if len(matches) > 1:
            return HttpResponse(
                "More than one remote service uses this "
                "port. The current log URL is ambiguous.",
                status=409,
            )

        configured = matches[0]
        node = config.nodes.get(configured.node)

        if node is None:
            return HttpResponse(
                "Configured service references an "
                "unknown compute node.",
                status=400,
            )

        if node.mode == "local":
            logs = runtime.controller.get_logs(port)
        else:
            logs = InstructionClient(
                node.host,
                node.instruction_port or 9000,
            ).get_logs(port)

        return HttpResponse(
            logs,
            content_type="text/plain",
        )

    except (
        ArcadiaError,
        ValueError,
        OSError,
    ) as exc:
        return HttpResponse(
            str(exc),
            status=502,
            content_type="text/plain",
        )


# ============================================================
# Host listener
# ============================================================


@require_GET
def nodes(request: HttpRequest) -> HttpResponse:
    runtime = get_runtime()
    config = runtime.config_store.load()

    return render(
        request,
        "web/nodes.html",
        {
            "form": HostListenerForm(
                initial=config.host_listener.to_dict()
            ),
            "listener": (
                runtime.host_listener.status()
            ),
        },
    )


@require_POST
def save_host_listener(
    request: HttpRequest,
) -> HttpResponse:
    runtime = get_runtime()
    form = HostListenerForm(request.POST)

    if not form.is_valid():
        return render(
            request,
            "web/nodes.html",
            {
                "form": form,
                "listener": (
                    runtime.host_listener.status()
                ),
            },
            status=400,
        )

    with (
        runtime.config_lock,
        _host_listener_save_lock,
    ):
        config = runtime.config_store.load()
        previous = config.host_listener
        replacement = form.to_config()
        replacement.extra = copy.deepcopy(
            previous.extra
        )

        try:
            runtime.host_listener.restart(
                replacement,
                rollback_config=previous,
            )

        except HostListenerRestartError as exc:
            form.add_error(
                None,
                str(exc),
            )

            return render(
                request,
                "web/nodes.html",
                {
                    "form": form,
                    "listener": (
                        runtime.host_listener.status()
                    ),
                },
                status=502,
            )

        except HostListenerError as exc:
            form.add_error(
                None,
                "Instruction server was not changed: "
                f"{exc}",
            )

            return render(
                request,
                "web/nodes.html",
                {
                    "form": form,
                    "listener": (
                        runtime.host_listener.status()
                    ),
                },
                status=400,
            )

        config.host_listener = replacement

        try:
            runtime.config_store.save(config)

        except OSError as exc:
            try:
                runtime.host_listener.restart(
                    previous,
                    rollback_config=replacement,
                )

            except HostListenerRestartError as rollback_exc:
                form.add_error(
                    None,
                    "Configuration was not saved: "
                    f"{exc}. Listener rollback failed: "
                    f"{rollback_exc}",
                )
            else:
                form.add_error(
                    None,
                    "Configuration was not saved: "
                    f"{exc}. The previous listener "
                    "was restored.",
                )

            return render(
                request,
                "web/nodes.html",
                {
                    "form": form,
                    "listener": (
                        runtime.host_listener.status()
                    ),
                },
                status=500,
            )

    messages.success(
        request,
        "Instruction server is listening on "
        f"{replacement.host}:{replacement.port}.",
    )

    return redirect("host")


@require_GET
def host_listener_status(
    request: HttpRequest,
) -> JsonResponse:
    return JsonResponse(
        get_runtime()
        .host_listener.status()
        .to_dict()
    )


# ============================================================
# Analysis configuration and execution
# ============================================================


@require_GET
def analysis_page(
    request: HttpRequest,
) -> HttpResponse:
    runtime = get_runtime()
    config = runtime.config_store.load()

    return render(
        request,
        "web/analysis.html",
        {
            "config": config,
            "pipeline_form": (
                PipelineForm.from_config(
                    config.priority_map.pipeline
                )
            ),
            "analysis_form": AnalysisForm(),
            "analysis": runtime.analysis.status(),
        },
    )


@require_POST
def save_pipeline(
    request: HttpRequest,
) -> HttpResponse:
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
                        "analysis": (
                            runtime.analysis.status()
                        ),
                    },
                    status=400,
                )

            config.priority_map.pipeline = (
                form.to_config()
            )
            runtime.config_store.save(config)

            messages.success(
                request,
                "Saved pipeline settings.",
            )

        except (
            ArcadiaError,
            ValueError,
            OSError,
        ) as exc:
            messages.error(
                request,
                str(exc),
            )

    return redirect("analysis")


@require_POST
def start_analysis(
    request: HttpRequest,
) -> HttpResponse:
    runtime = get_runtime()
    form = AnalysisForm(request.POST)

    if not form.is_valid():
        messages.error(
            request,
            "Choose exactly one input path or "
            "retained upload before starting the analysis.",
        )
        return redirect("analysis")

    try:
        if form.cleaned_data["upload_id"]:
            input_path = runtime.uploads.input_path(
                form.cleaned_data["upload_id"]
            )
        else:
            input_path = (
                Path(
                    form.cleaned_data["input_path"]
                )
                .expanduser()
                .resolve(strict=True)
            )

            if (
                not input_path.is_file()
                and not input_path.is_dir()
            ):
                raise ValueError(
                    "Analysis input must be a regular "
                    "file or directory."
                )

        config = runtime.config_store.load()
        runtime.analysis.start(
            input_path,
            config,
        )

        messages.success(
            request,
            "Analysis started.",
        )

    except (
        ArcadiaError,
        ValueError,
        OSError,
    ) as exc:
        messages.error(
            request,
            str(exc),
        )

    return redirect("results")


# ============================================================
# Uploads
# ============================================================


@require_http_methods(
    [
        "GET",
        "POST",
    ]
)
def uploads(request: HttpRequest) -> JsonResponse:
    runtime = get_runtime()

    if request.method == "GET":
        return JsonResponse(
            {
                "uploads": [
                    _upload_payload(manifest)
                    for manifest in (
                        runtime.uploads.list()
                    )
                ]
            }
        )

    try:
        manifest = runtime.uploads.create(
            request.FILES.getlist("files"),
            request.POST.getlist(
                "relative_paths"
            ),
        )

    except (
        ArcadiaError,
        ValueError,
        OSError,
    ) as exc:
        return JsonResponse(
            {
                "detail": str(exc),
            },
            status=400,
        )

    return JsonResponse(
        {
            "upload": _upload_payload(manifest),
        },
        status=201,
    )


@require_POST
def delete_upload(
    request: HttpRequest,
    upload_id: str,
) -> JsonResponse:
    runtime = get_runtime()

    try:
        input_path = runtime.uploads.input_path(
            upload_id
        )
        analysis = runtime.analysis
        active_input = analysis.status().input_path

        if (
            analysis.is_active()
            and active_input
            and input_path.resolve()
            == Path(active_input).resolve()
        ):
            return JsonResponse(
                {
                    "detail": (
                        "Cannot delete the upload used "
                        "by the active run."
                    )
                },
                status=409,
            )

        runtime.uploads.delete(upload_id)

    except FileNotFoundError:
        return JsonResponse(
            {
                "detail": "Upload not found.",
            },
            status=404,
        )

    except (
        ArcadiaError,
        ValueError,
        OSError,
    ) as exc:
        return JsonResponse(
            {
                "detail": str(exc),
            },
            status=400,
        )

    return JsonResponse(
        {
            "deleted": upload_id,
        }
    )


def _upload_payload(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    upload_id = str(manifest["id"])

    return {
        "id": upload_id,
        "source_type": manifest["source_type"],
        "file_count": manifest["file_count"],
        "size_bytes": manifest["size_bytes"],
        "created_at": manifest["created_at"],
        "delete_url": (
            "/client/priority-map/uploads/"
            f"{upload_id}/delete/"
        ),
    }


# ============================================================
# Analysis status and artifacts
# ============================================================


@require_GET
def analysis_status(
    request: HttpRequest,
) -> JsonResponse:
    return JsonResponse(
        get_runtime().analysis.status().to_dict()
    )


@require_POST
def cancel_run(
    request: HttpRequest,
    run_id: str,
) -> JsonResponse:
    analysis = get_runtime().analysis
    status = analysis.status()

    if (
        run_id != status.run_id
        or not analysis.is_active()
    ):
        return JsonResponse(
            {
                "detail": "Active run not found.",
            },
            status=404,
        )

    try:
        return JsonResponse(
            analysis.cancel_after_current_frame().to_dict()
        )

    except ArcadiaError as exc:
        return JsonResponse(
            {
                "detail": str(exc),
            },
            status=409,
        )


@require_GET
def run_stream(
    request: HttpRequest,
    run_id: str,
) -> HttpResponse:
    analysis = get_runtime().analysis
    status = analysis.status()

    if (
        run_id != status.run_id
        or status.state == "idle"
    ):
        return HttpResponse(
            "Run stream not found.",
            status=404,
        )

    def frames() -> Iterator[bytes]:
        version = 0

        while True:
            jpeg, newest_version, state = (
                analysis.preview(version)
            )

            if (
                jpeg is not None
                and newest_version > version
            ):
                version = newest_version

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + (
                        f"Content-Length: {len(jpeg)}"
                        "\r\n\r\n"
                    ).encode()
                    + jpeg
                    + b"\r\n"
                )
                continue

            if state in (
                "completed",
                "cancelled",
                "failed",
            ):
                return

    response = StreamingHttpResponse(
        frames(),
        content_type=(
            TOOLS["priority-map"]
            .presentation.stream_content_type
            + "; boundary=frame"
        ),
    )
    response["Cache-Control"] = "no-store"

    return response


@require_GET
def run_artifacts(
    request: HttpRequest,
    run_id: str,
) -> JsonResponse:
    runtime = get_runtime()

    try:
        store = ArtifactStore(
            runtime.config_store.load()
            .priority_map.output.root,
            run_id,
        )

        artifacts = [
            {
                "path": record.path,
                "size_bytes": record.size_bytes,
                "content_type": record.content_type,
                "inline_url": (
                    "/client/priority-map/runs/"
                    f"{run_id}/artifacts/"
                    f"{quote(record.path)}/"
                ),
                "download_url": (
                    "/client/priority-map/runs/"
                    f"{run_id}/artifacts/"
                    f"{quote(record.path)}/"
                    "?download=1"
                ),
            }
            for record in store.list()
        ]

    except (ArcadiaError, OSError) as exc:
        return JsonResponse(
            {
                "detail": str(exc),
            },
            status=404,
        )

    payload: dict[str, Any] = {
        "run_id": run_id,
        "artifacts": artifacts,
    }

    for artifact in artifacts:
        if artifact["path"] == "effective_settings.json":
            payload["effective_settings"] = artifact
        elif artifact["path"] == "analysis.log":
            payload["log"] = artifact

    return JsonResponse(payload)


@require_GET
def run_artifact(
    request: HttpRequest,
    run_id: str,
    artifact_path: str,
) -> HttpResponse:
    runtime = get_runtime()

    try:
        store = ArtifactStore(
            runtime.config_store.load()
            .priority_map.output.root,
            run_id,
        )
        artifact = store.resolve(artifact_path)

    except (ArcadiaError, OSError) as exc:
        return HttpResponse(
            str(exc),
            status=404,
        )

    extension = artifact.suffix.lower()
    inline = extension in (
        TOOLS["priority-map"]
        .presentation.inline_artifact_extensions
    )
    download = (
        request.GET.get("download") == "1"
    )

    content_type = next(
        (
            record.content_type
            for record in store.list()
            if record.path == artifact_path
        ),
        None,
    )

    return FileResponse(
        artifact.open("rb"),
        as_attachment=download or not inline,
        filename=artifact.name,
        content_type=content_type,
    )


@require_GET
def results(request: HttpRequest) -> HttpResponse:
    return render(
        request,
        "web/results.html",
        {
            "analysis": (
                get_runtime().analysis.status()
            )
        },
    )


# ============================================================
# Direct endpoint test
# ============================================================


@require_GET
def endpoint_test(
    request: HttpRequest,
) -> HttpResponse:
    return render(
        request,
        "web/endpoint_test.html",
        {
            "form": EndpointTestForm(),
        },
    )


@require_POST
def run_endpoint_test(
    request: HttpRequest,
) -> HttpResponse:
    form = EndpointTestForm(request.POST)
    response_text = None

    if form.is_valid():
        try:
            endpoint = ServiceEndpoint(
                form.cleaned_data["endpoint_host"],
                form.cleaned_data["endpoint_port"],
                "llm",
            )
            response_text = (
                LLMClient(endpoint)
                .chat(
                    form.cleaned_data["prompt"]
                )
                .text
            )

        except (
            ArcadiaError,
            ValueError,
            OSError,
        ) as exc:
            form.add_error(
                None,
                str(exc),
            )

    return render(
        request,
        "web/endpoint_test.html",
        {
            "form": form,
            "response_text": response_text,
        },
    )


# ============================================================
# Saved-settings Test Chat
# ============================================================


@require_POST
def test_llm_chat(
    request: HttpRequest,
) -> JsonResponse:
    return _test_chat(
        request,
        role="llm",
        require_vision=False,
    )


@require_POST
def test_visual_llm_chat(
    request: HttpRequest,
) -> JsonResponse:
    return _test_chat(
        request,
        role="visual_llm",
        require_vision=True,
    )


def _test_chat(
    request: HttpRequest,
    *,
    role: str,
    require_vision: bool,
) -> JsonResponse:
    from core.errors import (
        InferenceError,
        ServiceError,
    )

    runtime = get_runtime()

    try:
        runtime.analysis.assert_configuration_mutable()
        config = runtime.config_store.load()

        backing_role, configured = (
            _resolve_llm_backing_service(
                config,
                role,
            )
        )

        node = config.nodes.get(configured.node)

        if node is None:
            raise ValueError(
                f"{_service_label(backing_role)} "
                f"references unknown node "
                f"{configured.node!r}."
            )

        spec = _normalized_llm_spec(
            configured,
            backing_role=backing_role,
            node=node,
        )

        if (
            require_vision
            and not spec.settings.get(
                "vision_enabled",
                False,
            )
        ):
            return JsonResponse(
                {
                    "error": (
                        f"{_service_label(backing_role)} "
                        "does not have vision enabled. "
                        "Enable vision and save the "
                        "model settings first."
                    )
                },
                status=400,
            )

        prompt = request.POST.get(
            "prompt",
            "",
        ).strip()

        if not prompt:
            prompt = (
                "Describe the contents of this image."
                if require_vision
                else "Hello. Identify your model and "
                "confirm that chat is working."
            )

        images: list[tuple[str, bytes]] | None = None

        if require_vision:
            uploaded = request.FILES.get("image")

            if uploaded is None:
                return JsonResponse(
                    {
                        "error": (
                            "An image is required for "
                            "visual chat."
                        )
                    },
                    status=400,
                )

            if uploaded.size <= 0:
                return JsonResponse(
                    {
                        "error": (
                            "The uploaded image is empty."
                        )
                    },
                    status=400,
                )

            if uploaded.size > _IMAGE_UPLOAD_LIMIT:
                return JsonResponse(
                    {
                        "error": (
                            "Image must be 10 MB or smaller."
                        )
                    },
                    status=400,
                )

            content_type = (
                uploaded.content_type or ""
            ).lower()

            if content_type not in _ALLOWED_IMAGE_TYPES:
                return JsonResponse(
                    {
                        "error": (
                            "Unsupported image type. "
                            "Use JPEG, PNG, WebP, or GIF."
                        )
                    },
                    status=400,
                )

            raw_image = uploaded.read()

            if not raw_image:
                return JsonResponse(
                    {
                        "error": (
                            "The uploaded image is empty."
                        )
                    },
                    status=400,
                )

            images = [
                (
                    content_type,
                    raw_image,
                )
            ]

        if node.mode == "local":
            running_status = next(
                (
                    status
                    for status in (
                        runtime.controller.list_services()
                    )
                    if status.port == spec.port
                ),
                None,
            )

            same_running_service = (
                running_status is not None
                and running_status.running
                and running_status.service_type
                == spec.service_type
                and running_status.settings
                == spec.settings
            )

            if same_running_service:
                endpoint = ServiceEndpoint(
                    host=str(
                        spec.settings.get(
                            "bind_host",
                            "127.0.0.1",
                        )
                    ),
                    port=spec.port,
                    service_type=spec.service_type,
                )
            else:
                endpoint = runtime.controller.start(
                    spec
                )

        else:
            if node.instruction_port is None:
                raise ValueError(
                    f"Remote node {configured.node!r} "
                    "has no instruction port."
                )

            endpoint = InstructionClient(
                node.host,
                node.instruction_port,
            ).start_service(spec)

        defaults: dict[str, Any] = dict(
            generation_settings(spec.settings)
        )

        model_alias = spec.settings.get(
            "model_alias"
        )

        if not isinstance(model_alias, str) or not model_alias:
            model_alias = (
                "visual-model"
                if backing_role == "visual_llm"
                else "logical-model"
            )

        defaults["model"] = model_alias

        client = LLMClient(
            endpoint,
            role_defaults=defaults,
        )
        result = client.chat(
            prompt,
            images=images,
        )

        return JsonResponse(
            {
                "response": result.text,
                "requested_role": role,
                "backing_service": backing_role,
                "endpoint": endpoint.base_url,
                "model": model_alias,
            }
        )

    except ValueError as exc:
        return JsonResponse(
            {
                "error": str(exc),
            },
            status=400,
        )

    except (
        InferenceError,
        ServiceError,
        ArcadiaError,
        OSError,
    ) as exc:
        return JsonResponse(
            {
                "error": str(exc),
            },
            status=502,
        )

    except Exception as exc:
        return JsonResponse(
            {
                "error": (
                    "Unexpected Test Chat failure: "
                    f"{exc}"
                )
            },
            status=500,
        )