from __future__ import annotations

from typing import Any

from django.contrib import messages
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET, require_POST

from core.config import ConfiguredService
from core.errors import ArcadiaError
from core.inference.llm_client import LLMClient
from core.services.instruction_client import InstructionClient
from core.services.specs import ServiceEndpoint
from web.forms import AnalysisForm, EndpointTestForm, LLMServiceForm, NodeForm, PipelineForm, SAMServiceForm
from web.runtime import get_runtime


def _service_form(name: str, config: Any, data: Any = None):
    form_class = LLMServiceForm if name == "llm" else SAMServiceForm
    form = form_class(data, nodes=config.nodes) if data is not None else form_class(nodes=config.nodes)
    form.initial_from(config.services.get(name))
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
        spec = form.to_spec("sam3" if service_name == "sam3" else "llm")
        config.services[service_name] = ConfiguredService(node=form.cleaned_data["node"], spec=spec)
        runtime.config_store.save(config)
        node = config.nodes[form.cleaned_data["node"]]
        if node.mode == "local":
            endpoint = runtime.controller.start(spec)
        else:
            client = InstructionClient(node.host, node.instruction_port or 9000)
            endpoint = client.start_service(spec)
        messages.success(request, f"{service_name.upper()} ready at {endpoint.base_url}")
    except (ArcadiaError, ValueError, OSError) as exc:
        messages.error(request, str(exc))
    return redirect("services")


@require_POST
def stop_service(request: HttpRequest, service_name: str) -> HttpResponse:
    runtime = get_runtime()
    try:
        runtime.analysis.assert_configuration_mutable()
        config = runtime.config_store.load()
        configured = config.services.get(service_name)
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
        configured = next((service for service in config.services.values() if service.port == port), None)
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
            "config": config,
            "node_forms": {name: NodeForm(initial=node.to_dict()) for name, node in config.nodes.items()},
        },
    )


@require_POST
def save_node(request: HttpRequest, node_name: str) -> HttpResponse:
    runtime = get_runtime()
    try:
        runtime.analysis.assert_configuration_mutable()
        config = runtime.config_store.load()
        form = NodeForm(request.POST)
        if not form.is_valid():
            return render(request, "web/nodes.html", {"config": config, "node_forms": {node_name: form}}, status=400)
        config.nodes[node_name] = form.to_config()
        runtime.config_store.save(config)
        messages.success(request, f"Saved node {node_name}.")
    except (ArcadiaError, ValueError) as exc:
        messages.error(request, str(exc))
    return redirect("nodes")


@require_GET
def analysis_page(request: HttpRequest) -> HttpResponse:
    runtime = get_runtime()
    config = runtime.config_store.load()
    return render(
        request,
        "web/analysis.html",
        {
            "config": config,
            "pipeline_form": PipelineForm.from_config(config.pipeline),
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
        config.pipeline = form.to_config()
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
        messages.error(request, "Choose an input path before starting the analysis.")
        return redirect("analysis")
    try:
        config = runtime.config_store.load()
        runtime.analysis.start(form.cleaned_data["input_path"], config)
        messages.success(request, "Analysis started.")
    except (ArcadiaError, ValueError, OSError) as exc:
        messages.error(request, str(exc))
    return redirect("results")


@require_GET
def analysis_status(request: HttpRequest) -> JsonResponse:
    return JsonResponse(get_runtime().analysis.status().to_dict())


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
