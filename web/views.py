from __future__ import annotations

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from core.inference.llm_client import LLMClient
from core.services.controller import ServiceController
from core.services.instruction_client import InstructionClient
from core.services.specs import ServiceEndpoint, ServiceSpec

from .forms import PromptForm, ServiceForm


_local_controller = ServiceController(public_host="127.0.0.1")


def home(request: HttpRequest) -> HttpResponse:
    service_form = ServiceForm(prefix="service")
    prompt_form = PromptForm(prefix="prompt")
    context: dict[str, object] = {
        "service_form": service_form,
        "prompt_form": prompt_form,
    }

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "start":
            service_form = ServiceForm(request.POST, prefix="service")
            context["service_form"] = service_form

            if service_form.is_valid():
                spec = ServiceSpec(
                    service_type=service_form.cleaned_data["service_type"],
                    port=service_form.cleaned_data["inference_port"],
                    settings=service_form.cleaned_data["settings_json"],
                )

                if service_form.cleaned_data["location"] == "local":
                    endpoint = _local_controller.start(spec)
                else:
                    client = InstructionClient(
                        service_form.cleaned_data["remote_host"],
                        service_form.cleaned_data["instruction_port"],
                    )
                    endpoint = client.start_service(spec)

                context["message"] = f"Service started at {endpoint.base_url}"

        elif action == "prompt":
            prompt_form = PromptForm(request.POST, prefix="prompt")
            context["prompt_form"] = prompt_form

            if prompt_form.is_valid():
                endpoint = ServiceEndpoint(
                    host=prompt_form.cleaned_data["endpoint_host"],
                    port=prompt_form.cleaned_data["endpoint_port"],
                    service_type="llm",
                )
                context["response"] = LLMClient(endpoint).chat(
                    prompt_form.cleaned_data["prompt"]
                )

    return render(request, "web/home.html", context)
