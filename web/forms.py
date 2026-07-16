from __future__ import annotations

from typing import Any, cast

from django import forms

from core.config import ConfiguredService, NodeConfig, PipelineConfig
from core.services.specs import ServiceSpec, ServiceType


class NodeForm(forms.Form):
    mode = forms.ChoiceField(choices=[("local", "Local"), ("remote", "Remote")])
    host = forms.CharField(max_length=255, initial="127.0.0.1")
    instruction_port = forms.IntegerField(min_value=1, max_value=65535, required=False, initial=9000)

    def clean(self) -> dict[str, Any]:
        cleaned = super().clean()
        if cleaned.get("mode") == "remote" and not cleaned.get("instruction_port"):
            self.add_error("instruction_port", "Remote nodes require an instruction port.")
        return cleaned

    def to_config(self) -> NodeConfig:
        return NodeConfig(
            mode=self.cleaned_data["mode"],
            host=self.cleaned_data["host"],
            instruction_port=self.cleaned_data.get("instruction_port"),
        )


class ServiceForm(forms.Form):
    node = forms.ChoiceField()
    inference_port = forms.IntegerField(min_value=1, max_value=65535)
    settings_json = forms.JSONField(required=False, initial=dict)

    def __init__(self, *args: Any, nodes: dict[str, NodeConfig] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        node_map = nodes or {"local": NodeConfig("local", "127.0.0.1")}
        self.fields["node"].choices = [(name, f"{name} ({node.host})") for name, node in node_map.items()]

    def clean_settings_json(self) -> dict[str, Any]:
        value = self.cleaned_data.get("settings_json") or {}
        if not isinstance(value, dict):
            raise forms.ValidationError("Settings must be a JSON object.")
        return value

    def to_spec(self, service_type: str) -> ServiceSpec:
        return ServiceSpec(
            service_type=cast(ServiceType, service_type),
            port=self.cleaned_data["inference_port"],
            settings=self.cleaned_data["settings_json"],
        )

    def initial_from(self, configured: ConfiguredService | None) -> None:
        if configured is None:
            return
        self.initial.update(
            {
                "node": configured.node,
                "inference_port": configured.port,
                "settings_json": configured.settings,
            }
        )


class LLMServiceForm(ServiceForm):
    pass


class SAMServiceForm(ServiceForm):
    pass


class PipelineForm(forms.Form):
    task = forms.CharField(initial="Find cars")
    debrief = forms.CharField(required=False, widget=forms.Textarea)
    prompts = forms.CharField(required=False, help_text="Comma-separated optional labels")
    sam_step = forms.IntegerField(min_value=1, initial=5)
    sam_confidence = forms.FloatField(min_value=0, max_value=1, initial=0.25)
    sam_resize = forms.IntegerField(min_value=1, required=False)
    max_image_edge = forms.IntegerField(min_value=1, required=False, initial=640)
    run_at_source_fps = forms.BooleanField(required=False)
    debug = forms.BooleanField(required=False)
    record = forms.BooleanField(required=False, initial=True)
    panoramic = forms.BooleanField(required=False)
    graph_agent = forms.BooleanField(required=False)
    gps_csv = forms.CharField(required=False)
    camera_intrinsics = forms.CharField(required=False)
    scene_model = forms.CharField(required=False)

    @classmethod
    def from_config(cls, config: PipelineConfig) -> PipelineForm:
        return cls(
            initial={
                "task": config.task,
                "debrief": config.debrief,
                "prompts": ", ".join(config.prompts),
                "sam_step": config.sam_step,
                "sam_confidence": config.sam_confidence,
                "sam_resize": config.sam_resize,
                "max_image_edge": config.max_image_edge,
                "run_at_source_fps": config.run_at_source_fps,
                "debug": config.debug,
                "record": config.record,
                "panoramic": config.panoramic,
                "graph_agent": config.graph_agent,
                "gps_csv": config.gps_csv or "",
                "camera_intrinsics": config.camera_intrinsics or "",
                "scene_model": config.scene_model or "",
            }
        )

    def to_config(self) -> PipelineConfig:
        data = dict(self.cleaned_data)
        data["prompts"] = [item.strip() for item in data.pop("prompts", "").split(",") if item.strip()]
        return PipelineConfig(**data)


class AnalysisForm(forms.Form):
    input_path = forms.CharField(help_text="A local image folder, image file, or video path.")


class EndpointTestForm(forms.Form):
    endpoint_host = forms.CharField(initial="127.0.0.1")
    endpoint_port = forms.IntegerField(min_value=1, max_value=65535)
    prompt = forms.CharField(widget=forms.Textarea)
