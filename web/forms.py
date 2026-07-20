from __future__ import annotations

import ipaddress
import math
import re
import shlex
from typing import Any, ClassVar

from django import forms

from core.config import ConfiguredService, HostListenerConfig, NodeConfig, PipelineConfig
from core.errors import ConfigurationError
from core.services.llm_settings import (
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    format_hf_source,
    parse_additional_server_arguments,
    parse_hf_source,
    validate_llm_settings,
)
from core.services.sam_checkpoint import SAMCheckpointStore
from core.services.specs import ServiceSpec, ServiceType


def _finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise forms.ValidationError(f"{label} must be a finite number.")
    return value


def _ipv4(value: str, label: str) -> str:
    try:
        address = ipaddress.IPv4Address(value.strip())
    except (ipaddress.AddressValueError, AttributeError) as exc:
        raise forms.ValidationError(f"{label} must be a valid IPv4 address.") from exc
    if address.is_unspecified:
        raise forms.ValidationError(f"{label} cannot be 0.0.0.0.")
    return str(address)


class HostListenerForm(forms.Form):
    host = forms.CharField(label="IP address", initial="127.0.0.1")
    port = forms.IntegerField(label="Instruction port", min_value=1, max_value=65535, initial=9000)

    def clean_host(self) -> str:
        return _ipv4(self.cleaned_data["host"], "IP address")

    def to_config(self) -> HostListenerConfig:
        try:
            return HostListenerConfig(self.cleaned_data["host"], self.cleaned_data["port"])
        except ConfigurationError as exc:
            raise forms.ValidationError(str(exc)) from exc


class HostSAMCheckpointForm(forms.Form):
    checkpoint = forms.CharField(
        label="SAM3 checkpoint path",
        required=False,
        widget=forms.TextInput(attrs={"readonly": "readonly", "placeholder": "Upload a .pt checkpoint"}),
    )
    device = forms.ChoiceField(
        label="Preferred SAM3 device",
        choices=(("auto", "Automatic"), ("cuda", "CUDA"), ("mps", "Apple MPS"), ("cpu", "CPU")),
        initial="auto",
        required=False,
    )

    def clean_checkpoint(self) -> str:
        value = self.cleaned_data["checkpoint"].strip()
        if not value:
            return ""
        try:
            return str(SAMCheckpointStore.validate_checkpoint_path(value))
        except (ValueError, OSError) as exc:
            raise forms.ValidationError(str(exc)) from exc


class RemoteNodeForm(forms.Form):
    name = forms.CharField(label="Name", max_length=63)
    host = forms.CharField(label="Instruction-server IP", max_length=255)
    instruction_port = forms.IntegerField(label="Instruction port", min_value=1, max_value=65535, initial=9000)

    def clean_name(self) -> str:
        value = re.sub(r"\s+", "-", self.cleaned_data["name"].strip().lower())
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,62}", value):
            raise forms.ValidationError(
                "Use 1–63 letters, numbers, spaces, hyphens, or underscores; start with a letter or number."
            )
        if value == "local":
            raise forms.ValidationError("'local' is reserved for this computer.")
        return value

    def clean_host(self) -> str:
        return _ipv4(self.cleaned_data["host"], "Instruction-server IP")

    def to_config(self, *, extra: dict[str, Any] | None = None) -> NodeConfig:
        return NodeConfig(
            mode="remote",
            host=self.cleaned_data["host"],
            instruction_port=self.cleaned_data["instruction_port"],
            extra=extra or {},
        )


class NodeServiceForm(forms.Form):
    node = forms.ChoiceField(label="Compute node")
    inference_port = forms.IntegerField(label="Inference port", min_value=1, max_value=65535)
    bind_host = forms.CharField(
        label="Inference IP",
        help_text=(
            "Defaults to the selected compute node IP. Edit only when inference uses another reachable interface."
        ),
    )

    def __init__(
        self,
        *args: Any,
        nodes: dict[str, NodeConfig] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.nodes = nodes or {"local": NodeConfig("local", "127.0.0.1")}
        self.fields["node"].choices = [
            (
                name,
                "This computer" if name == "local" else f"{name} ({node.host})",
            )
            for name, node in sorted(self.nodes.items(), key=lambda item: (item[0] != "local", item[0].lower()))
        ]
        self.initial.setdefault("node", "local")
        self.initial.setdefault("bind_host", self.nodes.get("local", NodeConfig("local", "127.0.0.1")).host)

    def clean_bind_host(self) -> str:
        return _ipv4(self.cleaned_data["bind_host"], "Inference IP")

    def clean(self) -> dict[str, Any]:
        cleaned = super().clean()
        if cleaned.get("node") not in self.nodes:
            self.add_error("node", "Unknown compute node.")
        return cleaned


class LLMServiceForm(NodeServiceForm):
    service_type: ClassVar[ServiceType] = "llm"
    default_port: ClassVar[int] = 8081

    inference_port = forms.IntegerField(label="Inference port", min_value=1, max_value=65535, initial=8081)
    hf_source = forms.CharField(
        label="Hugging Face model",
        help_text=(
            "Enter owner/repository or paste an exact huggingface.co .gguf file link. "
            "An exact link is required when a repository contains multiple usable models."
        ),
    )
    vision_enabled = forms.BooleanField(label="Enable vision", required=False)
    mmproj_source = forms.CharField(
        label="Projector model",
        required=False,
        help_text="Enter owner/repository or an exact huggingface.co projector .gguf link.",
    )
    n_ctx = forms.IntegerField(label="Context size", min_value=1, initial=DEFAULT_CONTEXT_SIZE)
    max_tokens = forms.IntegerField(label="Max output tokens", min_value=1, initial=DEFAULT_MAX_TOKENS)
    temperature = forms.FloatField(label="Temperature", min_value=0, initial=DEFAULT_TEMPERATURE)
    additional_arguments = forms.CharField(
        label="Additional llama-server arguments",
        required=False,
        widget=forms.Textarea(attrs={"rows": 8, "spellcheck": "false"}),
        help_text=(
            "Separate arguments with spaces, commas, or line breaks. Quote values that contain commas or spaces. "
            'Example: --flash-attn on, --batch-size 2048, --tensor-split "1,1"'
        ),
    )

    def __init__(
        self,
        *args: Any,
        nodes: dict[str, NodeConfig] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, nodes=nodes, **kwargs)
        self.fields["inference_port"].initial = self.default_port
        self.initial.setdefault("inference_port", self.default_port)
        self._prior_settings: dict[str, Any] = {}
        self.legacy_local_model = False

    def clean_hf_source(self) -> str:
        value = self.cleaned_data["hf_source"].strip()
        try:
            parse_hf_source(value)
        except ValueError as exc:
            raise forms.ValidationError(str(exc)) from exc
        return value

    def clean_mmproj_source(self) -> str:
        value = self.cleaned_data.get("mmproj_source", "").strip()
        if not value:
            return ""
        try:
            parse_hf_source(value)
        except ValueError as exc:
            raise forms.ValidationError(str(exc)) from exc
        return value

    def clean_temperature(self) -> float:
        return _finite(self.cleaned_data["temperature"], "Temperature")

    def clean_additional_arguments(self) -> list[str]:
        try:
            return parse_additional_server_arguments(self.cleaned_data.get("additional_arguments", ""))
        except ValueError as exc:
            raise forms.ValidationError(str(exc)) from exc

    def clean(self) -> dict[str, Any]:
        cleaned = super().clean()
        if cleaned.get("vision_enabled") and not cleaned.get("mmproj_source"):
            self.add_error("mmproj_source", "A projector model is required when vision is enabled.")
        return cleaned

    def _settings(self) -> dict[str, Any]:
        model = parse_hf_source(self.cleaned_data["hf_source"])
        settings: dict[str, Any] = {
            "hf_repo": model.repo_id,
            "hf_revision": model.revision,
            "bind_host": self.cleaned_data["bind_host"],
            "n_ctx": self.cleaned_data["n_ctx"],
            "vision_enabled": bool(self.cleaned_data.get("vision_enabled")),
            "temperature": self.cleaned_data["temperature"],
            "max_tokens": self.cleaned_data["max_tokens"],
            "extra_args": list(self.cleaned_data.get("additional_arguments", [])),
        }
        if model.filename:
            settings["hf_file"] = model.filename
        if settings["vision_enabled"]:
            projector = parse_hf_source(self.cleaned_data["mmproj_source"])
            settings.update(
                {
                    "mmproj_repo": projector.repo_id,
                    "mmproj_revision": projector.revision,
                }
            )
            if projector.filename:
                settings["mmproj_file"] = projector.filename
        return validate_llm_settings(settings)

    def to_spec(self) -> ServiceSpec:
        return ServiceSpec(self.service_type, self.cleaned_data["inference_port"], self._settings())

    def initial_from(self, configured: ConfiguredService | None) -> None:
        if configured is None:
            return
        settings = configured.settings
        self._prior_settings = dict(settings)
        self.initial.update(
            {
                "node": configured.node,
                "inference_port": configured.port,
                "bind_host": settings.get(
                    "bind_host", self.nodes.get(configured.node, NodeConfig("local", "127.0.0.1")).host
                ),
                "n_ctx": settings.get("n_ctx", DEFAULT_CONTEXT_SIZE),
                "vision_enabled": bool(settings.get("vision_enabled", False)),
                "temperature": settings.get("temperature", DEFAULT_TEMPERATURE),
                "max_tokens": settings.get("max_tokens", DEFAULT_MAX_TOKENS),
                "hf_source": format_hf_source(
                    str(settings.get("hf_repo", "")),
                    str(settings.get("hf_revision", "main")),
                    str(settings["hf_file"]) if settings.get("hf_file") else None,
                ),
                "mmproj_source": (
                    format_hf_source(
                        str(settings.get("mmproj_repo", "")),
                        str(settings.get("mmproj_revision", "main")),
                        str(settings["mmproj_file"]) if settings.get("mmproj_file") else None,
                    )
                    if settings.get("mmproj_repo")
                    else ""
                ),
                "additional_arguments": shlex.join(settings.get("extra_args", []))
                if isinstance(settings.get("extra_args"), list)
                else "",
            }
        )
        self.legacy_local_model = bool(settings.get("model_path") or settings.get("hf_file_pattern"))


class VisualLLMServiceForm(LLMServiceForm):
    service_type: ClassVar[ServiceType] = "visual_llm"
    default_port: ClassVar[int] = 8082

    inference_port = forms.IntegerField(label="Inference port", min_value=1, max_value=65535, initial=8082)
    vision_enabled = forms.BooleanField(
        label="Enable vision",
        required=False,
        initial=True,
        disabled=True,
        help_text="Vision is always enabled for a separate Visual LLM.",
    )

    def clean(self) -> dict[str, Any]:
        cleaned = super().clean()
        cleaned["vision_enabled"] = True
        if not cleaned.get("mmproj_source"):
            self.add_error("mmproj_source", "A projector model is required for a separate Visual LLM.")
        return cleaned

    def initial_from(self, configured: ConfiguredService | None) -> None:
        super().initial_from(configured)
        self.initial["vision_enabled"] = True


class SAMServiceForm(NodeServiceForm):
    inference_port = forms.IntegerField(label="Inference port", min_value=1, max_value=65535, initial=8090)
    checkpoint = forms.CharField(
        label="Checkpoint path",
        help_text=(
            "Path on the selected compute node. Browse uploads a .pt checkpoint into huggingface/models on that node."
        ),
        widget=forms.TextInput(attrs={"placeholder": ".../huggingface/models/sam3.pt"}),
    )
    confidence = forms.FloatField(label="Default confidence", min_value=0, max_value=1, initial=0.25)
    device = forms.ChoiceField(
        label="Inference device",
        choices=(("auto", "Automatic"), ("cuda", "CUDA"), ("mps", "Apple MPS"), ("cpu", "CPU")),
        initial="auto",
        required=False,
        help_text="Automatic prefers CUDA, then Apple MPS, then CPU on the selected compute host.",
    )
    additional_arguments = forms.CharField(
        label="Additional SAM arguments",
        required=False,
        widget=forms.Textarea(attrs={"rows": 4, "spellcheck": "false"}),
    )

    def clean_additional_arguments(self) -> list[str]:
        value = self.cleaned_data.get("additional_arguments", "")
        try:
            return shlex.split(value)
        except ValueError as exc:
            raise forms.ValidationError(f"Invalid argument syntax: {exc}") from exc

    def to_spec(self) -> ServiceSpec:
        return ServiceSpec(
            "sam3",
            self.cleaned_data["inference_port"],
            {
                "bind_host": self.cleaned_data["bind_host"],
                "checkpoint": self.cleaned_data["checkpoint"].strip(),
                "confidence": self.cleaned_data["confidence"],
                "device": self.cleaned_data["device"] or "auto",
                "extra_args": list(self.cleaned_data.get("additional_arguments", [])),
            },
        )

    def initial_from(self, configured: ConfiguredService | None) -> None:
        if configured is None:
            return
        settings = configured.settings
        self.initial.update(
            {
                "node": configured.node,
                "inference_port": configured.port,
                "bind_host": settings.get(
                    "bind_host", self.nodes.get(configured.node, NodeConfig("local", "127.0.0.1")).host
                ),
                "checkpoint": settings.get("checkpoint", ""),
                "confidence": settings.get("confidence", 0.25),
                "device": settings.get("device", "auto"),
                "additional_arguments": shlex.join(settings.get("extra_args", []))
                if isinstance(settings.get("extra_args"), list)
                else "",
            }
        )


class PipelineForm(forms.Form):
    task = forms.CharField(initial="Find cars")
    debrief = forms.CharField(required=False, widget=forms.Textarea(attrs={"rows": 3}))
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
    def from_config(cls, config: PipelineConfig) -> "PipelineForm":
        initial = config.to_dict()
        initial["prompts"] = ", ".join(config.prompts)
        return cls(initial=initial)

    def to_config(self) -> PipelineConfig:
        values = dict(self.cleaned_data)
        values["prompts"] = [item.strip() for item in values.pop("prompts", "").split(",") if item.strip()]
        for key in ("gps_csv", "camera_intrinsics", "scene_model"):
            values[key] = values.get(key) or None
        return PipelineConfig(**values)


class AnalysisForm(forms.Form):
    input_path = forms.CharField(required=False, help_text="Local image folder, image file, or video path.")
    upload_id = forms.CharField(required=False)

    def clean(self) -> dict[str, Any]:
        cleaned = super().clean()
        input_path = (cleaned.get("input_path") or "").strip()
        upload_id = (cleaned.get("upload_id") or "").strip()
        if bool(input_path) == bool(upload_id):
            raise forms.ValidationError("Choose exactly one existing path or retained upload.")
        cleaned["input_path"] = input_path
        cleaned["upload_id"] = upload_id
        return cleaned


class EndpointTestForm(forms.Form):
    endpoint_host = forms.CharField(initial="127.0.0.1")
    endpoint_port = forms.IntegerField(min_value=1, max_value=65535, initial=8081)
    prompt = forms.CharField(widget=forms.Textarea(attrs={"rows": 4}))
