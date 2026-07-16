from __future__ import annotations

import ipaddress
import re
import shlex
from typing import Any

from django import forms

from core.config import ConfiguredService, HostListenerConfig, NodeConfig, PipelineConfig
from core.errors import ConfigurationError
from core.services.specs import ServiceSpec


class HostListenerForm(forms.Form):
    host = forms.CharField(label="IP address", initial="127.0.0.1")
    port = forms.IntegerField(label="Instruction port", min_value=1, max_value=65535, initial=9000)

    def clean_host(self) -> str:
        value = self.cleaned_data["host"]
        try:
            return HostListenerConfig(host=value).host
        except ConfigurationError as exc:
            raise forms.ValidationError(str(exc)) from exc

    def to_config(self) -> HostListenerConfig:
        return HostListenerConfig(host=self.cleaned_data["host"], port=self.cleaned_data["port"])


class RemoteNodeForm(forms.Form):
    name = forms.CharField(label="Name", max_length=63)
    host = forms.CharField(label="Instruction-server IP", max_length=255)
    instruction_port = forms.IntegerField(label="Instruction port", min_value=1, max_value=65535)

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
        value = self.cleaned_data["host"].strip()
        try:
            address = ipaddress.ip_address(value)
        except ValueError as exc:
            raise forms.ValidationError("IP address must be a valid IPv4 address.") from exc
        if address.version != 4:
            raise forms.ValidationError("IP address must be a valid IPv4 address.")
        return str(address)

    def to_config(self, *, extra: dict[str, Any] | None = None) -> NodeConfig:
        return NodeConfig(
            mode="remote",
            host=self.cleaned_data["host"],
            instruction_port=self.cleaned_data["instruction_port"],
            extra=extra or {},
        )


class ServiceForm(forms.Form):
    node = forms.ChoiceField(label="Run on")
    inference_port = forms.IntegerField(label="Inference port", min_value=1, max_value=65535)
    bind_host = forms.CharField(label="Bind host", initial="0.0.0.0")
    startup_timeout = forms.FloatField(
        label="Startup timeout",
        min_value=1,
        initial=600,
        help_text="Maximum seconds to wait for the model server to become ready.",
    )

    def __init__(self, *args: Any, nodes: dict[str, NodeConfig] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        node_map = nodes or {"local": NodeConfig("local", "127.0.0.1")}
        self.fields["node"].choices = [
            (name, "This computer" if name == "local" else f"{name} ({node.host})") for name, node in node_map.items()
        ]

    def initial_from(self, configured: ConfiguredService | None) -> None:
        if configured is None:
            return
        settings = configured.settings
        self.initial.update(
            {
                "node": configured.node,
                "inference_port": configured.port,
                "bind_host": settings.get("bind_host", "0.0.0.0"),
                "startup_timeout": settings.get("startup_timeout", 600),
            }
        )


class LLMServiceForm(ServiceForm):
    CACHE_TYPE_CHOICES = [("", "Default"), ("f16", "F16"), ("q8_0", "Q8_0"), ("q4_0", "Q4_0")]

    hf_repo = forms.CharField(label="Hugging Face repository", required=True)
    n_ctx = forms.IntegerField(label="Context size", min_value=1, initial=32768)
    temperature = forms.FloatField(label="Temperature", min_value=0, initial=0.2)
    top_k = forms.IntegerField(label="Top K", min_value=0, initial=40)
    min_p = forms.FloatField(label="Min P", min_value=0, max_value=1, initial=0.05)
    top_p = forms.FloatField(label="Top P", min_value=0, max_value=1, initial=0.95)
    model_file_pattern = forms.CharField(label="Model file pattern", required=False)
    model_alias = forms.CharField(label="Model alias", required=False, initial="local-model")
    chat_format = forms.CharField(label="Chat format", required=False)
    vision_enabled = forms.BooleanField(label="Enable vision / MMProj", required=False)
    mmproj_repo = forms.CharField(label="MMProj repository", required=False)
    mmproj_file_pattern = forms.CharField(label="MMProj file pattern", required=False)
    local_bind_host = forms.CharField(label="Local bind host", required=False, initial="127.0.0.1")
    n_gpu_layers = forms.IntegerField(label="GPU layers", min_value=-1, required=False, initial=-1)
    n_threads = forms.IntegerField(label="CPU threads", min_value=1, required=False)
    n_batch = forms.IntegerField(label="Batch size", min_value=1, required=False, initial=2048)
    n_ubatch = forms.IntegerField(label="Microbatch size", min_value=1, required=False, initial=512)
    main_gpu = forms.IntegerField(label="Main GPU", min_value=0, required=False)
    tensor_split = forms.CharField(label="Tensor split", required=False)
    flash_attn = forms.BooleanField(label="Flash attention", required=False, initial=True)
    offload_kqv = forms.BooleanField(label="Offload K/Q/V", required=False, initial=True)
    cache_type_k = forms.ChoiceField(label="K-cache type", choices=CACHE_TYPE_CHOICES, required=False)
    cache_type_v = forms.ChoiceField(label="V-cache type", choices=CACHE_TYPE_CHOICES, required=False)
    use_mmap = forms.BooleanField(label="Use mmap", required=False, initial=True)
    use_mlock = forms.BooleanField(label="Use mlock", required=False)
    numa = forms.BooleanField(label="NUMA", required=False)
    rope_scaling_type = forms.IntegerField(label="RoPE scaling type", min_value=-1, max_value=3, required=False)
    rope_freq_base = forms.FloatField(label="RoPE frequency base", required=False)
    rope_freq_scale = forms.FloatField(label="RoPE frequency scale", required=False)
    yarn_ext_factor = forms.FloatField(label="YaRN extension factor", required=False)
    yarn_attn_factor = forms.FloatField(label="YaRN attention factor", required=False)
    yarn_beta_fast = forms.FloatField(label="YaRN beta fast", required=False)
    yarn_beta_slow = forms.FloatField(label="YaRN beta slow", required=False)
    yarn_orig_ctx = forms.IntegerField(label="YaRN original context", min_value=1, required=False)
    additional_arguments = forms.CharField(
        label="Additional server arguments",
        required=False,
        widget=forms.Textarea,
        help_text="Arguments not controlled above. Shell syntax and owned options are rejected.",
    )

    def __init__(self, *args: Any, nodes: dict[str, NodeConfig] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, nodes=nodes, **kwargs)
        self.nodes = nodes or {"local": NodeConfig("local", "127.0.0.1")}
        self.fields.pop("bind_host")
        self.fields.pop("startup_timeout")
        self.fields["startup_timeout"] = forms.FloatField(
            label="Startup timeout", min_value=1, initial=600, required=False
        )
        self._prior_settings: dict[str, Any] = {}
        self.legacy_local_model = False

    def clean_hf_repo(self) -> str:
        from core.services.llm_settings import validate_hf_repository

        try:
            return validate_hf_repository(self.cleaned_data["hf_repo"])
        except ValueError as exc:
            raise forms.ValidationError(str(exc)) from exc

    def clean_model_file_pattern(self) -> str | None:
        from core.services.llm_settings import validate_gguf_pattern

        try:
            return validate_gguf_pattern(self.cleaned_data.get("model_file_pattern"))
        except ValueError as exc:
            raise forms.ValidationError(str(exc)) from exc

    def clean_mmproj_repo(self) -> str:
        from core.services.llm_settings import validate_hf_repository

        value = self.cleaned_data.get("mmproj_repo", "")
        if not value.strip():
            return ""
        try:
            return validate_hf_repository(value)
        except ValueError as exc:
            raise forms.ValidationError(str(exc)) from exc

    def clean_top_p(self) -> float:
        value = self.cleaned_data["top_p"]
        if value <= 0:
            raise forms.ValidationError("Top P must be greater than zero.")
        return value

    def clean_mmproj_file_pattern(self) -> str | None:
        from core.services.llm_settings import validate_gguf_pattern

        try:
            return validate_gguf_pattern(self.cleaned_data.get("mmproj_file_pattern"))
        except ValueError as exc:
            raise forms.ValidationError(str(exc)) from exc

    def clean_additional_arguments(self) -> list[str]:
        from core.services.llm_settings import parse_additional_server_arguments

        try:
            return parse_additional_server_arguments(self.cleaned_data.get("additional_arguments", ""))
        except ValueError as exc:
            raise forms.ValidationError(str(exc)) from exc

    def clean(self) -> dict[str, Any]:
        cleaned = super().clean()
        if cleaned.get("n_batch") and cleaned.get("n_ubatch") and cleaned["n_batch"] < cleaned["n_ubatch"]:
            self.add_error("n_batch", "Batch size must be at least the microbatch size.")
        split = cleaned.get("tensor_split", "").strip()
        if split:
            try:
                values = [float(value.strip()) for value in split.split(",")]
                if (
                    not values
                    or any(not value >= 0 or not value < float("inf") for value in values)
                    or sum(values) <= 0
                ):
                    raise ValueError
            except ValueError:
                self.add_error("tensor_split", "Tensor split must be finite non-negative values with a positive total.")
            else:
                cleaned["tensor_split"] = ",".join(str(value) for value in values)
        return cleaned

    def to_spec(self) -> ServiceSpec:
        from core.services.llm_settings import RETIRED_SOURCE_KEYS, resolve_inference_bind_host

        cleaned = self.cleaned_data
        settings = {key: value for key, value in self._prior_settings.items() if key not in RETIRED_SOURCE_KEYS}
        settings.update(
            {
                "hf_repo": cleaned["hf_repo"],
                "models_cache_subdir": "huggingface",
                "bind_host": resolve_inference_bind_host(cleaned["node"], self.nodes, cleaned.get("local_bind_host")),
                "n_ctx": cleaned["n_ctx"],
                "temperature": cleaned["temperature"],
                "top_k": cleaned["top_k"],
                "min_p": cleaned["min_p"],
                "top_p": cleaned["top_p"],
            }
        )
        for key in (
            "model_file_pattern",
            "model_alias",
            "chat_format",
            "mmproj_repo",
            "mmproj_file_pattern",
            "n_gpu_layers",
            "n_threads",
            "n_batch",
            "n_ubatch",
            "main_gpu",
            "tensor_split",
            "cache_type_k",
            "cache_type_v",
            "rope_scaling_type",
            "rope_freq_base",
            "rope_freq_scale",
            "yarn_ext_factor",
            "yarn_attn_factor",
            "yarn_beta_fast",
            "yarn_beta_slow",
            "yarn_orig_ctx",
            "startup_timeout",
        ):
            value = cleaned.get(key)
            if value not in (None, ""):
                settings[key] = value
            else:
                settings.pop(key, None)
        for key in ("vision_enabled", "flash_attn", "offload_kqv", "use_mmap", "use_mlock", "numa"):
            settings[key] = bool(cleaned.get(key))
        if cleaned["additional_arguments"]:
            settings["extra_args"] = list(cleaned["additional_arguments"])
        else:
            settings.pop("extra_args", None)
        return ServiceSpec(service_type="llm", port=cleaned["inference_port"], settings=settings)

    def initial_from(self, configured: ConfiguredService | None) -> None:
        if configured is None:
            return
        settings = configured.settings
        self._prior_settings = dict(settings)
        self.initial.update({"node": configured.node, "inference_port": configured.port})
        if settings.get("model_path"):
            self.legacy_local_model = True
        initial = {
            key: settings[key]
            for key in (
                "hf_repo",
                "model_file_pattern",
                "model_alias",
                "chat_format",
                "vision_enabled",
                "mmproj_repo",
                "mmproj_file_pattern",
                "n_ctx",
                "temperature",
                "top_k",
                "min_p",
                "top_p",
                "n_gpu_layers",
                "n_threads",
                "n_batch",
                "n_ubatch",
                "main_gpu",
                "tensor_split",
                "flash_attn",
                "offload_kqv",
                "cache_type_k",
                "cache_type_v",
                "use_mmap",
                "use_mlock",
                "numa",
                "rope_scaling_type",
                "rope_freq_base",
                "rope_freq_scale",
                "yarn_ext_factor",
                "yarn_attn_factor",
                "yarn_beta_fast",
                "yarn_beta_slow",
                "yarn_orig_ctx",
                "startup_timeout",
            )
            if key in settings
        }
        initial["local_bind_host"] = settings.get("bind_host", "127.0.0.1")
        extra_args = settings.get("extra_args")
        if isinstance(extra_args, list) and all(isinstance(item, str) for item in extra_args):
            initial["additional_arguments"] = shlex.join(extra_args)
        if settings.get("hf_repo") and settings.get("hf_file") and not settings.get("model_file_pattern"):
            initial["model_file_pattern"] = settings["hf_file"]
        self.initial.update(initial)


class SAMServiceForm(ServiceForm):
    _OWNED_FLAGS = {"--host", "--port", "--checkpoint", "--confidence"}
    _DISALLOWED_ARGUMENTS = {
        "--",
        "-c",
        "-m",
        "-h",
        "--help",
        "--command",
        "--python_executable",
        "bash",
        "cmd",
        "powershell",
        "python",
        "python3",
        "sh",
        "zsh",
    }

    inference_port = forms.IntegerField(label="Inference port", min_value=1, max_value=65535, initial=8090)
    checkpoint = forms.CharField(label="Checkpoint path")
    confidence = forms.FloatField(label="Default confidence", min_value=0, max_value=1, initial=0.25)
    additional_arguments = forms.CharField(
        label="Additional arguments",
        required=False,
        widget=forms.Textarea,
        help_text="Optional SAM3 server arguments not covered above.",
    )

    def clean_additional_arguments(self) -> list[str]:
        try:
            args = shlex.split(self.cleaned_data.get("additional_arguments", ""))
        except ValueError as exc:
            raise forms.ValidationError(f"Invalid additional argument syntax: {exc}") from exc
        for argument in args:
            if argument in self._DISALLOWED_ARGUMENTS or any(
                argument == flag or argument.startswith(f"{flag}=") for flag in self._OWNED_FLAGS
            ):
                raise forms.ValidationError(f"Additional arguments cannot override {argument!r}.")
            if any(character in argument for character in (";", "|", "&", "`", "$", "<", ">")):
                raise forms.ValidationError("Additional arguments cannot contain shell-related syntax.")
        return args

    def to_spec(self) -> ServiceSpec:
        cleaned = self.cleaned_data
        settings: dict[str, Any] = {
            "checkpoint": cleaned["checkpoint"].strip(),
            "bind_host": cleaned["bind_host"],
            "startup_timeout": cleaned["startup_timeout"],
            "confidence": cleaned["confidence"],
        }
        if cleaned["additional_arguments"]:
            settings["extra_args"] = cleaned["additional_arguments"]
        return ServiceSpec(service_type="sam3", port=cleaned["inference_port"], settings=settings)

    def initial_from(self, configured: ConfiguredService | None) -> None:
        super().initial_from(configured)
        if configured is None:
            return
        settings = configured.settings
        self.initial.update(
            {
                "checkpoint": settings.get("checkpoint", ""),
                "confidence": settings.get("confidence", 0.25),
                "additional_arguments": (
                    shlex.join(settings["extra_args"])
                    if isinstance(settings.get("extra_args"), list)
                    and all(isinstance(argument, str) for argument in settings["extra_args"])
                    else ""
                ),
            }
        )


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
    input_path = forms.CharField(required=False, help_text="A local image folder, image file, or video path.")
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
    endpoint_port = forms.IntegerField(min_value=1, max_value=65535)
    prompt = forms.CharField(widget=forms.Textarea)
