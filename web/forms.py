from __future__ import annotations

import re
import shlex
from collections.abc import Callable
from typing import Any

from django import forms

from core.config import ConfiguredService, NodeConfig, PipelineConfig
from core.services.specs import ServiceSpec


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
            (name, "This computer" if name == "local" else f"{name} ({node.host})")
            for name, node in node_map.items()
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
    CACHE_TYPE_CHOICES = [
        ("", "Default"),
        ("f16", "F16"),
        ("q8_0", "Q8_0"),
        ("q4_0", "Q4_0"),
    ]
    _CACHE_TYPE_FLAGS = {"f16": "1", "q8_0": "8", "q4_0": "2"}
    _SPLIT_GGUF_PATTERN = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)
    _OWNED_FLAGS = {
        "--model",
        "--model_alias",
        "--host",
        "--port",
        "--hf_repo",
        "--hf_file",
        "--n_ctx",
        "--n_gpu_layers",
        "--n_threads",
        "--n_batch",
        "--n_ubatch",
        "--flash_attn",
        "--type_k",
        "--type_v",
        "--chat_format",
    }
    _DISALLOWED_ARGUMENTS = {
        "--",
        "-c",
        "-m",
        "-h",
        "--help",
        "--command",
        "--config_file",
        "--python_executable",
        "--server_module",
        "bash",
        "cmd",
        "powershell",
        "python",
        "python3",
        "sh",
        "zsh",
    }
    _KNOWN_EXTRA_ARGS: dict[str, tuple[str, Callable[[str], Any]]] = {
        "--n_ctx": ("n_ctx", int),
        "--n_gpu_layers": ("n_gpu_layers", int),
        "--n_threads": ("n_threads", int),
        "--n_batch": ("n_batch", int),
        "--n_ubatch": ("n_ubatch", int),
        "--flash_attn": ("flash_attn", lambda value: value.lower() in {"1", "true", "yes", "on"}),
        "--type_k": ("cache_type_k", str),
        "--type_v": ("cache_type_v", str),
        "--chat_format": ("chat_format", str),
        "--model_alias": ("model_alias", str),
    }

    model_source = forms.ChoiceField(
        label="Source type",
        choices=[("local", "Local GGUF file"), ("huggingface", "Hugging Face")],
        initial="local",
    )
    model_path = forms.CharField(label="Local model path", required=False)
    hf_repo = forms.CharField(label="Hugging Face repository", required=False)
    hf_file = forms.CharField(label="Exact filename", required=False)
    hf_cache_dir = forms.CharField(label="Cache directory", required=False)
    n_ctx = forms.IntegerField(label="Context size", min_value=1, initial=32768)
    n_gpu_layers = forms.IntegerField(
        label="GPU layers",
        min_value=-1,
        initial=-1,
        help_text="Use -1 to offload all supported layers.",
    )
    n_threads = forms.IntegerField(
        label="CPU threads",
        min_value=1,
        required=False,
        help_text="Leave empty to use llama-cpp-python's default.",
    )
    n_batch = forms.IntegerField(label="Batch size", min_value=1, initial=2048)
    n_ubatch = forms.IntegerField(label="Microbatch size", min_value=1, initial=512)
    n_parallel = forms.IntegerField(
        label="Parallel slots",
        min_value=1,
        initial=1,
        help_text="Stored for compatibility; llama-cpp-python 0.3.34 has no parallel-slots server flag.",
    )
    flash_attn = forms.BooleanField(label="Flash attention", required=False, initial=True)
    cache_type_k = forms.ChoiceField(label="K-cache type", choices=CACHE_TYPE_CHOICES, required=False)
    cache_type_v = forms.ChoiceField(label="V-cache type", choices=CACHE_TYPE_CHOICES, required=False)
    chat_format = forms.CharField(label="Chat format", required=False)
    model_alias = forms.CharField(label="Model alias", initial="local-model")
    additional_arguments = forms.CharField(
        label="Additional server arguments",
        required=False,
        widget=forms.Textarea,
        help_text="Optional llama-cpp-python server arguments not covered above.",
    )

    @staticmethod
    def _parse_arguments(value: str) -> list[str]:
        try:
            return shlex.split(value)
        except ValueError as exc:
            raise forms.ValidationError(f"Invalid additional argument syntax: {exc}") from exc

    @classmethod
    def _validate_additional_arguments(cls, args: list[str]) -> None:
        for argument in args:
            if argument in cls._DISALLOWED_ARGUMENTS or any(
                argument == flag or argument.startswith(f"{flag}=") for flag in cls._OWNED_FLAGS
            ):
                raise forms.ValidationError(f"Additional arguments cannot override {argument!r}.")
            if any(character in argument for character in (";", "|", "&", "`", "$", "<", ">")):
                raise forms.ValidationError("Additional arguments cannot contain shell-related syntax.")

    def clean_additional_arguments(self) -> list[str]:
        args = self._parse_arguments(self.cleaned_data.get("additional_arguments", ""))
        self._validate_additional_arguments(args)
        return args

    def clean(self) -> dict[str, Any]:
        cleaned = super().clean()
        if cleaned.get("n_batch") is not None and cleaned.get("n_ubatch") is not None:
            if cleaned["n_batch"] < cleaned["n_ubatch"]:
                self.add_error("n_batch", "Batch size must be at least the microbatch size.")
        source = cleaned.get("model_source")
        if source == "local":
            model_path = cleaned.get("model_path", "").strip()
            if not model_path:
                self.add_error("model_path", "A local model path is required.")
            elif self._SPLIT_GGUF_PATTERN.search(model_path):
                self.add_error("model_path", "Split GGUF files are not supported.")
        if source == "huggingface":
            hf_file = cleaned.get("hf_file", "").strip()
            if not cleaned.get("hf_repo", "").strip():
                self.add_error("hf_repo", "A Hugging Face repository is required.")
            if not hf_file:
                self.add_error("hf_file", "An exact Hugging Face filename is required.")
            elif self._SPLIT_GGUF_PATTERN.search(hf_file):
                self.add_error("hf_file", "Split GGUF files are not supported.")
        return cleaned

    def to_spec(self) -> ServiceSpec:
        cleaned = self.cleaned_data
        settings: dict[str, Any] = {
            "bind_host": cleaned["bind_host"],
            "startup_timeout": cleaned["startup_timeout"],
            "n_ctx": cleaned["n_ctx"],
            "n_gpu_layers": cleaned["n_gpu_layers"],
            "n_parallel": cleaned["n_parallel"],
            "model_alias": cleaned["model_alias"],
        }
        if cleaned["model_source"] == "local":
            settings["model_path"] = cleaned["model_path"].strip()
        else:
            settings["hf_repo"] = cleaned["hf_repo"].strip()
            settings["hf_file"] = cleaned["hf_file"].strip()
            if cleaned["hf_cache_dir"].strip():
                settings["hf_cache_dir"] = cleaned["hf_cache_dir"].strip()
        if cleaned.get("chat_format", "").strip():
            settings["chat_format"] = cleaned["chat_format"].strip()
        extra_args = list(cleaned["additional_arguments"])
        for field_name, flag in (
            ("n_threads", "--n_threads"),
            ("n_batch", "--n_batch"),
            ("n_ubatch", "--n_ubatch"),
            ("cache_type_k", "--type_k"),
            ("cache_type_v", "--type_v"),
        ):
            value = cleaned.get(field_name)
            if value not in (None, ""):
                if field_name in ("cache_type_k", "cache_type_v"):
                    value = self._CACHE_TYPE_FLAGS[value]
                extra_args.extend([flag, str(value)])
        extra_args.extend(["--flash_attn", str(cleaned["flash_attn"]).lower()])
        if extra_args:
            settings["extra_args"] = extra_args
        return ServiceSpec(service_type="llm", port=cleaned["inference_port"], settings=settings)

    @classmethod
    def _extract_known_extra_args(cls, args: list[str]) -> tuple[dict[str, Any], list[str]]:
        initial: dict[str, Any] = {}
        unknown: list[str] = []
        index = 0
        while index < len(args):
            argument = args[index]
            flag, separator, value = argument.partition("=")
            known = cls._KNOWN_EXTRA_ARGS.get(flag)
            if known is None:
                unknown.append(argument)
                index += 1
                continue
            if not separator:
                if index + 1 >= len(args):
                    unknown.append(argument)
                    index += 1
                    continue
                value = args[index + 1]
                index += 2
            else:
                index += 1
            field_name, converter = known
            try:
                parsed_value = converter(value)
                if field_name in ("cache_type_k", "cache_type_v"):
                    parsed_value = next(
                        choice for choice, flag_value in cls._CACHE_TYPE_FLAGS.items() if flag_value == parsed_value
                    )
                initial[field_name] = parsed_value
            except (StopIteration, TypeError, ValueError):
                unknown.extend([argument] if separator else [argument, value])
        return initial, unknown

    def initial_from(self, configured: ConfiguredService | None) -> None:
        super().initial_from(configured)
        if configured is None:
            return
        settings = configured.settings
        initial: dict[str, Any] = {
            name: settings[name]
            for name in (
                "n_ctx",
                "n_gpu_layers",
                "n_threads",
                "n_batch",
                "n_ubatch",
                "n_parallel",
                "flash_attn",
                "cache_type_k",
                "cache_type_v",
                "chat_format",
                "model_alias",
            )
            if name in settings
        }
        if settings.get("model_path"):
            initial.update({"model_source": "local", "model_path": settings["model_path"]})
        elif settings.get("hf_repo") or settings.get("hf_file"):
            initial.update(
                {
                    "model_source": "huggingface",
                    "hf_repo": settings.get("hf_repo", ""),
                    "hf_file": settings.get("hf_file", ""),
                    "hf_cache_dir": settings.get("hf_cache_dir", ""),
                }
            )
        extra_args = settings.get("extra_args", [])
        if isinstance(extra_args, list) and all(isinstance(argument, str) for argument in extra_args):
            known_initial, unknown = self._extract_known_extra_args(extra_args)
            initial.update(known_initial)
            initial["additional_arguments"] = shlex.join(unknown)
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
    input_path = forms.CharField(help_text="A local image folder, image file, or video path.")


class EndpointTestForm(forms.Form):
    endpoint_host = forms.CharField(initial="127.0.0.1")
    endpoint_port = forms.IntegerField(min_value=1, max_value=65535)
    prompt = forms.CharField(widget=forms.Textarea)
