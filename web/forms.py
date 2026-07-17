from __future__ import annotations

import ipaddress
import math
import re
import shlex
from typing import Any, ClassVar

from django import forms

from core.config import (
    ConfiguredService,
    HostListenerConfig,
    NodeConfig,
    PipelineConfig,
)
from core.errors import ConfigurationError
from core.services.llm_settings import (
    CACHE_TYPE_CHOICES,
    RETIRED_SOURCE_KEYS,
    parse_additional_server_arguments,
    resolve_inference_bind_host,
    validate_gguf_pattern,
    validate_hf_repository,
    validate_llm_settings,
)
from core.services.specs import ServiceSpec, ServiceType


# Do not expose the empty "backend default" choice in the normal UI.
# Almost ARCADIA should save explicit, predictable cache types.
LLM_CACHE_TYPE_CHOICES = tuple(
    (value, label)
    for value, label in CACHE_TYPE_CHOICES
    if value
)


DEFAULT_CONTEXT_SIZE = 32768
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_K = 20
DEFAULT_TOP_P = 0.9
DEFAULT_MIN_P = 0.05

DEFAULT_GPU_LAYERS: str = "all"
DEFAULT_BATCH_SIZE = 2048
DEFAULT_MICROBATCH_SIZE = 512
DEFAULT_FLASH_ATTENTION = "auto"
DEFAULT_CACHE_TYPE = "f16"

DEFAULT_MAX_TOKENS = 1024
DEFAULT_REPEAT_PENALTY = 1.0
DEFAULT_PRESENCE_PENALTY = 0.0
DEFAULT_FREQUENCY_PENALTY = 0.0

DEFAULT_DRAFT_METHOD = "draft-simple"
DEFAULT_DRAFT_MAX_TOKENS = 3
DEFAULT_DRAFT_MIN_PROBABILITY = 0.75
DEFAULT_DRAFT_CACHE_TYPE = "f16"


def _require_finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise forms.ValidationError(f"{label} must be a finite number.")
    return value


class HostListenerForm(forms.Form):
    host = forms.CharField(
        label="IP address",
        initial="127.0.0.1",
    )
    port = forms.IntegerField(
        label="Instruction port",
        min_value=1,
        max_value=65535,
        initial=9000,
    )

    def clean_host(self) -> str:
        value = self.cleaned_data["host"]

        try:
            return HostListenerConfig(host=value).host
        except ConfigurationError as exc:
            raise forms.ValidationError(str(exc)) from exc

    def to_config(self) -> HostListenerConfig:
        return HostListenerConfig(
            host=self.cleaned_data["host"],
            port=self.cleaned_data["port"],
        )


class RemoteNodeForm(forms.Form):
    name = forms.CharField(
        label="Name",
        max_length=63,
    )
    host = forms.CharField(
        label="Instruction-server IP",
        max_length=255,
    )
    instruction_port = forms.IntegerField(
        label="Instruction port",
        min_value=1,
        max_value=65535,
    )

    def clean_name(self) -> str:
        value = re.sub(
            r"\s+",
            "-",
            self.cleaned_data["name"].strip().lower(),
        )

        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,62}", value):
            raise forms.ValidationError(
                "Use 1–63 letters, numbers, spaces, hyphens, or "
                "underscores; start with a letter or number."
            )

        if value == "local":
            raise forms.ValidationError(
                "'local' is reserved for this computer."
            )

        return value

    def clean_host(self) -> str:
        value = self.cleaned_data["host"].strip()

        try:
            address = ipaddress.ip_address(value)
        except ValueError as exc:
            raise forms.ValidationError(
                "IP address must be a valid IPv4 address."
            ) from exc

        if address.version != 4:
            raise forms.ValidationError(
                "IP address must be a valid IPv4 address."
            )

        return str(address)

    def to_config(
        self,
        *,
        extra: dict[str, Any] | None = None,
    ) -> NodeConfig:
        return NodeConfig(
            mode="remote",
            host=self.cleaned_data["host"],
            instruction_port=self.cleaned_data["instruction_port"],
            extra=extra or {},
        )


class ServiceForm(forms.Form):
    node = forms.ChoiceField(
        label="Run on",
    )
    inference_port = forms.IntegerField(
        label="Inference port",
        min_value=1,
        max_value=65535,
    )
    bind_host = forms.CharField(
        label="Bind host",
        initial="0.0.0.0",
    )
    startup_timeout = forms.FloatField(
        label="Startup timeout",
        min_value=1,
        initial=600,
        help_text=(
            "Maximum seconds to wait for the model server to become ready."
        ),
    )

    def __init__(
        self,
        *args: Any,
        nodes: dict[str, NodeConfig] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        node_map = nodes or {
            "local": NodeConfig("local", "127.0.0.1"),
        }

        ordered_nodes = sorted(
            node_map.items(),
            key=lambda item: (
                item[0] != "local",
                item[0].lower(),
            ),
        )

        self.fields["node"].choices = [
            (
                name,
                (
                    "This computer"
                    if name == "local"
                    else f"{name} ({node.host})"
                ),
            )
            for name, node in ordered_nodes
        ]

    def clean_bind_host(self) -> str:
        value = self.cleaned_data["bind_host"].strip()

        try:
            address = ipaddress.ip_address(value)
        except ValueError as exc:
            raise forms.ValidationError(
                "Bind host must be a valid IPv4 address."
            ) from exc

        if address.version != 4:
            raise forms.ValidationError(
                "Bind host must be a valid IPv4 address."
            )

        return str(address)

    def initial_from(
        self,
        configured: ConfiguredService | None,
    ) -> None:
        if configured is None:
            return

        settings = configured.settings

        self.initial.update(
            {
                "node": configured.node,
                "inference_port": configured.port,
                "bind_host": settings.get(
                    "bind_host",
                    "0.0.0.0",
                ),
                "startup_timeout": settings.get(
                    "startup_timeout",
                    600,
                ),
            }
        )


class LLMServiceForm(ServiceForm):
    service_type: ClassVar[ServiceType] = "llm"
    default_model_alias: ClassVar[str] = "logical-model"
    default_inference_port: ClassVar[int] = 8081

    inference_port = forms.IntegerField(
        label="Inference port",
        min_value=1,
        max_value=65535,
        initial=8081,
    )

    # Quick settings
    hf_repo = forms.CharField(
        label="Hugging Face model repository",
        required=True,
        help_text=(
            "Format: owner/repository. "
            "Example: bartowski/Qwen2.5-7B-Instruct-GGUF"
        ),
    )
    n_ctx = forms.IntegerField(
        label="Context size",
        min_value=1,
        initial=DEFAULT_CONTEXT_SIZE,
    )
    vision_enabled = forms.BooleanField(
        label="Enable vision",
        required=False,
    )
    mmproj_repo = forms.CharField(
        label="Projector repository",
        required=False,
        help_text=(
            "Format: owner/repository. "
            "Leave blank to use the model repository."
        ),
    )
    temperature = forms.FloatField(
        label="Temperature",
        min_value=0,
        initial=DEFAULT_TEMPERATURE,
    )
    top_k = forms.IntegerField(
        label="Top K",
        min_value=0,
        initial=DEFAULT_TOP_K,
    )
    top_p = forms.FloatField(
        label="Top P",
        min_value=0,
        max_value=1,
        initial=DEFAULT_TOP_P,
    )
    min_p = forms.FloatField(
        label="Min P",
        min_value=0,
        max_value=1,
        initial=DEFAULT_MIN_P,
    )

    # Advanced: model selection
    model_file_pattern = forms.CharField(
        label="Model file pattern",
        required=False,
        help_text=(
            "Optional basename or glob ending in .gguf. "
            "Matching is case-insensitive."
        ),
    )
    model_alias = forms.CharField(
        label="Model alias",
        required=False,
        initial="logical-model",
        help_text=(
            "Stable model name used by the OpenAI-compatible endpoint."
        ),
    )
    chat_format = forms.CharField(
        label="Chat format / template",
        required=False,
        help_text=(
            "Leave blank to use the chat template stored in GGUF metadata."
        ),
    )

    # Advanced: projector
    mmproj_file_pattern = forms.CharField(
        label="Projector file pattern",
        required=False,
        help_text=(
            "Optional basename or glob ending in .gguf."
        ),
    )

    # Advanced: draft model
    draft_enabled = forms.BooleanField(
        label="Enable draft model",
        required=False,
    )
    draft_repo = forms.CharField(
        label="Draft model repository",
        required=False,
        help_text=(
            "Format: owner/repository. "
            "Leave blank to use the main model repository."
        ),
    )
    draft_file_pattern = forms.CharField(
        label="Draft file pattern",
        required=False,
        help_text=(
            "Optional basename or glob ending in .gguf."
        ),
    )
    draft_method = forms.ChoiceField(
        label="Draft method",
        choices=[
            ("draft-simple", "draft-simple"),
            ("draft-mtp", "draft-mtp"),
            ("draft-eagle3", "draft-eagle3"),
        ],
        initial=DEFAULT_DRAFT_METHOD,
        required=False,
    )
    draft_max_tokens = forms.IntegerField(
        label="Draft max tokens",
        min_value=1,
        initial=DEFAULT_DRAFT_MAX_TOKENS,
        required=False,
    )
    draft_min_prob = forms.FloatField(
        label="Draft min probability",
        min_value=0,
        max_value=1,
        initial=DEFAULT_DRAFT_MIN_PROBABILITY,
        required=False,
    )
    draft_cache_type_k = forms.ChoiceField(
        label="Draft K-cache type",
        choices=LLM_CACHE_TYPE_CHOICES,
        initial=DEFAULT_DRAFT_CACHE_TYPE,
        required=False,
    )
    draft_cache_type_v = forms.ChoiceField(
        label="Draft V-cache type",
        choices=LLM_CACHE_TYPE_CHOICES,
        initial=DEFAULT_DRAFT_CACHE_TYPE,
        required=False,
    )

    # Advanced: compute
    n_gpu_layers = forms.CharField(
        label="GPU layers",
        required=False,
        initial=DEFAULT_GPU_LAYERS,
        help_text=(
            "Use auto, all, -1, or a non-negative integer."
        ),
    )
    n_threads = forms.IntegerField(
        label="CPU threads",
        min_value=1,
        required=False,
        help_text=(
            "Leave blank to use llama-server's automatic thread count."
        ),
    )
    n_batch = forms.IntegerField(
        label="Batch size",
        min_value=1,
        initial=DEFAULT_BATCH_SIZE,
        required=False,
    )
    n_ubatch = forms.IntegerField(
        label="Microbatch size",
        min_value=1,
        initial=DEFAULT_MICROBATCH_SIZE,
        required=False,
    )
    flash_attn = forms.ChoiceField(
        label="Flash attention",
        choices=[
            ("auto", "Auto"),
            ("on", "On"),
            ("off", "Off"),
        ],
        initial=DEFAULT_FLASH_ATTENTION,
        required=False,
    )

    # Advanced: memory
    cache_type_k = forms.ChoiceField(
        label="K-cache type",
        choices=LLM_CACHE_TYPE_CHOICES,
        initial=DEFAULT_CACHE_TYPE,
        required=False,
    )
    cache_type_v = forms.ChoiceField(
        label="V-cache type",
        choices=LLM_CACHE_TYPE_CHOICES,
        initial=DEFAULT_CACHE_TYPE,
        required=False,
    )
    use_mmap = forms.BooleanField(
        label="Use mmap",
        required=False,
        initial=True,
    )
    use_mlock = forms.BooleanField(
        label="Use mlock",
        required=False,
    )

    # Advanced: generation
    max_tokens = forms.IntegerField(
        label="Max output tokens",
        min_value=1,
        initial=DEFAULT_MAX_TOKENS,
        required=False,
    )
    repeat_penalty = forms.FloatField(
        label="Repeat penalty",
        initial=DEFAULT_REPEAT_PENALTY,
        required=False,
        help_text="Must be greater than zero.",
    )
    presence_penalty = forms.FloatField(
        label="Presence penalty",
        initial=DEFAULT_PRESENCE_PENALTY,
        required=False,
    )
    frequency_penalty = forms.FloatField(
        label="Frequency penalty",
        initial=DEFAULT_FREQUENCY_PENALTY,
        required=False,
    )
    seed = forms.IntegerField(
        label="Seed",
        required=False,
        help_text=(
            "Leave blank to use the backend's random/default seed."
        ),
    )

    # Advanced: networking
    local_bind_host = forms.CharField(
        label="Local bind host",
        required=False,
        initial="127.0.0.1",
    )

    # Advanced: expert
    additional_arguments = forms.CharField(
        label="Additional arguments",
        required=False,
        widget=forms.Textarea(
            attrs={
                "rows": 3,
                "spellcheck": "false",
            }
        ),
        help_text=(
            "Arguments not controlled above. "
            "Shell syntax and owned options are rejected."
        ),
    )

    def __init__(
        self,
        *args: Any,
        nodes: dict[str, NodeConfig] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            nodes=nodes,
            **kwargs,
        )

        self.nodes = nodes or {
            "local": NodeConfig("local", "127.0.0.1"),
        }

        # LLM timeouts are application-owned, not editable settings.
        self.fields.pop("bind_host")
        self.fields.pop("startup_timeout")

        self.fields["inference_port"].initial = (
            self.default_inference_port
        )
        self.fields["model_alias"].initial = (
            self.default_model_alias
        )

        self.initial.setdefault(
            "inference_port",
            self.default_inference_port,
        )
        self.initial.setdefault(
            "model_alias",
            self.default_model_alias,
        )

        self._prior_settings: dict[str, Any] = {}
        self.legacy_local_model = False

    def clean_hf_repo(self) -> str:
        try:
            return validate_hf_repository(
                self.cleaned_data["hf_repo"]
            )
        except ValueError as exc:
            raise forms.ValidationError(str(exc)) from exc

    def clean_mmproj_repo(self) -> str:
        value = self.cleaned_data.get(
            "mmproj_repo",
            "",
        ).strip()

        if not value:
            return ""

        try:
            return validate_hf_repository(value)
        except ValueError as exc:
            raise forms.ValidationError(str(exc)) from exc

    def clean_draft_repo(self) -> str:
        value = self.cleaned_data.get(
            "draft_repo",
            "",
        ).strip()

        if not value:
            return ""

        try:
            return validate_hf_repository(value)
        except ValueError as exc:
            raise forms.ValidationError(str(exc)) from exc

    def clean_model_file_pattern(self) -> str | None:
        try:
            return validate_gguf_pattern(
                self.cleaned_data.get("model_file_pattern")
            )
        except ValueError as exc:
            raise forms.ValidationError(str(exc)) from exc

    def clean_mmproj_file_pattern(self) -> str | None:
        try:
            return validate_gguf_pattern(
                self.cleaned_data.get("mmproj_file_pattern")
            )
        except ValueError as exc:
            raise forms.ValidationError(str(exc)) from exc

    def clean_draft_file_pattern(self) -> str | None:
        try:
            return validate_gguf_pattern(
                self.cleaned_data.get("draft_file_pattern")
            )
        except ValueError as exc:
            raise forms.ValidationError(str(exc)) from exc

    def clean_model_alias(self) -> str:
        value = self.cleaned_data.get(
            "model_alias",
            "",
        ).strip()

        return value or self.default_model_alias

    def clean_chat_format(self) -> str:
        return self.cleaned_data.get(
            "chat_format",
            "",
        ).strip()

    def clean_temperature(self) -> float:
        return _require_finite(
            self.cleaned_data["temperature"],
            "Temperature",
        )

    def clean_top_p(self) -> float:
        return _require_finite(
            self.cleaned_data["top_p"],
            "Top P",
        )

    def clean_min_p(self) -> float:
        return _require_finite(
            self.cleaned_data["min_p"],
            "Min P",
        )

    def clean_draft_min_prob(self) -> float | None:
        value = self.cleaned_data.get("draft_min_prob")

        if value is None:
            return None

        return _require_finite(
            value,
            "Draft minimum probability",
        )

    def clean_repeat_penalty(self) -> float | None:
        value = self.cleaned_data.get("repeat_penalty")

        if value is None:
            return None

        value = _require_finite(
            value,
            "Repeat penalty",
        )

        if value <= 0:
            raise forms.ValidationError(
                "Repeat penalty must be greater than zero."
            )

        return value

    def clean_presence_penalty(self) -> float | None:
        value = self.cleaned_data.get("presence_penalty")

        if value is None:
            return None

        return _require_finite(
            value,
            "Presence penalty",
        )

    def clean_frequency_penalty(self) -> float | None:
        value = self.cleaned_data.get("frequency_penalty")

        if value is None:
            return None

        return _require_finite(
            value,
            "Frequency penalty",
        )

    def clean_additional_arguments(self) -> list[str]:
        try:
            return parse_additional_server_arguments(
                self.cleaned_data.get(
                    "additional_arguments",
                    "",
                )
            )
        except ValueError as exc:
            raise forms.ValidationError(str(exc)) from exc

    def clean_n_gpu_layers(self) -> int | str:
        value = self.cleaned_data.get("n_gpu_layers")

        if value is None or value == "":
            return DEFAULT_GPU_LAYERS

        if isinstance(value, str):
            normalized = value.strip().lower()

            if normalized in ("auto", "all"):
                return normalized

            if normalized == "-1":
                return "all"

            try:
                integer_value = int(normalized)
            except ValueError:
                integer_value = -2

            if integer_value >= 0:
                return integer_value

        if isinstance(value, int):
            if value >= 0:
                return value

            if value == -1:
                return "all"

        raise forms.ValidationError(
            "GPU layers must be 'auto', 'all', "
            "or a non-negative integer."
        )

    def clean_local_bind_host(self) -> str:
        node_name = self.cleaned_data.get("node")

        # Remote bind hosts are derived from the selected node.
        if node_name != "local":
            return ""

        value = (
            self.cleaned_data.get("local_bind_host")
            or "127.0.0.1"
        ).strip()

        try:
            address = ipaddress.ip_address(value)
        except ValueError as exc:
            raise forms.ValidationError(
                "Local bind host must be a valid IPv4 address."
            ) from exc

        if address.version != 4:
            raise forms.ValidationError(
                "Local bind host must be a valid IPv4 address."
            )

        return str(address)

    def clean(self) -> dict[str, Any]:
        cleaned = super().clean()

        batch_size = cleaned.get("n_batch")
        microbatch_size = cleaned.get("n_ubatch")

        if (
            batch_size is not None
            and microbatch_size is not None
            and batch_size < microbatch_size
        ):
            self.add_error(
                "n_batch",
                "Batch size must be at least the microbatch size.",
            )

        return cleaned

    def _build_settings(self) -> dict[str, Any]:
        cleaned = self.cleaned_data

        excluded_prior_keys = (
            set(RETIRED_SOURCE_KEYS)
            | {
                "startup_timeout",
                "local_bind_host",
            }
        )

        settings = {
            key: value
            for key, value in self._prior_settings.items()
            if key not in excluded_prior_keys
        }

        bind_host = resolve_inference_bind_host(
            cleaned["node"],
            self.nodes,
            cleaned.get("local_bind_host"),
        )

        settings.update(
            {
                "hf_repo": cleaned["hf_repo"],
                "models_cache_subdir": "huggingface",
                "bind_host": bind_host,
                "n_ctx": cleaned["n_ctx"],
                "temperature": cleaned["temperature"],
                "top_k": cleaned["top_k"],
                "top_p": cleaned["top_p"],
                "min_p": cleaned["min_p"],
                "model_alias": (
                    cleaned.get("model_alias")
                    or self.default_model_alias
                ),
                "n_gpu_layers": cleaned.get(
                    "n_gpu_layers",
                    DEFAULT_GPU_LAYERS,
                ),
                "n_batch": (
                    cleaned.get("n_batch")
                    if cleaned.get("n_batch") is not None
                    else DEFAULT_BATCH_SIZE
                ),
                "n_ubatch": (
                    cleaned.get("n_ubatch")
                    if cleaned.get("n_ubatch") is not None
                    else DEFAULT_MICROBATCH_SIZE
                ),
                "flash_attn": (
                    cleaned.get("flash_attn")
                    or DEFAULT_FLASH_ATTENTION
                ),
                "cache_type_k": (
                    cleaned.get("cache_type_k")
                    or DEFAULT_CACHE_TYPE
                ),
                "cache_type_v": (
                    cleaned.get("cache_type_v")
                    or DEFAULT_CACHE_TYPE
                ),
                "use_mmap": bool(
                    cleaned.get("use_mmap")
                ),
                "use_mlock": bool(
                    cleaned.get("use_mlock")
                ),
                "vision_enabled": bool(
                    cleaned.get("vision_enabled")
                ),
                "draft_enabled": bool(
                    cleaned.get("draft_enabled")
                ),
                "draft_method": (
                    cleaned.get("draft_method")
                    or DEFAULT_DRAFT_METHOD
                ),
                "draft_max_tokens": (
                    cleaned.get("draft_max_tokens")
                    if cleaned.get("draft_max_tokens") is not None
                    else DEFAULT_DRAFT_MAX_TOKENS
                ),
                "draft_min_prob": (
                    cleaned.get("draft_min_prob")
                    if cleaned.get("draft_min_prob") is not None
                    else DEFAULT_DRAFT_MIN_PROBABILITY
                ),
                "draft_cache_type_k": (
                    cleaned.get("draft_cache_type_k")
                    or DEFAULT_DRAFT_CACHE_TYPE
                ),
                "draft_cache_type_v": (
                    cleaned.get("draft_cache_type_v")
                    or DEFAULT_DRAFT_CACHE_TYPE
                ),
                "max_tokens": (
                    cleaned.get("max_tokens")
                    if cleaned.get("max_tokens") is not None
                    else DEFAULT_MAX_TOKENS
                ),
                "repeat_penalty": (
                    cleaned.get("repeat_penalty")
                    if cleaned.get("repeat_penalty") is not None
                    else DEFAULT_REPEAT_PENALTY
                ),
                "presence_penalty": (
                    cleaned.get("presence_penalty")
                    if cleaned.get("presence_penalty") is not None
                    else DEFAULT_PRESENCE_PENALTY
                ),
                "frequency_penalty": (
                    cleaned.get("frequency_penalty")
                    if cleaned.get("frequency_penalty") is not None
                    else DEFAULT_FREQUENCY_PENALTY
                ),
            }
        )

        optional_string_settings = (
            "model_file_pattern",
            "chat_format",
            "mmproj_repo",
            "mmproj_file_pattern",
            "draft_repo",
            "draft_file_pattern",
        )

        for key in optional_string_settings:
            value = cleaned.get(key)

            if isinstance(value, str):
                value = value.strip()

            if value:
                settings[key] = value
            else:
                settings.pop(key, None)

        thread_count = cleaned.get("n_threads")

        if thread_count is not None:
            settings["n_threads"] = thread_count
        else:
            settings.pop("n_threads", None)

        seed = cleaned.get("seed")

        if seed is not None:
            settings["seed"] = seed
        else:
            settings.pop("seed", None)

        additional_arguments = cleaned.get(
            "additional_arguments",
            [],
        )

        if additional_arguments:
            settings["extra_args"] = list(
                additional_arguments
            )
        else:
            settings.pop("extra_args", None)

        # Central backend validation provides a second layer after
        # Django field validation and keeps saved settings normalized.
        return validate_llm_settings(
            settings,
            remote=False,
        )

    def to_spec(self) -> ServiceSpec:
        settings = self._build_settings()

        return ServiceSpec(
            service_type=self.service_type,
            port=self.cleaned_data["inference_port"],
            settings=settings,
        )

    def initial_from(
        self,
        configured: ConfiguredService | None,
    ) -> None:
        if configured is None:
            return

        settings = configured.settings
        self._prior_settings = dict(settings)

        self.initial.update(
            {
                "node": configured.node,
                "inference_port": configured.port,
                "model_alias": settings.get(
                    "model_alias",
                    self.default_model_alias,
                ),
                "local_bind_host": settings.get(
                    "bind_host",
                    "127.0.0.1",
                ),
            }
        )

        if settings.get("model_path"):
            self.legacy_local_model = True

        initial_keys = (
            "hf_repo",
            "model_file_pattern",
            "chat_format",
            "mmproj_repo",
            "mmproj_file_pattern",
            "n_ctx",
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "n_gpu_layers",
            "n_threads",
            "n_batch",
            "n_ubatch",
            "flash_attn",
            "cache_type_k",
            "cache_type_v",
            "max_tokens",
            "repeat_penalty",
            "presence_penalty",
            "frequency_penalty",
            "seed",
            "draft_repo",
            "draft_file_pattern",
            "draft_method",
            "draft_max_tokens",
            "draft_min_prob",
            "draft_cache_type_k",
            "draft_cache_type_v",
        )

        for key in initial_keys:
            if key in settings:
                self.initial[key] = settings[key]

        self.initial["vision_enabled"] = bool(
            settings.get("vision_enabled", False)
        )
        self.initial["use_mmap"] = bool(
            settings.get("use_mmap", True)
        )
        self.initial["use_mlock"] = bool(
            settings.get("use_mlock", False)
        )
        self.initial["draft_enabled"] = bool(
            settings.get("draft_enabled", False)
        )

        extra_args = settings.get("extra_args")

        if (
            isinstance(extra_args, list)
            and all(
                isinstance(item, str)
                for item in extra_args
            )
        ):
            self.initial["additional_arguments"] = shlex.join(
                extra_args
            )

        # Legacy migration support.
        if (
            settings.get("hf_repo")
            and settings.get("hf_file")
            and not settings.get("model_file_pattern")
        ):
            self.initial["model_file_pattern"] = settings[
                "hf_file"
            ]


class VisualLLMServiceForm(LLMServiceForm):
    """Separate Visual LLM settings with vision forced on."""

    service_type: ClassVar[ServiceType] = "visual_llm"
    default_model_alias: ClassVar[str] = "visual-model"
    default_inference_port: ClassVar[int] = 8082

    vision_enabled = forms.BooleanField(
        label="Enable vision",
        required=False,
        initial=True,
        disabled=True,
        help_text="Vision is always enabled for Visual LLM.",
    )
    model_alias = forms.CharField(
        label="Model alias",
        required=False,
        initial="visual-model",
        help_text=(
            "Stable model name used by the OpenAI-compatible endpoint."
        ),
    )
    inference_port = forms.IntegerField(
        label="Inference port",
        min_value=1,
        max_value=65535,
        initial=8082,
    )

    def __init__(
        self,
        *args: Any,
        nodes: dict[str, NodeConfig] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            nodes=nodes,
            **kwargs,
        )

        self.fields["vision_enabled"].disabled = True
        self.fields["vision_enabled"].initial = True

        self.initial.setdefault(
            "vision_enabled",
            True,
        )
        self.initial.setdefault(
            "model_alias",
            self.default_model_alias,
        )
        self.initial.setdefault(
            "inference_port",
            self.default_inference_port,
        )

    def clean(self) -> dict[str, Any]:
        cleaned = super().clean()
        cleaned["vision_enabled"] = True
        return cleaned

    def initial_from(
        self,
        configured: ConfiguredService | None,
    ) -> None:
        super().initial_from(configured)

        # A separately configured Visual role is always multimodal.
        self.initial["vision_enabled"] = True

        if not self.initial.get("model_alias"):
            self.initial["model_alias"] = (
                self.default_model_alias
            )


class SAMServiceForm(ServiceForm):
    _OWNED_FLAGS = {
        "--host",
        "--port",
        "--checkpoint",
        "--confidence",
    }

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

    inference_port = forms.IntegerField(
        label="Inference port",
        min_value=1,
        max_value=65535,
        initial=8090,
    )
    checkpoint = forms.CharField(
        label="Checkpoint path",
    )
    confidence = forms.FloatField(
        label="Default confidence",
        min_value=0,
        max_value=1,
        initial=0.25,
    )
    additional_arguments = forms.CharField(
        label="Additional arguments",
        required=False,
        widget=forms.Textarea(
            attrs={
                "rows": 3,
                "spellcheck": "false",
            }
        ),
        help_text=(
            "Optional SAM3 server arguments not covered above."
        ),
    )

    def clean_checkpoint(self) -> str:
        value = self.cleaned_data["checkpoint"].strip()

        if not value:
            raise forms.ValidationError(
                "Checkpoint path cannot be empty."
            )

        return value

    def clean_confidence(self) -> float:
        return _require_finite(
            self.cleaned_data["confidence"],
            "Default confidence",
        )

    def clean_additional_arguments(self) -> list[str]:
        try:
            arguments = shlex.split(
                self.cleaned_data.get(
                    "additional_arguments",
                    "",
                )
            )
        except ValueError as exc:
            raise forms.ValidationError(
                f"Invalid additional argument syntax: {exc}"
            ) from exc

        for argument in arguments:
            if (
                argument in self._DISALLOWED_ARGUMENTS
                or any(
                    argument == flag
                    or argument.startswith(f"{flag}=")
                    for flag in self._OWNED_FLAGS
                )
            ):
                raise forms.ValidationError(
                    "Additional arguments cannot override "
                    f"{argument!r}."
                )

            if any(
                character in argument
                for character in (
                    ";",
                    "|",
                    "&",
                    "`",
                    "$",
                    "<",
                    ">",
                )
            ):
                raise forms.ValidationError(
                    "Additional arguments cannot contain "
                    "shell-related syntax."
                )

        return arguments

    def to_spec(self) -> ServiceSpec:
        cleaned = self.cleaned_data

        settings: dict[str, Any] = {
            "checkpoint": cleaned["checkpoint"],
            "bind_host": cleaned["bind_host"],
            "startup_timeout": cleaned["startup_timeout"],
            "confidence": cleaned["confidence"],
        }

        if cleaned["additional_arguments"]:
            settings["extra_args"] = list(
                cleaned["additional_arguments"]
            )

        return ServiceSpec(
            service_type="sam3",
            port=cleaned["inference_port"],
            settings=settings,
        )

    def initial_from(
        self,
        configured: ConfiguredService | None,
    ) -> None:
        super().initial_from(configured)

        if configured is None:
            return

        settings = configured.settings
        extra_args = settings.get("extra_args")

        self.initial.update(
            {
                "checkpoint": settings.get(
                    "checkpoint",
                    "",
                ),
                "confidence": settings.get(
                    "confidence",
                    0.25,
                ),
                "additional_arguments": (
                    shlex.join(extra_args)
                    if (
                        isinstance(extra_args, list)
                        and all(
                            isinstance(argument, str)
                            for argument in extra_args
                        )
                    )
                    else ""
                ),
            }
        )


class PipelineForm(forms.Form):
    task = forms.CharField(
        initial="Find cars",
    )
    debrief = forms.CharField(
        required=False,
        widget=forms.Textarea,
    )
    prompts = forms.CharField(
        required=False,
        help_text="Comma-separated optional labels",
    )
    sam_step = forms.IntegerField(
        min_value=1,
        initial=5,
    )
    sam_confidence = forms.FloatField(
        min_value=0,
        max_value=1,
        initial=0.25,
    )
    sam_resize = forms.IntegerField(
        min_value=1,
        required=False,
    )
    max_image_edge = forms.IntegerField(
        min_value=1,
        required=False,
        initial=640,
    )
    run_at_source_fps = forms.BooleanField(
        required=False,
    )
    debug = forms.BooleanField(
        required=False,
    )
    record = forms.BooleanField(
        required=False,
        initial=True,
    )
    panoramic = forms.BooleanField(
        required=False,
    )
    graph_agent = forms.BooleanField(
        required=False,
    )
    gps_csv = forms.CharField(
        required=False,
    )
    camera_intrinsics = forms.CharField(
        required=False,
    )
    scene_model = forms.CharField(
        required=False,
    )

    @classmethod
    def from_config(
        cls,
        config: PipelineConfig,
    ) -> PipelineForm:
        return cls(
            initial={
                "task": config.task,
                "debrief": config.debrief,
                "prompts": ", ".join(config.prompts),
                "sam_step": config.sam_step,
                "sam_confidence": config.sam_confidence,
                "sam_resize": config.sam_resize,
                "max_image_edge": config.max_image_edge,
                "run_at_source_fps": (
                    config.run_at_source_fps
                ),
                "debug": config.debug,
                "record": config.record,
                "panoramic": config.panoramic,
                "graph_agent": config.graph_agent,
                "gps_csv": config.gps_csv or "",
                "camera_intrinsics": (
                    config.camera_intrinsics or ""
                ),
                "scene_model": config.scene_model or "",
            }
        )

    def to_config(self) -> PipelineConfig:
        data = dict(self.cleaned_data)

        data["prompts"] = [
            item.strip()
            for item in data.pop(
                "prompts",
                "",
            ).split(",")
            if item.strip()
        ]

        return PipelineConfig(**data)


class AnalysisForm(forms.Form):
    input_path = forms.CharField(
        required=False,
        help_text=(
            "A local image folder, image file, or video path."
        ),
    )
    upload_id = forms.CharField(
        required=False,
    )

    def clean(self) -> dict[str, Any]:
        cleaned = super().clean()

        input_path = (
            cleaned.get("input_path")
            or ""
        ).strip()
        upload_id = (
            cleaned.get("upload_id")
            or ""
        ).strip()

        if bool(input_path) == bool(upload_id):
            raise forms.ValidationError(
                "Choose exactly one existing path "
                "or retained upload."
            )

        cleaned["input_path"] = input_path
        cleaned["upload_id"] = upload_id

        return cleaned


class EndpointTestForm(forms.Form):
    endpoint_host = forms.CharField(
        initial="127.0.0.1",
    )
    endpoint_port = forms.IntegerField(
        min_value=1,
        max_value=65535,
    )
    prompt = forms.CharField(
        widget=forms.Textarea,
    )