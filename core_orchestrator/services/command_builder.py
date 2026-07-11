"""
command_builder.py

Builds command-line argument arrays for managed model services.
Structured arguments are emitted first; raw user-supplied arguments
are appended last so that the backend's normal last-flag-wins precedence
applies.

All commands are returned as ``list[str]`` suitable for ``subprocess.Popen``
with ``shell=False``.
"""

from __future__ import annotations

import logging
import shlex
from typing import Optional

from .settings_store import LLMServiceSettings, SAMServiceSettings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM command
# ---------------------------------------------------------------------------
def build_llm_command(
    settings: LLMServiceSettings,
) -> list[str]:
    """
    Build the command array for a managed llama.cpp-style LLM server.

    Structured arguments
    --------------------
    ``executable --model <model_path> --host <host> --port <port>``
    plus ``--ctx-size``, ``--n-gpu-layers``, ``--batch-size``,
    ``--ubatch-size``, ``--threads``, ``--flash-attn``,
    ``--cache-type-k``, ``--cache-type-v``, ``--parallel`` when set.

    After structured args the raw ``settings.arguments`` list is appended.
    """
    cmd = [settings.executable]

    if settings.model_path:
        cmd.extend(["--model", settings.model_path])
    if settings.host:
        cmd.extend(["--host", settings.host])
    if settings.port:
        cmd.extend(["--port", str(settings.port)])

    cmd.extend(settings.arguments)

    return cmd


def build_sam_command(settings: SAMServiceSettings) -> list[str]:
    """
    Build the command array for a managed SAM service.

    SAM models are typically loaded via Python, not a separate binary.
    If an executable is supplied, use it; otherwise return an empty list.
    """
    if not settings.weights_path:
        return []
    cmd = [settings.weights_path] if settings.arguments else []
    cmd.extend(settings.arguments)
    return cmd


# ---------------------------------------------------------------------------
# Command preview (for UI display)
# ---------------------------------------------------------------------------
def preview_command(settings: LLMServiceSettings | SAMServiceSettings) -> str:
    """Return a human-readable command preview string."""
    if isinstance(settings, LLMServiceSettings):
        cmd = build_llm_command(settings)
    else:
        cmd = build_sam_command(settings)

    if not cmd:
        return "(no command generated — check configuration)"

    return " ".join(shlex.quote(c) for c in cmd)


def preview_command_parts(
    settings: LLMServiceSettings | SAMServiceSettings,
) -> dict:
    """Return structured command preview with separated parts."""
    if isinstance(settings, LLMServiceSettings):
        cmd = build_llm_command(settings)
        structured_args = ["--model", settings.model_path] if settings.model_path else []
        if settings.host:
            structured_args.extend(["--host", settings.host])
        if settings.port:
            structured_args.extend(["--port", str(settings.port)])
    else:
        cmd = build_sam_command(settings)
        structured_args = []

    return {
        "executable": cmd[0] if cmd else "",
        "structured_arguments": structured_args,
        "raw_arguments": settings.arguments,
        "full_command": " ".join(shlex.quote(c) for c in cmd),
        "argument_array": cmd,
    }